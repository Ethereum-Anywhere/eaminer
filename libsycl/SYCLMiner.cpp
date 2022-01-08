/* Copyright (C) 1883 Thomas Edison - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3 license, which unfortunately won't be
 * written for another century.
 *
 * You should have received a copy of the LICENSE file with
 * this file.
 */

#include "include_sycl.h"

#include <ethash/ethash.hpp>
#include <libeth/Farm.h>

#include "SYCLMiner.h"
#include "ethash_miner_kernels.hpp"

using namespace dev;
using namespace eth;
using namespace std::string_literals;

/**
 *
 * @return
 */
const std::vector<sycl::device>& get_platform_devices() {
    static std::vector<sycl::device> all_devices = []() {
        auto devices = sycl::device::get_devices();
        auto out = std::vector<sycl::device>{};
        for (auto& dev: devices) {
            /* Splitting CPUs into Numa nodes so each note will have its own DAG */
            if (dev.is_cpu()) {
                try {
                    auto numa_nodes = dev.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);
                    for (auto& numa_node: numa_nodes) { out.emplace_back(numa_node); }
                } catch (...) { out.emplace_back(dev); }
            } else if (dev.is_host()) {   // Ignoring host device
                continue;
            } else {
                out.emplace_back(dev);
            }
        }
        return out;
    }();
    return all_devices;
}


struct SYCLMiner::sycl_impl {
    //    sycl_impl(sycl_impl) = delete;

public:
    sycl::queue q{sycl::default_selector{}};
    uint32_t* d_kill_signal_host = nullptr;
    hash128_t* d_dag_global = nullptr;
    hash64_t* d_light_global = nullptr;
    hash32_t d_header_global{};
    uint64_t d_target_global{};

    sycl_device_task new_search_task{};
    sycl_device_task previous_search_task{};
};


/**
 *
 * @param _index
 * @param _device
 */
SYCLMiner::SYCLMiner(unsigned _index, DeviceDescriptor& _device) : Miner("SYCL-", _index), impl(new SYCLMiner::sycl_impl) {
    impl->q = sycl::queue{get_platform_devices()[_device.sycl_device_idx]};   //, sycl::property::queue::enable_profiling{}};
    m_deviceDescriptor = _device;
    m_block_multiple = 1024;
}

/**
 *
 */
SYCLMiner::~SYCLMiner() {
    stopWorking();
    kick_miner();
}

/**
 *
 * @return
 */
bool SYCLMiner::initDevice() {
    cextr << "Using SYCL device: " << impl->q.get_device().get_info<sycl::info::device::name>()
          << ") Memory : " << dev::getFormattedMemory(static_cast<double>(m_deviceDescriptor.totalMemory));


    m_deviceDescriptor.sycl_work_items_gen_kernel = sycl_max_work_items<sycl_ethash_calculate_dag_item_kernel_tag>(impl->q);
    m_deviceDescriptor.sycl_work_items_search_kernel = sycl_max_work_items<sycl_ethash_search_kernel_tag>(impl->q);
    // Set Hardware Monitor Info
    m_hwmoninfo.deviceType = HwMonitorInfoType::UNKNOWN;
    m_hwmoninfo.devicePciId = impl->q.get_device().get_info<sycl::info::device::name>();
    m_hwmoninfo.deviceIndex = -1;   // Will be later on mapped by nvml (see Farm() constructor)

    try {
        reset_device();
        impl->q.single_task([]() { (void) nullptr; }).wait_and_throw();
    } catch (std::exception& ec) {
        cnote << "Could not set SYCL device: " << m_deviceDescriptor.uniqueId << " Error : " << ec.what();
        cnote << "Mining aborted on this device.";
        return false;
    }
    return true;
}

/**
 *
 */
void SYCLMiner::reset_device() noexcept {
    try {
        impl->q.wait_and_throw();
    } catch (const std::exception& e) { cwarn << "Caught exception " << e.what() << " while resetting: " << impl->q.get_device().get_info<sycl::info::device::name>(); }

    if (impl->d_light_global) {
        sycl::free(impl->d_light_global, impl->q);
        impl->d_light_global = nullptr;
    }

    if (impl->d_dag_global) {
        sycl::free(impl->d_dag_global, impl->q);
        impl->d_dag_global = nullptr;
    }

    if (impl->d_kill_signal_host) {
        sycl::free(impl->d_kill_signal_host, impl->q);
        impl->d_kill_signal_host = nullptr;
    }

    if (impl->previous_search_task.res) {
        sycl::free(impl->previous_search_task.res, impl->q);
        impl->previous_search_task.res = nullptr;
    }

    if (impl->new_search_task.res) {
        sycl::free(impl->new_search_task.res, impl->q);
        impl->new_search_task.res = nullptr;
    }
}

/**
 *
 * @return
 */
bool SYCLMiner::initEpoch() {
    m_initialized = false;
    // If we get here it means epoch has changed, so it's not necessary
    // to check again dag sizes. They're changed for sure

    auto startInit = std::chrono::steady_clock::now();
    size_t RequiredTotalMemory(m_epochContext.dagSize + m_epochContext.lightSize + sizeof(Search_results));
    ReportGPUMemoryRequired(m_epochContext.lightSize, m_epochContext.dagSize, sizeof(Search_results));
    try {

        // Allocate GPU buffers
        // We need to reset the device and (re)create the dag
        reset_device();

        // Check whether the current device has sufficient memory every time we recreate the dag
        if (m_deviceDescriptor.totalMemory < RequiredTotalMemory) {
            ReportGPUNoMemoryAndPause("required", RequiredTotalMemory, m_deviceDescriptor.totalMemory);
            return false;   // This will prevent to exit the thread and
                            // Eventually resume mining when changing coin or epoch (NiceHash)
        }

        // create buffer for cache
        try {
            impl->d_light_global = reinterpret_cast<hash64_t*>(sycl::malloc_device(m_epochContext.lightSize, impl->q));
        } catch (...) {
            ReportGPUNoMemoryAndPause("light cache", m_epochContext.lightSize, m_deviceDescriptor.totalMemory);
            return false;   // This will prevent to exit the thread and
        }
        try {
            impl->d_dag_global = reinterpret_cast<hash128_t*>(sycl::malloc_device(m_epochContext.dagSize, impl->q));
        } catch (...) {
            ReportGPUNoMemoryAndPause("DAG", m_epochContext.dagSize, m_deviceDescriptor.totalMemory);
            return false;   // This will prevent to exit the thread and
        }

        // Allocate and initialize kill switches to abort the kernel.
        try {
            impl->d_kill_signal_host = sycl::malloc_host<uint32_t>(1U, impl->q);
            *(impl->d_kill_signal_host) = 0;
        } catch (...) {
            ReportGPUNoMemoryAndPause("d_kill_signal", sizeof(uint32_t), m_deviceDescriptor.totalMemory);
            return false;   // This will prevent to exit the thread and
        }

        try {
            impl->new_search_task.res = sycl::malloc_device<Search_results>(1, impl->q);
            impl->previous_search_task.res = sycl::malloc_device<Search_results>(1, impl->q);
        } catch (...) {
            ReportGPUNoMemoryAndPause("mining buffer", sizeof(Search_results), m_deviceDescriptor.totalMemory);
            return false;   // This will prevent to exit the thread and
        }


        // Release the pause flag if any
        resume(MinerPauseEnum::PauseDueToInsufficientMemory);
        resume(MinerPauseEnum::PauseDueToInitEpochError);

        sycl::event light_dag_copy_evt = impl->q.memcpy(impl->d_light_global, m_epochContext.lightCache, m_epochContext.lightSize);
        auto gen_events = ethash_generate_dag(                   //
                m_epochContext.dagSize,                          //
                m_block_multiple,                                //
                m_deviceDescriptor.sycl_work_items_gen_kernel,   //
                impl->q,                                         //
                m_epochContext.dagNumItems,                      //
                m_epochContext.lightNumItems,                    //
                impl->d_dag_global,                              //
                impl->d_light_global,                            //
                light_dag_copy_evt);

        for (auto& e: gen_events) { e.wait_and_throw(); }

        ReportDAGDone(m_epochContext.dagSize,                                                                                                  //
                      uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startInit).count()),   //
                      true);
    } catch (std::exception& ec) {
        cnote << "Unexpected error " << ec.what() << " on SYCL device " << m_deviceDescriptor.uniqueId;
        cnote << "Mining suspended ...";
        pause(MinerPauseEnum::PauseDueToInitEpochError);
        return false;
    }

    m_initialized = true;
    return true;
}

/**
 *
 */
void SYCLMiner::workLoop() {
    WorkPackage last;
    last.header = h256();

    if (!initDevice()) return;

    try {
        while (!shouldStop()) {
            const WorkPackage current(work());
            if (!current) {
                m_hung_miner.store(false);
                std::unique_lock<std::mutex> l(miner_work_mutex);
                m_new_work_signal.wait_for(l, std::chrono::seconds(3));
                continue;
            }

            // Epoch change ?
            if (current.epoch != last.epoch) {
                setEpoch(current);
                if (g_seqDAG) g_seqDAGMutex.lock();
                bool b = initEpoch();
                if (g_seqDAG) g_seqDAGMutex.unlock();
                if (!b) break;
                freeCache();

                // As DAG generation takes a while we need to
                // ensure we're on latest job, not on the one
                // which triggered the epoch change
                last = current;
                continue;
            }

            // Persist most recent job.
            // Job's differences should be handled at higher level
            last = current;

            auto upper64OfBoundary = ((u64) ((u256) current.boundary >> 192)).template convert_to<uint64_t>();

            // adjust work multiplier
            float hr = RetrieveHashRate();
            if (hr >= 1e7) m_block_multiple = uint32_t((hr * target_batch_time) / (m_deviceDescriptor.sycl_work_items_search_kernel));

            // Eventually start searching
            search(current.header.data(), upper64OfBoundary, current.startNonce, current);
        }

        // Reset miner and stop working
        reset_device();
    } catch (std::exception& e) { throw std::runtime_error("GPU error: "s + e.what()); }
}


/**
 *
 */
void SYCLMiner::kick_miner() {
    std::unique_lock<std::mutex> l(m_doneMutex);
    if (!m_done) {
        m_done = true;
        if (impl->d_kill_signal_host) {
            *(impl->d_kill_signal_host) = 1;
            impl->previous_search_task.e.wait();
            impl->new_search_task.e.wait();
            *(impl->d_kill_signal_host) = 0;
        }
    }
}

/**
 *
 * @return
 */
int SYCLMiner::getNumDevices() { return (int) get_platform_devices().size(); }

/**
 *
 * @param DevicesCollection
 */
void SYCLMiner::enumDevices(minerMap& DevicesCollection) {
    int numDevices(getNumDevices());

    for (int i = 0; i < numDevices; i++) {
        DeviceDescriptor deviceDescriptor;
        try {
            deviceDescriptor.uniqueId = "[SYCL "s + std::to_string(i) + "] ";
            if (get_platform_devices()[i].is_gpu()) {
                deviceDescriptor.type = DeviceTypeEnum::Gpu;
            } else if (get_platform_devices()[i].is_accelerator()) {
                deviceDescriptor.type = DeviceTypeEnum::Accelerator;
            } else if (get_platform_devices()[i].is_cpu()) {
                deviceDescriptor.type = DeviceTypeEnum::Cpu;
            } else {
                deviceDescriptor.type = DeviceTypeEnum::Unknown;
            }
            deviceDescriptor.totalMemory = get_platform_devices()[i].get_info<sycl::info::device::global_mem_size>();
            deviceDescriptor.sycl_device_idx = i;
            deviceDescriptor.boardName = "[SYCL "s + std::to_string(i) + "] " + get_platform_devices()[i].get_info<sycl::info::device::name>();
            DevicesCollection[deviceDescriptor.uniqueId] = deviceDescriptor;
        } catch (std::exception& e) { ccrit << e.what(); }
    }
}

/**
 *
 * @param header
 * @param target
 * @param start_nonce
 * @param w
 */
void SYCLMiner::search(uint8_t const* header, uint64_t target, uint64_t start_nonce, const dev::eth::WorkPackage& w) {
    impl->d_header_global = *(reinterpret_cast<const hash32_t*>(header));
    impl->d_target_global = target;

    uint32_t batch_blocks(m_block_multiple * m_deviceDescriptor.sycl_work_items_search_kernel);

    {
        std::unique_lock<std::mutex> l(m_doneMutex);
        m_hung_miner.store(false);
        impl->new_search_task = run_ethash_search(                  //
                m_block_multiple,                                   //
                m_deviceDescriptor.sycl_work_items_search_kernel,   //
                impl->q,                                            //
                impl->new_search_task,                              //
                start_nonce,                                        //
                m_epochContext.dagNumItems,                         //
                impl->d_dag_global,                                 //
                impl->d_header_global,                              //
                impl->d_target_global,                              //
                impl->d_kill_signal_host);
        start_nonce += batch_blocks;
        m_done = false;
    }

    bool busy = true;

    // Runs while there is work to do
    while (busy) {
        if (paused()) {
            std::unique_lock<std::mutex> l(m_doneMutex);
            m_done = true;
        }

        // We switch the futures as `previous_job` was already consumed in the previous loop iteration.
        std::swap(impl->previous_search_task, impl->new_search_task);

        // Eventually enqueue new work on the device
        if (m_done) {
            busy = false;
        } else {
            m_hung_miner.store(false);
            impl->new_search_task = run_ethash_search(                  //
                    m_block_multiple,                                   //
                    m_deviceDescriptor.sycl_work_items_search_kernel,   //
                    impl->q,                                            //
                    impl->new_search_task,                              //
                    start_nonce,                                        //
                    m_epochContext.dagNumItems,                         //
                    impl->d_dag_global,                                 //
                    impl->d_header_global,                              //
                    impl->d_target_global,                              //
                    impl->d_kill_signal_host);
        }

        Search_results results = impl->previous_search_task.get_result(impl->q);

        if (results.solCount > MAX_SEARCH_RESULTS) results.solCount = MAX_SEARCH_RESULTS;

        // Register potential results
        for (uint32_t i = 0; i < results.solCount; i++) {
            uint64_t nonce(start_nonce - batch_blocks + results.gid[i]);
            Farm::f().submitProof(Solution{nonce, h256(), w, std::chrono::steady_clock::now(), m_index});
            ReportSolution(w.header, nonce);
        }

        if (shouldStop()) {
            std::unique_lock<std::mutex> l(m_doneMutex);
            m_done = true;
        }

        start_nonce += batch_blocks;
        updateHashRate(m_deviceDescriptor.sycl_work_items_search_kernel, results.hashCount);
    }

#ifdef DEV_BUILD
    // Optionally log job switch time
    if (!shouldStop() && (g_logOptions & LOG_SWITCH)) {
        cnote << "Switch time: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_workSwitchStart).count() << " us.";
    }
#endif
}
