
/* Copyright (C) 1883 Thomas Edison - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3 license, which unfortunately won't be
 * written for another century.
 *
 * You should have received a copy of the LICENSE file with
 * this file.
 */

#include <ethash/ethash.hpp>
#include <libeth/Farm.h>

#include "SYCLMiner.h"

using namespace dev;
using namespace eth;

/**
 *
 * @param _index
 * @param _device
 */
SYCLMiner::SYCLMiner(unsigned _index, DeviceDescriptor& _device) : Miner("SYCL-", _index) {
    q = sycl::queue{get_platform_devices()[_device.sycl_device_idx]};   //, sycl::property::queue::enable_profiling{}};
    m_deviceDescriptor = _device;
    m_block_multiple = 1000;
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
    cextr << "Using SYCL device: " << q.get_device().get_info<sycl::info::device::name>()
          << ") Memory : " << dev::getFormattedMemory((double) m_deviceDescriptor.totalMemory);

    // Set Hardware Monitor Info
    m_hwmoninfo.deviceType = HwMonitorInfoType::NVIDIA;
    m_hwmoninfo.devicePciId = m_deviceDescriptor.uniqueId;
    m_hwmoninfo.deviceIndex = -1;   // Will be later on mapped by nvml (see Farm() constructor)

    try {
        reset_device();
        q.single_task([]() { (void) nullptr; }).wait_and_throw();
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
void SYCLMiner::reset_device() {

    try {
        q.wait_and_throw();
    } catch (const std::exception& e) { cwarn << "Caught exception " << e.what() << " while reseting: " << q.get_device().get_info<sycl::info::device::name>(); }

    if (d_light_global) {
        sycl::free(d_light_global, q);
        d_light_global = nullptr;
    }

    if (d_dag_global) {
        sycl::free(d_dag_global, q);
        d_dag_global = nullptr;
    }

    if (m_search_buf) {
        sycl::free(m_search_buf, q);
        m_search_buf = nullptr;
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
            d_light_global = reinterpret_cast<hash64_t*>(sycl::malloc_device(m_epochContext.lightSize, q));
        } catch (...) {
            ReportGPUNoMemoryAndPause("light cache", m_epochContext.lightSize, m_deviceDescriptor.totalMemory);
            return false;   // This will prevent to exit the thread and
        }
        try {
            d_dag_global = reinterpret_cast<hash128_t*>(sycl::malloc_device(m_epochContext.dagSize, q));
        } catch (...) {
            ReportGPUNoMemoryAndPause("DAG", m_epochContext.dagSize, m_deviceDescriptor.totalMemory);
            return false;   // This will prevent to exit the thread and
        }

        // create mining buffer
        try {
            m_search_buf = sycl::malloc_device<Search_results>(1, q);
        } catch (...) {
            ReportGPUNoMemoryAndPause("mining buffer", sizeof(Search_results), m_deviceDescriptor.totalMemory);
            return false;   // This will prevent to exit the thread and
        }

        // Release the pause flag if any
        resume(MinerPauseEnum::PauseDueToInsufficientMemory);
        resume(MinerPauseEnum::PauseDueToInitEpochError);

        sycl::event e = q.memcpy(d_light_global, m_epochContext.lightCache, m_epochContext.lightSize);
        auto gen_events = ethash_generate_dag(        //
                m_epochContext.dagSize,               //
                m_block_multiple,                     //
                m_deviceDescriptor.sycl_work_items,   //
                q,                                    //
                m_epochContext.dagNumItems,           //
                m_epochContext.lightNumItems,         //
                d_dag_global,                         //
                d_light_global, e);

        for (auto& e: gen_events) { e.wait_and_throw(); }

        ReportDAGDone(   //
                m_epochContext.dagSize, uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startInit).count()), true);
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

            uint64_t upper64OfBoundary((uint64_t) (u64) ((u256) current.boundary >> 192));

            // adjust work multiplier
            float hr = RetrieveHashRate();
            if (hr >= 1e7) m_block_multiple = uint32_t((hr * target_batch_time) / (m_deviceDescriptor.sycl_work_items));

            // Eventually start searching
            search(current.header.data(), upper64OfBoundary, current.startNonce, current);
        }

        // Reset miner and stop working
        reset_device();
    } catch (std::exception& e) {
        std::string _what = "GPU error: ";
        _what.append(e.what());
        throw std::runtime_error(_what);
    }
}

/**
 *
 * @return
 */
const std::vector<sycl::device>& SYCLMiner::get_platform_devices() {
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

/**
 *
 */
void SYCLMiner::kick_miner() {
    static const uint32_t one(1);
    std::unique_lock<std::mutex> l(m_doneMutex);
    if (!m_done) {
        m_done = true;
        q.memcpy((uint8_t*) m_search_buf + offsetof(Search_results, done), &one, sizeof(one)).wait();
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
            deviceDescriptor.uniqueId = "[SYCL " + std::to_string(i) + "] ";

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
            deviceDescriptor.sycl_work_items = 128;
            deviceDescriptor.boardName = "[SYCL " + std::to_string(i) + "] " + get_platform_devices()[i].get_info<sycl::info::device::name>();
            DevicesCollection[deviceDescriptor.uniqueId] = deviceDescriptor;
        } catch (std::exception& e) { ccrit << e.what(); }
    }
}

static const uint32_t zero3[3] = {0, 0, 0};   // zero the result count

/**
 *
 * @param header
 * @param target
 * @param start_nonce
 * @param w
 */
void SYCLMiner::search(uint8_t const* header, uint64_t target, uint64_t start_nonce, const dev::eth::WorkPackage& w) {
    d_header_global = *(reinterpret_cast<const hash32_t*>(header));
    d_target_global = target;

    uint32_t batch_blocks(m_block_multiple * m_deviceDescriptor.sycl_work_items);

    // prime the queue , clear search result buffers and start the search
    sycl::event running_ethash_event{};
    {
        std::unique_lock<std::mutex> l(m_doneMutex);
        sycl::event copy_in_evt = q.memcpy(m_search_buf, zero3, sizeof(zero3));
        m_hung_miner.store(false);
        running_ethash_event = run_ethash_search(     //
                m_block_multiple,                     //
                m_deviceDescriptor.sycl_work_items,   //
                q, m_search_buf,                      //
                start_nonce,                          //
                m_epochContext.dagNumItems,           //
                d_dag_global,                         //
                d_header_global,                      //
                d_target_global, copy_in_evt);
        start_nonce += batch_blocks;
        m_done = false;
    }

    bool busy = true;

    // process stream batches until we get new work.
    while (busy) {
        if (paused()) {
            std::unique_lock<std::mutex> l(m_doneMutex);
            m_done = true;
        }

        uint32_t batchCount(0);

        // This inner loop will process each cuda stream individually

        // Wait for the stream complete

        Search_results r{};
        sycl::event retrieve_result_evt = q.memcpy(&r, m_search_buf, sizeof(Search_results), running_ethash_event);
        // clear solution count, hash count and done
        sycl::event reset_after_retrieve = q.memcpy(m_search_buf, zero3, sizeof(zero3), retrieve_result_evt);


        if (m_done) {
            busy = false;
        } else {
            m_hung_miner.store(false);
            running_ethash_event = run_ethash_search(     //
                    m_block_multiple,                     //
                    m_deviceDescriptor.sycl_work_items,   //
                    q,                                    //
                    m_search_buf,                         //
                    start_nonce,                          //
                    m_epochContext.dagNumItems,           //
                    d_dag_global,                         //
                    d_header_global,                      //
                    d_target_global, reset_after_retrieve);
        }


        retrieve_result_evt.wait_and_throw();
        if (r.solCount > MAX_SEARCH_RESULTS) r.solCount = MAX_SEARCH_RESULTS;
        batchCount += r.hashCount;

        for (uint32_t i = 0; i < r.solCount; i++) {
            uint64_t nonce(start_nonce - batch_blocks + r.gid[i]);
            Farm::f().submitProof(Solution{nonce, h256(), w, std::chrono::steady_clock::now(), m_index});
            ReportSolution(w.header, nonce);
        }

        if (shouldStop()) {
            std::unique_lock<std::mutex> l(m_doneMutex);
            m_done = true;
        }

        start_nonce += batch_blocks;

        updateHashRate(m_deviceDescriptor.sycl_work_items, batchCount);
    }

    running_ethash_event.wait_and_throw();

#ifdef DEV_BUILD
    // Optionally log job switch time
    if (!shouldStop() && (g_logOptions & LOG_SWITCH)) {
        cnote << "Switch time: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_workSwitchStart).count() << " us.";
    }
#endif
}
