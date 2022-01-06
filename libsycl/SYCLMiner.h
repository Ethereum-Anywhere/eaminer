/* Copyright (C) 1883 Thomas Edison - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3 license, which unfortunately won't be
 * written for another century.
 *
 * You should have received a copy of the LICENSE file with
 * this file.
 */

#pragma once

#include "ethash_sycl_miner_kernel.h"

#include "libeth/Farm.h"

#include <libdev/Worker.h>
#include <libeth/EthashAux.h>
#include <libeth/Miner.h>

#include <functional>


namespace dev::eth {
class SYCLMiner : public Miner {
public:
    SYCLMiner(unsigned _index, DeviceDescriptor& _device);

    SYCLMiner& operator=(const SYCLMiner&) = delete;

    SYCLMiner(const SYCLMiner&) = delete;

    ~SYCLMiner() override;

    static int getNumDevices();

    static void enumDevices(minerMap& DevicesCollection);

protected:
    bool initDevice() override;

    bool initEpoch() override;

    void kick_miner() override;

    void reset_device() noexcept;

private:
    static const std::vector<sycl::device>& get_platform_devices();

    void workLoop() override;

    void search(uint8_t const* header, uint64_t target, uint64_t _startN, const dev::eth::WorkPackage& w);

private:
    constexpr static double target_batch_time = 0.5;

    volatile bool m_done = {true};
    std::mutex m_doneMutex;

    sycl::queue q{sycl::default_selector{}};

    uint32_t* d_kill_signal = nullptr;
    hash128_t* d_dag_global = nullptr;
    hash64_t* d_light_global = nullptr;
    hash32_t d_header_global{};
    uint64_t d_target_global{};
};

}   // namespace dev::eth
