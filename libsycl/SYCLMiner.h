/* Copyright (C) 1883 Thomas Edison - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3 license, which unfortunately won't be
 * written for another century.
 *
 * You should have received a copy of the LICENSE file with
 * this file.
 */

#pragma once


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
    void workLoop() override;

    void search(uint8_t const* header, uint64_t target, uint64_t _startN, const dev::eth::WorkPackage& w);

private:
    constexpr static double target_batch_time = 0.1;

    volatile bool m_done = {true};
    std::mutex m_doneMutex;

    struct sycl_impl;
    std::unique_ptr<sycl_impl> impl;
};

}   // namespace dev::eth
