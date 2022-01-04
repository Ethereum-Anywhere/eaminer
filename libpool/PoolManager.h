/* Copyright (C) 1883 Thomas Edison - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3 license, which unfortunately won't be
 * written for another century.
 *
 * You should have received a copy of the LICENSE file with
 * this file.
 */

#pragma once

#include <iostream>

#include <json/json.h>

#include <libdev/Worker.h>
#include <libeth/Farm.h>
#include <libeth/Miner.h>

#include "PoolClient.h"
#include "getwork/EthGetworkClient.h"
#include "stratum/EthStratumClient.h"
#include "testing/SimulateClient.h"

using namespace std;

namespace dev::eth {
struct PoolSettings {
    std::vector<std::shared_ptr<URI>> connections;                 // List of connection definitions
    unsigned getWorkPollInterval = 500;                            // Interval (ms) between getwork requests
    unsigned noWorkTimeout = 180;                                  // If no new jobs in this number of seconds drop connection
    unsigned noResponseTimeout = 2;                                // If no response in this number of seconds drop connection
    unsigned poolFailoverTimeout = 0;                              // Return to primary pool after this number of minutes
    bool reportHashrate = false;                                   // Whether to report hashrate to pool
    unsigned hashRateInterval = 60;                                // Interval in seconds among hashrate submissions
    std::string hashRateId = h256::random().hex(HexPrefix::Add);   // Unique identifier for HashRate submission
    unsigned connectionMaxRetries = 3;                             // Max number of connection retries
    unsigned delayBeforeRetry = 0;                                 // Delay seconds before connect retry
    unsigned benchmarkBlock = 0;                                   // Block number used by SimulateClient to test performances
};

class PoolManager {
public:
    explicit PoolManager(PoolSettings _settings);
    static PoolManager& p() { return *m_this; }
    void addConnection(const std::string& _connstring);
    void addConnection(const std::shared_ptr<URI>& _uri);
    Json::Value getConnectionsJson();
    void setActiveConnection(unsigned int idx);
    void setActiveConnection(std::string& _connstring);
    std::shared_ptr<URI> getActiveConnection();
    void removeConnection(unsigned int idx);
    void start();
    void stop();
    bool isConnected() { return p_client->isConnected(); };
    bool isRunning() { return m_running; };
    int getCurrentEpoch() const;
    uint64_t getCurrentClientDuration() {
        if (p_client && isConnected()) {
            return p_client->m_session->usDuration();
        } else {
            return 0;
        }
    };
    double getPoolDifficulty();
    unsigned getConnectionSwitches();
    unsigned getEpochChanges();

private:
    void rotateConnect();
    void setClientHandlers();
    void showMiningAt();
    void setActiveConnectionCommon(unsigned int idx);
    void failovertimer_elapsed(const boost::system::error_code& ec);
    void submithrtimer_elapsed(const boost::system::error_code& ec);
    void reconnecttimer_elapsed(const boost::system::error_code& ec);

    PoolSettings m_Settings;
    std::atomic<bool> m_running = {false};
    std::atomic<bool> m_stopping = {false};
    std::atomic<bool> m_async_pending = {false};
    unsigned m_connectionAttempt = 0;
    std::string m_selectedHost;   // Holds host name (and endpoint) of selected connection
    std::atomic<unsigned> m_connectionSwitches = {0};
    unsigned m_activeConnectionIdx = 0;
    WorkPackage m_currentWp;
    boost::asio::io_service::strand m_io_strand;
    boost::asio::deadline_timer m_failovertimer;
    boost::asio::deadline_timer m_submithrtimer;
    boost::asio::deadline_timer m_reconnecttimer;
    std::unique_ptr<PoolClient> p_client = nullptr;
    std::atomic<unsigned> m_epochChanges = {0};
    static PoolManager* m_this;
    int m_lastBlock;
};

}   // namespace dev::eth
