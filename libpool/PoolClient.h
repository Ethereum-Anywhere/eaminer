/* Copyright (C) 1883 Thomas Edison - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3 license, which unfortunately won't be
 * written for another century.
 *
 * You should have received a copy of the LICENSE file with
 * this file.
 */

#pragma once

#include <queue>

#include <boost/asio/ip/address.hpp>
#include <utility>

#include <libeth/Miner.h>
#include <libpool/PoolURI.h>

extern boost::asio::io_service g_io_service;

//using namespace std;

namespace dev::eth {
struct Session {
    // Tstamp of sessio start
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    // Whether worker is subscribed
    std::atomic<bool> subscribed = {false};
    // Whether worker is authorized
    std::atomic<bool> authorized = {false};
    // Total duration of session in minutes
    [[nodiscard]] unsigned long duration() const { return (std::chrono::duration_cast<std::chrono::minutes>(std::chrono::steady_clock::now() - start)).count(); }
    [[nodiscard]] uint64_t usDuration() const { return (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start)).count(); }

    // EthereumStratum (1 and 2)

    // Extranonce currently active
    uint64_t extraNonce = 0;
    // Length of extranonce in bytes
    unsigned int extraNonceSizeBytes = 0;
    // Next work target
    h256 nextWorkBoundary = h256("0x00000000ffff0000000000000000000000000000000000000000000000000000");
    double nextWorkDifficulty = 0;

    // EthereumStratum (2 only)
    bool firstMiningSet = false;
    unsigned int timeout = 30;   // Default to 30 seconds
    std::string sessionId;
    std::string workerId;
    unsigned int epoch = 0;
    std::chrono::steady_clock::time_point lastTxStamp = std::chrono::steady_clock::now();
};

class PoolClient {
public:
    virtual ~PoolClient() noexcept = default;

    // Sets the connection definition to be used by the client
    void setConnection(std::shared_ptr<URI> _conn) {
        m_conn = std::move(_conn);
        m_conn->Responds(false);
    }

    // Gets a pointer to the currently active connection definition
    std::shared_ptr<URI> getConnection() { return m_conn; }

    // Releases the pointer to the connection definition
    void unsetConnection() { m_conn = nullptr; }

    virtual void connect() = 0;
    virtual void disconnect() = 0;
    virtual void submitHashrate(uint64_t const& rate, std::string const& id) = 0;
    virtual void submitSolution(const Solution& solution) = 0;
    virtual bool isConnected() { return m_connected.load(std::memory_order_relaxed); }
    virtual bool isPendingState() { return false; }

    virtual bool isSubscribed() { return m_session && m_session->subscribed.load(std::memory_order_relaxed); }
    virtual bool isAuthorized() { return m_session && m_session->authorized.load(std::memory_order_relaxed); }

    virtual std::string ActiveEndPoint() { return (m_connected.load(std::memory_order_relaxed) ? " [" + toString(m_endpoint) + "]" : ""); }

    using SolutionAccepted = std::function<void(std::chrono::milliseconds const&, unsigned const&, bool)>;
    using SolutionRejected = std::function<void(std::chrono::milliseconds const&, unsigned const&)>;
    using Disconnected = std::function<void()>;
    using Connected = std::function<void()>;
    using WorkReceived = std::function<void(WorkPackage const&)>;

    void onSolutionAccepted(SolutionAccepted const& _handler) { m_onSolutionAccepted = _handler; }
    void onSolutionRejected(SolutionRejected const& _handler) { m_onSolutionRejected = _handler; }
    void onDisconnected(Disconnected const& _handler) { m_onDisconnected = _handler; }
    void onConnected(Connected const& _handler) { m_onConnected = _handler; }
    void onWorkReceived(WorkReceived const& _handler) { m_onWorkReceived = _handler; }

    std::unique_ptr<Session> m_session = nullptr;

protected:
    std::atomic<bool> m_connected = {false};   // This is related to socket ! Not session

    boost::asio::ip::basic_endpoint<boost::asio::ip::tcp> m_endpoint;

    std::shared_ptr<URI> m_conn = nullptr;

    SolutionAccepted m_onSolutionAccepted;
    SolutionRejected m_onSolutionRejected;
    Disconnected m_onDisconnected;
    Connected m_onConnected;
    WorkReceived m_onWorkReceived;
};
}   // namespace dev::eth
