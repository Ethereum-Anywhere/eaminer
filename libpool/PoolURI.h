/* Copyright (C) 1883 Thomas Edison - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3 license, which unfortunately won't be
 * written for another century.
 *
 * You should have received a copy of the LICENSE file with
 * this file.
 */

#pragma once

#include <regex>
#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/asio.hpp>
#include <boost/lexical_cast.hpp>

// A simple URI parser specifically for mining pool endpoints
namespace dev {
enum class SecureLevel { NONE = 0, TLS };

enum class ProtocolFamily { GETWORK = 0, STRATUM, SIMULATION };

enum class UriHostNameType {
    Unknown = 0,   // The type of the host name is not supplied
    Basic = 1,     // The host is set, but the type cannot be determined
    Dns = 2,       // The host name is a domain name system(DNS) style host name
    IPV4 = 3,      // The host name is an Internet Protocol(IP) version 4 host address
    IPV6 = 4       // The host name is an Internet Protocol(IP) version 6 host address.
};

class URI {
public:
    URI() = delete;
    explicit URI(std::string uri, bool _sim = false);

    [[nodiscard]] std::string Scheme() const { return m_scheme; }
    [[nodiscard]] std::string Host() const { return m_host; }
    [[nodiscard]] std::string Path() const { return m_path; }
    [[nodiscard]] unsigned short Port() const { return m_port; }
    [[nodiscard]] std::string User() const { return m_user; }
    [[nodiscard]] std::string Pass() const { return m_password; }
    [[nodiscard]] std::string Workername() const { return m_worker; }
    [[nodiscard]] std::string UserDotWorker() const;
    [[nodiscard]] SecureLevel SecLevel() const;
    [[nodiscard]] ProtocolFamily Family() const;
    [[nodiscard]] UriHostNameType HostNameType() const;
    [[nodiscard]] bool IsLoopBack() const;
    [[nodiscard]] unsigned Version() const;
    [[nodiscard]] std::string str() const { return m_uri; }

    static std::string KnownSchemes(ProtocolFamily family);

    void SetStratumMode(unsigned mode, bool confirmed) {
        m_stratumMode = mode;
        m_stratumModeConfirmed = confirmed;
    }
    void SetStratumMode(unsigned mode) { m_stratumMode = mode; }
    [[nodiscard]] unsigned StratumMode() const { return m_stratumMode; }
    [[nodiscard]] bool StratumModeConfirmed() const { return m_stratumModeConfirmed; }
    [[nodiscard]] bool IsUnrecoverable() const { return m_unrecoverable; }
    void MarkUnrecoverable() { m_unrecoverable = true; }
    [[nodiscard]] bool Responds() const { return m_responds; }
    void Responds(bool _value) { m_responds = _value; }
    void addDuration(unsigned long _minutes) { m_totalDuration += _minutes; }
    [[nodiscard]] unsigned long getDuration() const { return m_totalDuration; }

private:
    std::string m_scheme;
    std::string m_authority;   // Contains all text after scheme
    std::string m_userinfo;    // Contains the userinfo part
    std::string m_urlinfo;     // Contains the urlinfo part
    std::string m_hostinfo;    // Contains the hostinfo part
    std::string m_pathinfo;    // Contains the pathinfo part

    std::string m_host;
    std::string m_path;
    std::string m_query;
    std::string m_fragment;
    std::string m_user;
    std::string m_password = "X";
    std::string m_worker;
    std::string m_uri;

    unsigned short m_stratumMode = 999;   // Initial value 999 means not tested yet
    unsigned short m_port = 0;
    bool m_stratumModeConfirmed = false;
    bool m_unrecoverable = false;
    bool m_responds = false;
    UriHostNameType m_hostType = UriHostNameType::Unknown;
    bool m_isLoopBack;
    unsigned long m_totalDuration;   // Total duration on this connection in minutes
};
}   // namespace dev
