# eaminer's API documentation

## Table of Contents

* [Introduction](#introduction)
* [Activation and Security](#activation-and-security)
* [Usage](#usage)
* [List of requests](#list-of-requests)
    * [api_authorize](#api_authorize)
    * [miner_ping](#miner_ping)
    * [miner_getstatdetail](#miner_getstatdetail)
    * [miner_getstat1](#miner_getstat1)
    * [miner_restart](#miner_restart)
    * [miner_reboot](#miner_reboot)
    * [miner_getconnections](#miner_getconnections)
    * [miner_setactiveconnection](#miner_setactiveconnection)
    * [miner_addconnection](#miner_addconnection)
    * [miner_removeconnection](#miner_removeconnection)
    * [miner_pausegpu](#miner_pausegpu)
    * [miner_setverbosity](#miner_setverbosity)
    * [miner_setnonce](#miner_setnonce)
    * [miner_getnonce](#miner_getnonce)

## Introduction

eaminer implements an API (Application Programming Interface) interface which allows to monitor/control some of the run-time values endorsed by this miner. The API interface is available under the following
circumstances:

* If you build the application from source ensuring you add the compilation switch `-D APICORE=ON`

## Activation and Security

Whenever the above depicted conditions are met you can take advantage of the API support by adding the `--api-bind` argument to the command line used to launch eaminer. The format of this argument
is `--api-bind address:port` where `nnnn` is any valid TCP port number (1-65535) and is required, and the `address` dictates what ip the api will listen on, and is optional, and defaults to "all ipv4 addresses".
Examples:

```shell
./eaminer [...] --api-bind 3333
```

This example puts the API interface listening on port 3333 of **any** local IPv4 address which means the loop-back interface (127.0.0.1/127.0.1.1) and any configured IPv4 address of the network card(s). To only listen to localhost connections (which may be a more secure setting),

```shell
./eaminer [...] --api-bind 127.0.0.1:3333
```
and likewise, to only listen on a specific address, replace `127.0.0.1` accordingly.



The API interface not only offers monitoring queries but also implements some methods which may affect the functioning of the miner. These latter operations are named _write_ actions: if you want to inhibit the invocation of such methods you may want to put the API interface in **read-only** mode which means only query to **get** data will be allowed and no _write_ methods will be allowed. To do this simply add the - (minus) sign in front of the port number thus transforming the port number into a negative number. Example for read-only mode:

```shell
./eaminer [...] --api-bind -3333
```

_Note. The port number in this examples is taken randomly and does not imply a suggested value. You can use any port number you wish while it's not in use by other applications._

To gain further security you may wish to password protect the access to your API interface simply by adding the `--api-password` argument to the command line sequence, followed by the password you wish. Password may be composed by any printable char and **must not** have spaces. Password checking is **case sensitive**. Example for password protected API interface:

```shell
./eaminer [...] --api-bind -3333 --api-password MySuperSecurePassword!!#123456
```

At the time of writing of this document eaminer's API interface does not implement any sort of data encryption over SSL secure channel so **be advised your passwords will be sent as plain text over plain TCP sockets**.

## Usage

Access to API interface is performed through a TCP socket connection to the API endpoint (which is the IP address of the computer running eaminer's API instance at the configured port). For instance if your computer
address is 192.168.1.1 and have configured eaminer to run with `--api-bind 3333` your endpoint will be 192.168.1.1:3333.

Messages exchanged through this channel must conform to the [JSON-RPC 2.0 specification](http://www.jsonrpc.org/specification) so basically you will issue **requests** and will get back **responses**. At the time of
writing this document do not expect any **notification**. All messages must be line feed terminated.

To quickly test if your eaminer's API instance is working properly you can issue this simple command:

```shell
echo '{"id":0,"jsonrpc":"2.0","method":"miner_ping"}' | netcat 192.168.1.1 3333
```

and will get back a response like this:

```shell
{"id":0,"jsonrpc":"2.0","result":"pong"}
```

This shows the API interface is live and listening on the configured endpoint.

## List of requests

|   Method  | Description  | Write Protected |
| --------- | ------------ | --------------- |
| [api_authorize](#api_authorize) | Issues the password to authenticate the session | No |
| [miner_ping](#miner_ping) | Responds back with a "pong" | No |
| [miner_getstatdetail](#miner_getstatdetail) | Request the retrieval of operational data in most detailed form | No
| [miner_getstat1](#miner_getstat1) | Request the retrieval of operational data in compatible format | No
| [miner_restart](#miner_restart) | Instructs eaminer to stop and restart mining | Yes |
| [miner_reboot](#miner_reboot) | Try to launch reboot.bat (on Windows) or reboot.sh (on Linux) in the eaminer executable directory | Yes
| [miner_getconnections](#miner_getconnections) | Returns the list of connections held by eaminer | No
| [miner_setactiveconnection](#miner_setactiveconnection) | Instruct eaminer to immediately connect to the specified connection | Yes
| [miner_addconnection](#miner_addconnection) | Provides eaminer with a new connection to use | Yes
| [miner_removeconnection](#miner_removeconnection) | Removes the given connection from the list of available so it won't be used again | Yes
| [miner_pausegpu](#miner_pausegpu) | Pause/Start mining on specific GPU | Yes
| [miner_setverbosity](#miner_setverbosity) | Set console log verbosity level | Yes
| [miner_setnonce](#miner_setnonce) | Sets the miner's start nonce | Yes
| [miner_getnonce](#miner_getnonce) | Gets miner's start nonce | no

### api_authorize

If your API instance is password protected by the usage of `--api-password` any remote trying to interact with the API interface **must** send this method immediately after connection to get authenticated. The message to send is:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "api_authorize",
  "params": {
    "psw": "MySuperSecurePassword!!#123456"
  }
}
```

where the member `psw` **must** contain the very same password configured with `--api-password` argument. As expected result you will get a JSON-RPC 2.0 response with positive or negative values. For instance if the password matches you will get a response like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": true,
}
```

or, in case of any error:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "error": {
    "code": -401,
    "message": "Invalid password"
  }
}
```

### miner_ping

This method is primarily used to check the liveness of the API interface.

To invoke the action:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_ping"
}
```

and expect back a result like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": "pong"
}
```

which confirms the action has been performed.

If you get no response or the socket timeouts it's likely your eaminer's instance has become unresponsive (or in worst cases the OS of your mining rig is unresponsive) and needs to be re-started/re-booted.

### miner_getstatdetail

With this method you expect back a detailed collection of statistical data. To issue a request:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_getstatdetail"
}
```

and expect back a response like this:

```js
{
  "id": 0,
  "jsonrpc": "2.0",
  "result": {
    "connection": {                                     // Current active connection
      "connected": true,
      "switches": 1,
      "uri": "stratum1+ssl://<ethaddress>.wworker@eu1.ethermine.org:5555"
    },
    "devices": [                                        // Array subscribed of devices
      {
        "_index": 0,                                    // Miner ordinal 
        "_mode": "CUDA",                                // Miner mode : "OpenCL" / "CUDA"
        "hardware": {                                   // Device hardware info
          "name": "GeForce GTX 1050 Ti 3.95 GB",        // Name
          "pci": "01:00.0",                             // Pci Id
          "sensors": [                                  // An array made of ...
            47,                                         //  + Detected temp
            70,                                         //  + Fan percent
            0,                                          //  + Power drain in watts
            55                                          //  + Detected memory temp
          ],
          "type": "GPU"                                 // Device Type : "CPU" / "GPU" / "ACCELERATOR"
        },
        "mining": {                                     // Mining info
          "hashrate": "0x0000000000e3fcbb",             // Current hashrate in hashes per second
          "pause_reason": null,                         // If the device is paused this contains the reason
          "paused": false,                              // Wheter or not the device is paused
          "segment": [                                  // The search segment of the device
            "0xbcf0a663bfe75dab",                       //  + Lower bound
            "0xbcf0a664bfe75dab"                        //  + Upper bound
          ],
          "shares": [                                   // Shares / Solutions stats
            1,                                          //  + Found shares
            0,                                          //  + Rejected (by pool) shares
            0,                                          //  + Failed shares (always 0 if --no-eval is set)
            15                                          //  + Time in seconds since last found share
          ]
        }
      },
      { ... }                                           // Another device
      { ... }                                           // And another ...
    ],
    "host": {
      "name": "miner01",                                // Host name of the computer running eaminer
      "runtime": 121,                                   // Duration time (in seconds)
      "version": "eaminer-1.0.0"
    },
    "mining": {                                         // Mining info for the whole instance
      "difficulty": 3999938964,                         // Actual difficulty in hashes
      "epoch": 227,                                     // Current epoch
      "epoch_changes": 1,                               // How many epoch changes occurred during the run
      "hashrate": "0x00000000054a89c8",                 // Overall hashrate (sum of hashrate of all devices)
      "shares": [                                       // Shares / Solutions stats
        2,                                              //  + Found shares
        0,                                              //  + Rejected (by pool) shares
        0,                                              //  + Failed shares (always 0 if --no-eval is set)
        15                                              //  + Time in seconds since last found share
      ]
    },
    "monitors": {                                       // A nullable object which may contain some triggers
      "temperatures": [                                 // Monitor temperature
        60,                                             //  + Resume mining if device temp is <= this threshold
        75                                              //  + Suspend mining if device temp is >= this threshold
      ]
    }
  }
}
```

### miner_getstat1

With this method you expect back a collection of statistical data. To issue a request:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_getstat1"
}
```

and expect back a response like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": [
    "eaminer-1.0.0", // Running eaminer's version
    "48",                                   // Total running time in minutes
    "87221;54;0",                           // ETH hashrate in KH/s, submitted shares, rejected shares
    "14683;14508;14508;14508;14508;14508",  // Detailed ETH hashrate in KH/s per GPU
    "0;0;0",                                // DCR hashrate in KH/s, submitted shares, rejected shares (not used)
    "off;off;off;off;off;off",              // Detailed DCR hashrate in KH/s per GPU (not used)
    "53;90;50;90;56;90;58;90;61;90;60;90",  // Temp and fan speed pairs per GPU
    "eu1.ethermine.org:4444",               // Mining pool currently active
    "0;0;0;0",                              // ETH invalid shares, ETH pool switches, DCR invalid shares, DCR pool switches
    "0;59;68"                               // Memory temp per GPU, 0 if unavailable
  ]
}
```

Some of the arguments here expressed have been set for compatibility with other miners so their values are not set. For instance, eaminer **does not** support dual (ETH/DCR) mining.

### miner_restart

With this method you instruct eaminer to _restart_ mining. Restarting means:

* Stop actual mining work
* Unload generated DAG files
* Reset devices (GPU)
* Regenerate DAG files
* Restart mining

The invocation of this method **_may_** be useful if you detect one or more GPUs are in error, but in a recoverable state (eg. no hashrate but the GPU has not fallen off the bus). In other words, this method works like
stopping eaminer and restarting it **but without loosing connection to the pool**.

To invoke the action:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_restart"
}
```

and expect back a result like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": true
}
```

which confirms the action has been performed.

**Note**: This method is not available if the API interface is in read-only mode (see above).

### miner_reboot

With this method you instruct eaminer to execute reboot.bat (on Windows) or reboot.sh (on Linux) script which must exists and being executable in the eaminer directory. As eaminer has no idea what's going on in the
script, eaminer continues with it's normal work. If you invoke this function `api_miner_reboot` is passed to the script as first parameter.

To invoke the action:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_reboot"
}
```

and expect back a result like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": true
}
```

which confirms an executable file was found and eaminer tried to start it.

**Note**: This method is not available if the API interface is in read-only mode (see above).

### miner_getconnections

When you launch eaminer you provide a list of connections specified by the `-P` argument. If you want to remotely check which is the list of connections eaminer is using, you can issue this method:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_getconnections"
}
```

and expect back a result like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": [
    {
      "active": false,
      "index": 0,
      "uri": "stratum+tcp://<omitted-ethereum-address>.worker@eu1.ethermine.org:4444"
    },
    {
      "active": true,
      "index": 1,
      "uri": "stratum+tcp://<omitted-ethereum-address>.worker@eu1.ethermine.org:14444"
    },
    {
      "active": false,
      "index": 2,
      "uri": "stratum+tcp://<omitted-ethereum-classic-address>.worker@eu1-etc.ethermine.org:4444"
    }
  ]
}
```

The `result` member contains an array of objects, each one with the definition of the connection (in the form of the URI entered with the `-P` argument), its ordinal index and the indication if it's the currently active connetion.

### miner_setactiveconnection

Given the example above for the method [miner_getconnections](#miner_getconnections) you see there is only one active connection at a time. If you want to control remotely your mining facility and want to force the switch from one connection to another you can issue this method:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_setactiveconnection",
  "params": {
    "index": 0
  }
}
```
or
```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_setactiveconnection",
  "params": {
    "URI": ".*etc.*"
  }
}
```

You have to pass the `params` member as an object which has member `index` valued to the ordinal index of the connection you want to activate. Alternatively, you can pass a regular expression to be matched against the connection URIs. As a result you expect one of the following:

* Nothing happens if the provided index is already bound to an _active_ connection
* If the selected index is not of an active connection then eaminer will disconnect from currently active connection and reconnect immediately to the newly selected connection
* An error result if the index is out of bounds or the request is not properly formatted

**Please note** that this method changes the runtime behavior only. If you restart eaminer from a batch file the active connection will become again the first one of the `-P` arguments list.

### miner_addconnection

If you want to remotely add a new connection to the running instance of eaminer you can use this this method by sending a message like this

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_addconnection",
  "params": {
    "uri": "stratum+tcp://<ethaddress>.<workername>@eu1.ethermine.org:4444"
  }
}
```

You have to pass the `params` member as an object which has member `uri` valued exactly the same way you'd add a connection using the `-P` argument. As a result you expect one of the following:

* An error if the uri is not properly formatted
* An error if you try to _mix_ stratum mode with getwork mode (which begins with `http://`)
* A success message if the newly defined connection has been properly added

Eventually you may want to issue [miner_getconnections](#miner_getconnections) method to identify which is the ordinal position assigned to the newly added connection and make use
of [miner_setactiveconnection](#miner_setactiveconnection) method to instruct eaminer to use it immediately.

**Please note** that this method changes the runtime behavior only. If you restart eaminer from a batch file the added connection won't be available if not present in the `-P` arguments list.

### miner_removeconnection

Recall once again the example for the method [miner_getconnections](#miner_getconnections). If you wish to remove the third connection (the Ethereum classic one) from the list of connections (so it won't be used in case of failover) you can send this method:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_removeconnection",
  "params": {
    "index": 2
  }
}
```

You have to pass the `params` member as an object which has member `index` valued to the ordinal index (zero based) of the connection you want to remove. As a result you expect one of the following:

* An error if the index is out of bounds **or if the index corresponds to the currently active connection**
* A success message. In such case you can later reissue [miner_getconnections](#miner_getconnections) method to check the connection has been effectively removed.

**Please note** that this method changes the runtime behavior only. If you restart eaminer from a batch file the removed connection will become again again available if provided in the `-P` arguments list.

### miner_pausegpu

Pause or (restart) mining on specific GPU.
This ONLY (re)starts mining if GPU was paused via a previous API call and not if GPU pauses for other reasons.

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_pausegpu",
  "params": {
    "index": 0,
    "pause": true
  }
}
```

and expect a result like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": true
}
```

which confirms the action has been performed.
Again: This ONLY (re)starts mining if GPU was paused via a previous API call and not if GPU pauses for other reasons.

### miner_setverbosity

Set the verbosity level of eaminer.

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_setverbosity",
  "params": {
    "verbosity": 9
  }
}
```

and expect a result like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": true
}
```

### miner_setnonce

Set the miner's start nonce. Can be useful in avoiding search range overlaps in multi-miner situations.

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_setnonce",
  "params": {
    "nonce": "12ab"
  }
}
```

and expect a result like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": true
}
```

### miner_getnonce

Set the miner's start nonce. Can be useful in avoiding search range overlaps in multi-miner situations.

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "method": "miner_getnonce"
}
```

and expect a result like this:

```js
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": "123"
}
```
