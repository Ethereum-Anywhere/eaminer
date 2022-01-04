
/* Copyright (C) 1883 Thomas Edison - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3 license, which unfortunately won't be
 * written for another century.
 *
 * You should have received a copy of the LICENSE file with
 * this file.
 */

#include <cstdio>
#include <cstdlib>

#include "wrapnvml.h"

#if defined(__cplusplus)
extern "C" {
#endif

wrap_nvml_handle* wrap_nvml_create() {
    wrap_nvml_handle* nvmlh = nullptr;

    /*
     * We use hard-coded library installation locations for the time being...
     * No idea where or if libnvidia-ml.so is installed on Mac OS X, a
     * deep scouring of the filesystem on one of the Mac CUDA build boxes
     * I used turned up nothing, so for now it's not going to work on OSX.
     */
#if defined(_WIN32)

/* Windows */
#    define libnvidia_ml1 "nvml.dll"
#    define libnvidia_ml2 "%WINDIR%/system32/nvml.dll"
#    define libnvidia_ml3 "%PROGRAMFILES%/NVIDIA Corporation/NVSMI/nvml.dll"

#elif defined(__linux)

/* In rpm based linux distributions link name is with extension .1 */
/* 32-bit linux assumed */
#    define libnvidia_ml "libnvidia-ml.so.1"

#else

#    define libnvidia_ml ""
#    warning "Unrecognized platform: need NVML DLL path for this platform..."
    return nullptr;

#endif

    void* nvml_dll = nullptr;

#ifdef _WIN32
    char tmp[512];
    ExpandEnvironmentStringsA(libnvidia_ml1, tmp, sizeof(tmp));
    nvml_dll = wrap_dlopen(tmp);
    if (nvml_dll == nullptr) {
        ExpandEnvironmentStringsA(libnvidia_ml2, tmp, sizeof(tmp));
        nvml_dll = wrap_dlopen(tmp);
        if (nvml_dll == nullptr) {
            ExpandEnvironmentStringsA(libnvidia_ml3, tmp, sizeof(tmp));
            nvml_dll = wrap_dlopen(tmp);
        }
    }
#else
    nvml_dll = wrap_dlopen(libnvidia_ml);
#endif
    if (!nvml_dll) {
        cwarn << "Failed to load NVML library";
        cwarn << "NVIDIA hardware monitoring disabled";
        return nullptr;
    }

    nvmlh = (wrap_nvml_handle*) calloc(1, sizeof(wrap_nvml_handle));
    if (nvmlh == nullptr) return nullptr;

    nvmlh->nvml_dll = nvml_dll;

    nvmlh->nvmlInit = (wrap_nvmlReturn_t(*)(void)) wrap_dlsym(nvmlh->nvml_dll, "nvmlInit");
    nvmlh->nvmlDeviceGetCount = (wrap_nvmlReturn_t(*)(int*)) wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetCount_v2");
    nvmlh->nvmlDeviceGetHandleByIndex = (wrap_nvmlReturn_t(*)(int, wrap_nvmlDevice_t*)) wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetHandleByIndex_v2");
    nvmlh->nvmlDeviceGetPciInfo = (wrap_nvmlReturn_t(*)(wrap_nvmlDevice_t, wrap_nvmlPciInfo_t*)) wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPciInfo");
    nvmlh->nvmlDeviceGetName = (wrap_nvmlReturn_t(*)(wrap_nvmlDevice_t, char*, int)) wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetName");
    nvmlh->nvmlDeviceGetTemperature = (wrap_nvmlReturn_t(*)(wrap_nvmlDevice_t, int, unsigned int*)) wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetTemperature");
    nvmlh->nvmlDeviceGetFanSpeed = (wrap_nvmlReturn_t(*)(wrap_nvmlDevice_t, unsigned int*)) wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetFanSpeed");
    nvmlh->nvmlDeviceGetPowerUsage = (wrap_nvmlReturn_t(*)(wrap_nvmlDevice_t, unsigned int*)) wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPowerUsage");
    nvmlh->nvmlShutdown = (wrap_nvmlReturn_t(*)()) wrap_dlsym(nvmlh->nvml_dll, "nvmlShutdown");
    nvmlh->nvmlDeviceGetFieldValues = (wrap_nvmlReturn_t(*)(wrap_nvmlDevice_t, int, wrap_nvmlFieldValue*)) wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetFieldValues");

    if (nvmlh->nvmlInit == nullptr || nvmlh->nvmlShutdown == nullptr || nvmlh->nvmlDeviceGetCount == nullptr || nvmlh->nvmlDeviceGetHandleByIndex == nullptr ||
        nvmlh->nvmlDeviceGetPciInfo == nullptr || nvmlh->nvmlDeviceGetName == nullptr || nvmlh->nvmlDeviceGetTemperature == nullptr ||
        nvmlh->nvmlDeviceGetFanSpeed == nullptr || nvmlh->nvmlDeviceGetPowerUsage == nullptr || nvmlh->nvmlDeviceGetFieldValues == nullptr) {
        cwarn << "Failed to obtain all required NVML function pointers";
        cwarn << "NVIDIA hardware monitoring disabled";

        wrap_dlclose(nvmlh->nvml_dll);
        free(nvmlh);
        return nullptr;
    }

    nvmlh->nvmlInit();
    nvmlh->nvmlDeviceGetCount(&nvmlh->nvml_gpucount);

    nvmlh->devs = (wrap_nvmlDevice_t*) calloc(nvmlh->nvml_gpucount, sizeof(wrap_nvmlDevice_t));
    if (!nvmlh->devs) {
        cwarn << "Failed to load NVML library";
        cwarn << "NVIDIA hardware monitoring disabled";
        wrap_dlclose(nvmlh->nvml_dll);
        free(nvmlh);
        return nullptr;
    }
    nvmlh->nvml_pci_domain_id = (unsigned int*) calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
    nvmlh->nvml_pci_bus_id = (unsigned int*) calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
    nvmlh->nvml_pci_device_id = (unsigned int*) calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));

    /* Obtain GPU device handles we're going to need repeatedly... */
    for (int i = 0; i < nvmlh->nvml_gpucount; i++) { nvmlh->nvmlDeviceGetHandleByIndex(i, &nvmlh->devs[i]); }

    /* Query PCI info for each NVML device, and build table for mapping of */
    /* CUDA device IDs to NVML device IDs and vice versa                   */
    for (int i = 0; i < nvmlh->nvml_gpucount; i++) {
        wrap_nvmlPciInfo_t pciinfo;
        nvmlh->nvmlDeviceGetPciInfo(nvmlh->devs[i], &pciinfo);
        nvmlh->nvml_pci_domain_id[i] = pciinfo.domain;
        nvmlh->nvml_pci_bus_id[i] = pciinfo.bus;
        nvmlh->nvml_pci_device_id[i] = pciinfo.device;
    }

    return nvmlh;
}

int wrap_nvml_destroy(wrap_nvml_handle* nvmlh) {
    nvmlh->nvmlShutdown();

    wrap_dlclose(nvmlh->nvml_dll);
    free(nvmlh);
    return 0;
}

int wrap_nvml_get_gpucount(wrap_nvml_handle* nvmlh, int* gpucount) {
    *gpucount = nvmlh->nvml_gpucount;
    return 0;
}

int wrap_nvml_get_gpu_name(wrap_nvml_handle* nvmlh, int gpuindex, char* namebuf, int bufsize) {
    if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount) return -1;

    if (nvmlh->nvmlDeviceGetName(nvmlh->devs[gpuindex], namebuf, bufsize) != WRAPNVML_SUCCESS) return -1;

    return 0;
}

int wrap_nvml_get_tempC(wrap_nvml_handle* nvmlh, int gpuindex, unsigned int* tempC) {
    if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount) return -1;

    if (nvmlh->nvmlDeviceGetTemperature(nvmlh->devs[gpuindex], 0u /* NVML_TEMPERATURE_GPU */, tempC) != WRAPNVML_SUCCESS) return -1;

    return 0;
}

int wrap_nvml_get_mem_tempC(wrap_nvml_handle* nvmlh, int gpuindex, unsigned int* tempC) {
    if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount) return -1;
    wrap_nvmlFieldValue f;
    f.fieldId = 82;
    if ((nvmlh->nvmlDeviceGetFieldValues(nvmlh->devs[gpuindex], 1, &f) != WRAPNVML_SUCCESS) || (f.nvmlReturn != WRAPNVML_SUCCESS)) return -1;
    *tempC = f.value.uiVal;
    return 0;
}

int wrap_nvml_get_fanpcnt(wrap_nvml_handle* nvmlh, int gpuindex, unsigned int* fanpcnt) {
    if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount) return -1;

    if (nvmlh->nvmlDeviceGetFanSpeed(nvmlh->devs[gpuindex], fanpcnt) != WRAPNVML_SUCCESS) return -1;

    return 0;
}

int wrap_nvml_get_power_usage(wrap_nvml_handle* nvmlh, int gpuindex, unsigned int* milliwatts) {
    if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount) return -1;

    if (nvmlh->nvmlDeviceGetPowerUsage(nvmlh->devs[gpuindex], milliwatts) != WRAPNVML_SUCCESS) return -1;

    return 0;
}

#if defined(__cplusplus)
}
#endif
