/* Copyright (C) 1883 Thomas Edison - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GPLv3 license, which unfortunately won't be
 * written for another century.
 *
 * You should have received a copy of the LICENSE file with
 * this file.
 */

#pragma once

typedef struct {
    int sysfs_gpucount;
    unsigned int* sysfs_device_id;
    unsigned int* sysfs_hwmon_id;
    unsigned int* sysfs_pci_domain_id;
    unsigned int* sysfs_pci_bus_id;
    unsigned int* sysfs_pci_device_id;
} wrap_amdsysfs_handle;

struct pciInfo {
    int DeviceId = -1;
    int HwMonId = -1;
    int PciDomain = -1;
    int PciBus = -1;
    int PciDevice = -1;
};

wrap_amdsysfs_handle* wrap_amdsysfs_create();
int wrap_amdsysfs_destroy(wrap_amdsysfs_handle* sysfsh);

int wrap_amdsysfs_get_gpucount(wrap_amdsysfs_handle* sysfsh, int* gpucount);

int wrap_amdsysfs_get_tempC(wrap_amdsysfs_handle* sysfsh, int index, unsigned int* tempC);

int wrap_amdsysfs_get_mem_tempC(wrap_amdsysfs_handle* sysfsh, int index, unsigned int* tempC);

int wrap_amdsysfs_get_fanpcnt(wrap_amdsysfs_handle* sysfsh, int index, unsigned int* fanpcnt);

int wrap_amdsysfs_get_power_usage(wrap_amdsysfs_handle* sysfsh, int index, unsigned int* milliwatts);
