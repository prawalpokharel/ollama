//go:build linux && cgo

package energy

/*
#cgo LDFLAGS: -ldl

#include <dlfcn.h>
#include <stdlib.h>

typedef int (*nvml_init_fn)(void);
typedef int (*nvml_shutdown_fn)(void);
typedef int (*nvml_device_by_pci_bus_id_fn)(const char *, void **);
typedef int (*nvml_device_total_energy_fn)(void *, unsigned long long *);

static int call_nvml_init(void *fn) {
	return ((nvml_init_fn) fn)();
}

static int call_nvml_shutdown(void *fn) {
	return ((nvml_shutdown_fn) fn)();
}

static int call_nvml_device_by_pci_bus_id(void *fn, const char *id, void **device) {
	return ((nvml_device_by_pci_bus_id_fn) fn)(id, device);
}

static int call_nvml_device_total_energy(void *fn, void *device, unsigned long long *mj) {
	return ((nvml_device_total_energy_fn) fn)(device, mj);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

type nvmlMeter struct {
	mu       sync.RWMutex
	lib      unsafe.Pointer
	shutdown unsafe.Pointer
	energy   unsafe.Pointer
	devices  []unsafe.Pointer
}

// Open loads NVML and finds the GPUs with the given PCI bus IDs. It fails if
// any of them has no energy counter, so that a reading always covers every GPU.
func Open(pciBusIDs []string) (Meter, error) {
	if len(pciBusIDs) == 0 {
		return nil, errors.New("no GPUs to meter")
	}

	var lib unsafe.Pointer
	for _, name := range []string{"libnvidia-ml.so.1", "libnvidia-ml.so"} {
		cname := C.CString(name)
		lib = C.dlopen(cname, C.RTLD_NOW|C.RTLD_LOCAL)
		C.free(unsafe.Pointer(cname))
		if lib != nil {
			break
		}
	}
	if lib == nil {
		return nil, errors.New("unable to load NVML")
	}

	m := &nvmlMeter{lib: lib}
	var initFn, byPCIFn unsafe.Pointer
	for _, sym := range []struct {
		name string
		dst  *unsafe.Pointer
	}{
		{"nvmlInit_v2", &initFn},
		{"nvmlShutdown", &m.shutdown},
		{"nvmlDeviceGetHandleByPciBusId_v2", &byPCIFn},
		{"nvmlDeviceGetTotalEnergyConsumption", &m.energy},
	} {
		cname := C.CString(sym.name)
		*sym.dst = C.dlsym(lib, cname)
		C.free(unsafe.Pointer(cname))
		if *sym.dst == nil {
			C.dlclose(lib)
			return nil, fmt.Errorf("NVML has no %s", sym.name)
		}
	}

	if ret := C.call_nvml_init(initFn); ret != 0 {
		C.dlclose(lib)
		return nil, fmt.Errorf("nvmlInit_v2 failed: %d", int(ret))
	}

	for _, id := range pciBusIDs {
		cid := C.CString(id)
		var device unsafe.Pointer
		ret := C.call_nvml_device_by_pci_bus_id(byPCIFn, cid, &device)
		C.free(unsafe.Pointer(cid))
		if ret != 0 {
			m.Close()
			return nil, fmt.Errorf("no NVML device for PCI bus ID %s: %d", id, int(ret))
		}
		m.devices = append(m.devices, device)
	}

	if _, err := m.Read(); err != nil {
		m.Close()
		return nil, err
	}

	return m, nil
}

func (m *nvmlMeter) Read() (uint64, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.lib == nil {
		return 0, errors.New("meter is closed")
	}

	var total uint64
	for _, device := range m.devices {
		var mj C.ulonglong
		if ret := C.call_nvml_device_total_energy(m.energy, device, &mj); ret != 0 {
			return 0, fmt.Errorf("nvmlDeviceGetTotalEnergyConsumption failed: %d", int(ret))
		}
		total += uint64(mj)
	}
	return total, nil
}

func (m *nvmlMeter) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.lib == nil {
		return nil
	}
	ret := C.call_nvml_shutdown(m.shutdown)
	C.dlclose(m.lib)
	m.lib = nil
	if ret != 0 {
		return fmt.Errorf("nvmlShutdown failed: %d", int(ret))
	}
	return nil
}
