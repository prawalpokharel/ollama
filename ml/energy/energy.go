// Package energy reads the cumulative energy counters of NVIDIA GPUs.
//
// NVML is initialized when a meter is opened and shut down when it is closed,
// so no GPU device files stay open once every meter is closed. On the drivers
// tested, NVML also loads libcuda, which stays mapped after NVML is unloaded.
package energy

import "errors"

// ErrUnsupported is returned by Open on platforms where NVML isn't used.
var ErrUnsupported = errors.New("GPU energy counters are not supported on this platform")

// Meter reads the summed energy counters of a fixed set of GPUs.
type Meter interface {
	// Read returns the summed counters in millijoules. The counters only
	// increase while the meter is open, so the difference between two reads
	// is the energy the GPUs drew in between.
	Read() (uint64, error)
	Close() error
}
