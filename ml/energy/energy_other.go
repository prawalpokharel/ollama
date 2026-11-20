//go:build !linux || !cgo

package energy

func Open(pciBusIDs []string) (Meter, error) {
	return nil, ErrUnsupported
}
