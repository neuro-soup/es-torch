package worker

import "fmt"

// Slice represents a contiguous part in the population.
type Slice struct {
	// Start is the start of the slice in the population (inclusive).
	Start uint32

	// End is the end of the slice in the population (exclusive).
	End uint32
}

func (s Slice) String() string {
	return fmt.Sprintf("Slice(Start=%d, End=%d)", s.Start, s.End)
}

// Width returns the width of the slice.
func (s Slice) Width() uint32 {
	return s.End - s.Start
}
