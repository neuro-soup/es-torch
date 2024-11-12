package main

import (
	"fmt"
	"log/slog"
	"strings"
	"sync"
)

type slice struct {
	start   uint32
	end     uint32
	worker  *worker
	rewards [][]byte
}

type slices struct {
	sync.RWMutex

	numPop uint32
	slices []*slice
}

func newSlices(numPop uint32) *slices {
	sl := new(slices)
	sl.reset(numPop)
	return sl
}

func (s *slices) String() string {
	var sb strings.Builder
	for _, sl := range s.slices {
		if sl.worker != nil {
			sb.WriteString(fmt.Sprintf("slice %d-%d (assigned)\n", sl.start, sl.end))
		} else {
			sb.WriteString(fmt.Sprintf("slice %d-%d (unassigned)\n", sl.start, sl.end))
		}
	}
	return fmt.Sprintf("slices: %d\n", len(s.slices)) + sb.String()
}

func (s *slices) reset(numPop uint32) {
	s.Lock()
	defer s.Unlock()

	s.numPop = numPop
	s.slices = []*slice{
		{
			start: uint32(0),
			end:   uint32(numPop),
		},
	}
}

func (s *slices) find(start, end uint32) *slice {
	s.Lock()
	defer s.Unlock()

	for _, sl := range s.slices {
		if sl.start == start && sl.end == end {
			return sl
		}
	}

	return nil
}

func (s *slices) assign(id uint8, w *worker) *slice {
	s.Lock()
	defer s.Unlock()

	// TODO: optimise for non-contiguous slices (array of slices)

	slog.Debug("assigning slice to worker...", "num_cpus", w.numCPUs, "worker_id", id)

	fmt.Println("\nBefore assignment:")
	fmt.Println(s)
	defer func() {
		fmt.Println("\nAfter assignment:")
		fmt.Println(s)
	}()

	for i, sl := range s.slices {
		if sl.worker != nil {
			// ignore slices assigned to other workers
			continue
		}
		if sl.rewards != nil {
			// ignore slices with rewards
			continue
		}

		// claim slice to worker
		sl.worker = w

		diff := sl.end - sl.start
		if diff <= uint32(w.numCPUs) {
			// claim existing slice fits on worker's CPUs
			return sl
		}

		// split slice into multiple slices
		sl.end = sl.start + uint32(w.numCPUs)

		newSl := &slice{
			start: sl.end,
			end:   sl.end + diff - uint32(w.numCPUs),
		}

		right := s.slices[i+1:]
		s.slices = append(s.slices[:i+1], newSl)
		s.slices = append(s.slices, right...)

		return sl
	}

	return nil
}

func (s *slices) free(w *worker) {
	s.Lock()
	defer s.Unlock()

	for _, sl := range s.slices {
		if sl.worker == w {
			// free slice
			sl.worker = nil
		}
	}
}

func (s *slices) isEpochDone() bool {
	s.RLock()
	defer s.RUnlock()

	for _, sl := range s.slices {
		if sl.worker == nil || sl.rewards == nil {
			return false
		}
	}
	return true
}

func (s *slices) rewards() [][]byte {
	s.RLock()
	defer s.RUnlock()

	rewards := make([][]byte, s.numPop)
	i := 0
	for _, sl := range s.slices {
		for _, reward := range sl.rewards {
			rewards[i] = reward
			i++
		}
	}
	return rewards
}
