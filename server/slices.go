package main

import (
	"sort"
	"sync"
)

type slice struct {
	start  uint32
	end    uint32
	worker *worker
}

type slices struct {
	sync.RWMutex

	// assignments are the slices allocated to each worker. Each worker can have
	// multiple slices assigned to it, which they process sequentially.
	assignments []slice

	// inconsistent is true if one or more workers left or joined the
	// experiment during the current epoch.
	inconsistent bool
}

// assign assigns batch slices to each worker. One worker can have multiple
// slices assigned to it, which they process sequentially. Each worker slice
// has a slice width of maximum `numCPUs` of the worker.
func (s *slices) assign(numPop uint32, pool *workerPool) {
	s.Lock()
	defer s.Unlock()

	workers := pool.slice()
	sort.Slice(workers, func(i, j int) bool {
		// sort by number of CPUs (descending)
		return workers[i].numCPUs > workers[j].numCPUs
	})

	// reset slices
	s.assignments = nil

	offset := uint32(0)
	remaining := numPop
	for remaining > 0 {
		for _, w := range workers {
			width := min(uint32(w.numCPUs), remaining)
			s.assignments = append(s.assignments, slice{
				start:  uint32(offset),
				end:    uint32(offset) + width,
				worker: w,
			})
			offset += width
			remaining -= width
		}
	}
}
