package main

import (
	"iter"
	"sync"
	"time"

	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed"
)

type worker struct {
	// numCPUs is the number of CPUs the worker has. This value corresponds to
	// the width of the slices that are assigned to the worker.
	numCPUs uint8

	device string

	// joinedAt is the time the worker joined the experiment.
	joinedAt time.Time

	// lastHeartBeat is the time the worker last sent a heartbeat.
	lastHeartBeat time.Time
	ping          time.Duration

	// ping is the time the worker took to send a heartbeat.
	ping time.Duration

	// events is the channel that sends worker events to the worker.
	events chan *distributed.SubscribeResponse

	// disconnect is the channel that disconnects the worker as soon as a signal
	// is received.
	disconnect chan struct{}
}

// newWorker creates a new worker.
func newWorker(numCPUs uint8, device string) *worker {
	return &worker{
		numCPUs: numCPUs,
		device:  device,

		joinedAt: time.Now(),

		events:     make(chan *distributed.SubscribeResponse, 15),
		disconnect: make(chan struct{}, 1),
	}
}

// workerPool is a pool of workers.
type workerPool struct {
	sync.RWMutex

	// workers are the currently added workers.
	workers map[uint8]*worker

	// nextID is the id that will be assigned to the next worker that is added.
	nextID uint8
}

// newWorkerPool creates a new worker pool.
func newWorkerPool() *workerPool {
	return &workerPool{
		nextID:  1,
		workers: make(map[uint8]*worker),
		nextID:  1,
	}
}

// add adds a worker to the pool.
func (wp *workerPool) add(w *worker) (id uint8) {
	wp.Lock()
	defer wp.Unlock()

	id = wp.nextID
	wp.workers[id] = w
	wp.nextID++

	return id
}

// remove removes the worker with the given ID.
func (wp *workerPool) remove(id uint8) {
	wp.Lock()
	defer wp.Unlock()

	delete(wp.workers, id)
}

// get returns the worker with the given ID.
func (wp *workerPool) get(id uint8) *worker {
	wp.RLock()
	defer wp.RUnlock()

	return wp.workers[id]
}

// iter returns an iterator over the workers in the pool. The pool is locked
// during the iteration.
func (wp *workerPool) iter() iter.Seq2[uint8, *worker] {
	return func(yield func(uint8, *worker) bool) {
		wp.Lock()
		defer wp.Unlock()

		for id, w := range wp.workers {
			if !yield(id, w) {
				break
			}
		}
	}
}

// slice returns a slice of all workers in the pool.
func (wp *workerPool) slice() (sl []*worker) {
	wp.RLock()
	defer wp.RUnlock()

	sl = make([]*worker, len(wp.workers))
	i := 0
	for _, w := range wp.workers {
		sl[i] = w
		i++
	}
	return sl
}

// trusted returns the worker with the lowest ping time, excluding the worker
// with the given ID.
func (wp *workerPool) trusted(not uint8) (trustedID uint8, trusted *worker) {
	for id, w := range wp.iter() {
		if !w.lastHeartBeat.IsZero() && (trusted == nil || w.ping < trusted.ping) && id != not {
			trustedID = id
			trusted = w
		}
	}
	return trustedID, trusted
}

// returns the ID of the worker that has been in the pool the longest
func (wp *workerPool) oldestWorker() (oldestID uint8) {
	wp.RLock()
	defer wp.RUnlock()

	var oldestJoinTime time.Time
	for id, w := range wp.workers {
		if oldestJoinTime.IsZero() || w.joinedAt.Before(oldestJoinTime) {
			oldestJoinTime = w.joinedAt
			oldestID = id
		}
	}
	return oldestID
}

