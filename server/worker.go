package main

import (
	"iter"
	"sync"
	"time"

	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed"
)

type worker struct {
	numCPUs  uint8
	joinedAt time.Time

	lastHeartBeat time.Time
	ping          time.Duration

	events     chan *distributed.SubscribeResponse
	disconnect chan struct{}
}

func newWorker(numCPUs uint8) *worker {
	return &worker{
		numCPUs:  numCPUs,
		joinedAt: time.Now(),

		lastHeartBeat: time.Now(),

		events:     make(chan *distributed.SubscribeResponse, 15),
		disconnect: make(chan struct{}, 1),
	}
}

type workerPool struct {
	workers map[uint8]*worker
	nextID  uint8
	mu      *sync.RWMutex
}

func newWorkerPool() *workerPool {
	return &workerPool{
		workers: make(map[uint8]*worker),
		nextID:  1,
		mu:      new(sync.RWMutex),
	}
}

func (wp *workerPool) read(fn func(workers map[uint8]*worker)) {
	wp.mu.RLock()
	defer wp.mu.RUnlock()

	fn(wp.workers)
}

func (wp *workerPool) write(fn func(workers map[uint8]*worker)) {
	wp.mu.Lock()
	defer wp.mu.Unlock()

	fn(wp.workers)
}

func (wp *workerPool) len() (l int) {
	wp.read(func(workers map[uint8]*worker) { l = len(workers) })
	return l
}

func (wp *workerPool) add(w *worker) (id uint8) {
	wp.write(func(workers map[uint8]*worker) {
		id = wp.nextID
		workers[id] = w
		wp.nextID++
	})
	return id
}

func (wp *workerPool) remove(id uint8) {
	wp.write(func(workers map[uint8]*worker) {
		delete(wp.workers, id)
	})
}

func (wp *workerPool) get(id uint8) (found *worker) {
	wp.read(func(workers map[uint8]*worker) { found = wp.workers[id] })
	return found
}

func (wp *workerPool) iter() iter.Seq2[uint8, *worker] {
	return func(yield func(uint8, *worker) bool) {
		wp.mu.Lock()
		defer wp.mu.Unlock()

		for id, w := range wp.workers {
			if !yield(id, w) {
				break
			}
		}
	}
}

func (wp *workerPool) slice() (sl []*worker) {
	wp.read(func(workers map[uint8]*worker) {
		sl = make([]*worker, len(workers))
		i := 0
		for _, w := range workers {
			sl[i] = w
			i++
		}
	})
	return sl
}

func (wp *workerPool) trusted(not uint8) (trustedID uint8, trusted *worker) {
	for id, w := range wp.iter() {
		if trusted == nil || w.ping < trusted.ping || id != not {
			trustedID = id
			trusted = w
		}
	}
	return trustedID, trusted
}
