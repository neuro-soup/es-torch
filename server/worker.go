package main

import (
	"sync"
	"time"

	"github.com/neuro-soup/es-torch/server/pkg/proto/es"
)

type worker struct {
	joinedAt      time.Time
	lastHeartBeat time.Time

	numCPUs uint8
	rewards []byte

	events chan *es.SubscribeResponse
}

func newWorker(numCPUs uint8) *worker {
	return &worker{
		joinedAt:      time.Now(),
		lastHeartBeat: time.Now(),
		numCPUs:       numCPUs,
		events:        make(chan *es.SubscribeResponse, 15),
	}
}

type workerPool struct {
	nextID  uint8
	workers map[uint8]*worker
	mu      *sync.RWMutex
}

func newWorkerPool() *workerPool {
	return &workerPool{
		nextID:  1,
		workers: make(map[uint8]*worker),
		mu:      new(sync.RWMutex),
	}
}

func (wp *workerPool) add(w *worker) uint8 {
	wp.mu.Lock()
	defer wp.mu.Unlock()

	id := wp.nextID
	wp.workers[id] = w
	wp.nextID++

	return id
}

func (wp *workerPool) remove(id uint8) {
	wp.mu.Lock()
	defer wp.mu.Unlock()

	delete(wp.workers, id)
}

func (wp *workerPool) get(id uint8) *worker {
	wp.mu.RLock()
	defer wp.mu.RUnlock()

	w, ok := wp.workers[id]
	if !ok {
		return nil
	}
	return w
}

func (wp *workerPool) done() bool {
	wp.mu.RLock()
	defer wp.mu.RUnlock()

	for _, w := range wp.workers {
		if w.rewards == nil {
			return false
		}
	}
	return true
}

func (wp *workerPool) broadcast(evt *es.SubscribeResponse) {
	wp.mu.Lock()
	defer wp.mu.Unlock()

	for _, w := range wp.workers {
		w.events <- evt
	}
}

func (wp *workerPool) rewards() [][]byte {
	wp.mu.RLock()
	defer wp.mu.RUnlock()

	rewards := make([][]byte, len(wp.workers))
	for i, w := range wp.workers {
		rewards[i] = w.rewards
	}
	return rewards
}

func (wp *workerPool) random() *worker {
	wp.mu.RLock()
	defer wp.mu.RUnlock()

	var w *worker
	for _, w = range wp.workers {
		break
	}
	return w
}

func (wp *workerPool) resetRewards() {
	wp.mu.Lock()
	defer wp.mu.Unlock()

	for _, w := range wp.workers {
		w.rewards = nil
	}
}
