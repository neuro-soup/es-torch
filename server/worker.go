package main

import (
	"log/slog"
	"sync"
	"time"

	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed"
)

type worker struct {
	joinedAt time.Time

	lastHeartBeat time.Time
	ping          time.Duration

	numCPUs uint8
	rewards []byte

	events     chan *distributed.SubscribeResponse
	disconnect chan struct{}
}

func newWorker(numCPUs uint8) *worker {
	return &worker{
		joinedAt:      time.Now(),
		lastHeartBeat: time.Now(),
		numCPUs:       numCPUs,
		events:        make(chan *distributed.SubscribeResponse, 15),
		disconnect:    make(chan struct{}, 1),
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

	w, ok := wp.workers[id]
	if !ok {
		return
	}
	w.disconnect <- struct{}{}
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

func (wp *workerPool) broadcast(evt *distributed.SubscribeResponse) {
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
	i := 0
	for _, w := range wp.workers {
		rewards[i] = w.rewards
		i++
	}
	return rewards
}

func (wp *workerPool) trusted(not uint8) (uint8, *worker) {
	wp.mu.RLock()
	defer wp.mu.RUnlock()

	var (
		trustedID uint8
		trusted   *worker
	)
	for id, w := range wp.workers {
		if trusted == nil || w.ping < trusted.ping || id != not {
			trustedID = id
			trusted = w
		}
	}
	return trustedID, trusted
}

func (wp *workerPool) resetRewards() {
	wp.mu.Lock()
	defer wp.mu.Unlock()

	for _, w := range wp.workers {
		w.rewards = nil
	}
}

func (wp *workerPool) cleanTimeouts() {
	wp.mu.Lock()
	defer wp.mu.Unlock()

	for id, w := range wp.workers {
		if time.Since(w.lastHeartBeat) > time.Minute {
			slog.Error("worker timed out", "worker_id", id)
			w.disconnect <- struct{}{}
			delete(wp.workers, id)
		}
	}
}
