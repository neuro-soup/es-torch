package worker

import (
	"fmt"
	"iter"
	"log/slog"
	"sync"
	"time"
)

type Pool struct {
	// mu protects the workers slice.
	mu sync.RWMutex

	// workers are the currently connected workers.
	workers []*Worker

	// nextID is the next ID to be assigned to a worker.
	nextID uint8
}

// NewPool creates a new worker pool.
func NewPool() *Pool {
	p := &Pool{nextID: 1}
	go p.watch()
	return p
}

func (p *Pool) watch() {
	for range time.Tick(heartbeatTimeout) {
		slog.Debug("checking for timed out workers...")
		for w := range p.Participating() {
			if time.Since(w.LastHeartbeat) > heartbeatTimeout {
				slog.Error("worker timed out", "worker", w.ID)
				w.Disconnect()
			}
		}
	}
}

func (p *Pool) String() string {
	return fmt.Sprintf("Pool(Len=%d)", p.Len())
}

// Create creates a new worker and adds it to the pool.
func (p *Pool) Create(cfg Config) *Worker {
	p.mu.Lock()
	defer p.mu.Unlock()

	w := newWorker(p.nextID, cfg)
	p.workers = append(p.workers, w)
	p.nextID++
	return w
}

// Remove removes a worker from the pool.
func (p *Pool) Remove(w *Worker) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	slog.Debug("removing worker from pool...", "worker", w.ID)

	for i, worker := range p.workers {
		if worker == w {
			p.workers = append(p.workers[:i], p.workers[i+1:]...)
			slog.Debug("removed worker from pool", "worker", w.ID)
			return nil
		}
	}

	slog.Error("cannot remove unknown worker from pool", "worker", w.ID)
	return fmt.Errorf("worker pool: worker %d not found", w.ID)
}

// Get returns the worker with the given ID. If the worker is not found, nil is
// returned.
func (p *Pool) Get(id uint8) *Worker {
	p.mu.RLock()
	defer p.mu.RUnlock()

	for _, w := range p.workers {
		if w.ID == id {
			return w
		}
	}
	return nil
}

// Len returns the number of workers in the pool.
func (p *Pool) Len() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.workers)
}

// Iter returns an iterator over all workers in the pool. The pool is
// read-locked during the iteration.
func (p *Pool) Iter(filter func(*Worker) bool) iter.Seq[*Worker] {
	return func(yield func(*Worker) bool) {
		p.mu.RLock()
		defer p.mu.RUnlock()

		for _, w := range p.workers {
			if filter != nil && !filter(w) {
				continue
			}
			if !yield(w) {
				break
			}
		}
	}
}

// Oldest returns the worker that first joined the pool. If no workers have
// joined, nil is returned.
func (p *Pool) Oldest(filter func(*Worker) bool) *Worker {
	var oldest *Worker
	for w := range p.Iter(filter) {
		if oldest == nil || w.JoinedAt.Before(oldest.JoinedAt) {
			oldest = w
		}
	}
	return oldest
}

// LowestPing returns the worker that has pinged the least. If no workers have
// pinged, nil is returned.
func (p *Pool) LowestPing(filter func(*Worker) bool) *Worker {
	var lowest *Worker
	for w := range p.Iter(filter) {
		if lowest == nil || w.Ping < lowest.Ping {
			lowest = w
		}
	}
	return lowest
}

func (p *Pool) Participating() iter.Seq[*Worker] {
	return p.Iter(func(w *Worker) bool { return w.Participates() })
}
