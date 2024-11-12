package main

import (
	"log/slog"
	"sync"
	"time"

	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed/distributedconnect"
)

const (
	// watchInterval is the interval at which the server watches for workers
	// that have not sent a heartbeat.
	watchInterval = 10 * time.Second

	// heartbeatTimeout is the duration after which a worker is disconnected if
	// it hasn't sent a heartbeat.
	heartbeatTimeout = 10 * time.Second
)

type params struct {
	sync.RWMutex

	// initialized is true if the server has received the first subscription
	// request from a worker.
	initialized bool

	// numPop is the number of population to be evaluated.
	numPop uint32
}

func (p *params) isInitialized() bool {
	p.RLock()
	defer p.RUnlock()
	return p.initialized
}

type server struct {
	// workers is the pool of workers connected to the server.
	workers *workerPool

	// params are the server's parameters which are sent by the first worker
	// that subscribes to the server.
	params *params

	// slices are the slices assigned to each worker.
	slices *slices

	// hellos are the workers that have sent their first hello requests and
	// are waiting for their state to be sent.
	hellos   []*worker
	hellosMu sync.RWMutex
}

var _ distributedconnect.ESServiceHandler = (*server)(nil)

// newServer creates a new server.
func newServer() *server {
	s := &server{
		workers: newWorkerPool(),
		params:  new(params),
	}
	go s.watch()
	return s
}

// watch loops every minute and disconnects workers that have not sent a heartbeat
// within a minute.
func (s *server) watch() {
	for range time.Tick(watchInterval) {
		for id, w := range s.workers.iter() {
			if !w.lastHeartBeat.IsZero() && time.Since(w.lastHeartBeat) > heartbeatTimeout {
				slog.Error("worker timed out", "worker_id", id)
				s.disconnect(id, w)
			}
		}
	}
}

func (s *server) clean(id uint8, w *worker) {
	s.workers.remove(id)
	s.slices.free(w)
}

// disconnect disconnects a worker from the server and removes it from the
// worker pool.
func (s *server) disconnect(id uint8, w *worker) {
	slog.Debug("disconnecting worker", "worker_id", id)

	w.disconnect <- struct{}{}
	s.clean(id, w)
}
