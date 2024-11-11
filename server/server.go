package main

import (
	"log/slog"
	"sync"
	"time"

	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed/distributedconnect"
)

type params struct {
	sync.RWMutex

	// initialized is true if the server has received the first subscription
	// request from a worker.
	initialized bool

	// numPop is the number of population to be evaluated.
	numPop uint32
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

	// rewards are the rewards received from each worker, where `[]byte` is the
	// encoded reward (most likely a tensor).
	rewards   [][]byte
	rewardsMu sync.RWMutex
}

var _ distributedconnect.ESServiceHandler = (*server)(nil)

// newServer creates a new server.
func newServer() *server {
	s := &server{
		workers: newWorkerPool(),
		slices:  new(slices),
		params:  new(params),
	}
	go s.watch()
	return s
}

// watch loops every minute and disconnects workers that have not sent a heartbeat
// within a minute.
func (s *server) watch() {
	for range time.Tick(time.Minute) {
		for id, w := range s.workers.iter() {
			if time.Since(w.lastHeartBeat) > time.Minute {
				slog.Error("worker timed out", "worker_id", id)
				s.disconnect(id, w)
			}
		}
	}
}

// disconnect disconnects a worker from the server.
func (s *server) disconnect(id uint8, w *worker) {
	w.disconnect <- struct{}{}
	s.workers.remove(id)
}
