package main

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/bufbuild/connect-go"
	"github.com/neuro-soup/es-torch/server/pkg/proto/es"
	"github.com/neuro-soup/es-torch/server/pkg/proto/es/esconnect"
)

type server struct {
	workers *workerPool

	// hellos are the workers that have sent their first hello requests and
	// are waiting for their state to be sent.
	hellos   []*worker
	hellosMu *sync.RWMutex
}

var _ esconnect.ESServiceHandler = (*server)(nil)

func newServer() *server {
	return &server{
		workers:  newWorkerPool(),
		hellosMu: new(sync.RWMutex),
	}
}

func (s *server) Hello(
	ctx context.Context,
	req *connect.Request[es.HelloRequest],
) (*connect.Response[es.HelloResponse], error) {
	now := time.Now()
	slog.Debug("received hello request", "timestamp", now)

	// add worker to pool
	w := newWorker(uint8(req.Msg.NumCpus))
	id := s.workers.add(w)
	slog.Debug("added worker", "id", id, "timestamp", now)

	// add worker to hellos list
	s.hellosMu.Lock()
	s.hellos = append(s.hellos, w)
	s.hellosMu.Unlock()

	// request state from one of the workers
	s.workers.random().events <- &es.SubscribeResponse{
		Type: es.ServerEventType_SEND_STATE,
	}

	// TODO: maybe request state from a small percentage of multiple workers for
	// fault tolerance?

	return connect.NewResponse(&es.HelloResponse{Id: int32(id)}), nil
}

func (s *server) Done(
	ctx context.Context,
	req *connect.Request[es.DoneRequest],
) (*connect.Response[es.DoneResponse], error) {
	slog.Debug("received done request", "worker_id", req.Msg.Id)

	w := s.workers.get(uint8(req.Msg.Id))
	if w == nil {
		slog.Error("worker not found", "worker_id", req.Msg.Id)
		return nil, errors.New("worker not found")
	}

	w.rewards = req.Msg.Reward
	slog.Debug("updated worker rewards", "worker_id", req.Msg.Id)

	if s.workers.done() {
		slog.Debug("all worker rewards received, triggering next epoch...")
		s.workers.resetRewards()
		s.workers.broadcast(&es.SubscribeResponse{
			Type:    es.ServerEventType_NEXT_EPOCH,
			Rewards: s.workers.rewards(),
		})
		slog.Debug("broadcasted next epoch event to all workers", "workers", len(s.workers.workers))
	}

	return connect.NewResponse(&es.DoneResponse{}), nil
}

func (s *server) Heartbeat(
	ctx context.Context,
	req *connect.Request[es.HeartbeatRequest],
) (*connect.Response[es.HeartbeatResponse], error) {
	now := time.Now()
	slog.Debug("received heartbeat request", "timestamp", now)

	w := s.workers.get(uint8(req.Msg.Id))
	if w == nil {
		slog.Error("worker not found", "worker_id", req.Msg.Id)
		return nil, errors.New("worker not found")
	}

	w.lastHeartBeat = now

	slog.Debug("acknowledged worker heartbeat", "worker_id", req.Msg.Id)
	return connect.NewResponse(&es.HeartbeatResponse{Ok: true}), nil
}

func (s *server) SendState(
	ctx context.Context,
	req *connect.Request[es.SendStateRequest],
) (*connect.Response[es.SendStateResponse], error) {
	slog.Debug("received send state request", "worker_id", req.Msg.Id)

	w := s.workers.get(uint8(req.Msg.Id))
	if w == nil {
		slog.Error("worker not found", "worker_id", req.Msg.Id)
		return nil, errors.New("worker not found")
	}

	s.hellosMu.Lock()
	evt := &es.SubscribeResponse{
		Type:         es.ServerEventType_STATE_UPDATE,
		UpdatedState: req.Msg.State,
	}
	for _, hello := range s.hellos {
		hello.events <- evt
	}
	s.hellos = nil
	s.hellosMu.Unlock()

	slog.Debug("sent state update to new workers", "worker_id", req.Msg.Id)
	return connect.NewResponse(&es.SendStateResponse{}), nil
}

func (s *server) Subscribe(
	ctx context.Context,
	req *connect.Request[es.SubscribeRequest],
	stream *connect.ServerStream[es.SubscribeResponse],
) error {
	slog.Debug("received subscribe request", "worker_id", req.Msg.Id)

	w := s.workers.get(uint8(req.Msg.Id))
	if w == nil {
		slog.Error("worker not found", "worker_id", req.Msg.Id)
		return errors.New("worker not found")
	}

	go func() {
		for {
			select {
			case <-ctx.Done():
				slog.Debug("received cancel signal", "worker_id", req.Msg.Id)
				s.workers.remove(uint8(req.Msg.Id))
				break

			case evt := <-w.events: // TODO: satisfy govet
				if err := stream.Send(evt); err != nil {
					slog.Error("failed to send event", "err", err, "worker_id", req.Msg.Id)
				}
				slog.Debug("sent event", "worker_id", req.Msg.Id, "event", evt.Type.String())
			}
		}
	}()

	slog.Debug("subscribed to worker events", "worker_id", req.Msg.Id)
	return nil
}
