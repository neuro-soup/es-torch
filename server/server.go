package main

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/bufbuild/connect-go"
	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed"
	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed/distributedconnect"
)

type server struct {
	workers *workerPool

	// hellos are the workers that have sent their first hello requests and
	// are waiting for their state to be sent.
	hellos   []*worker
	hellosMu *sync.RWMutex
}

var _ distributedconnect.ESServiceHandler = (*server)(nil)

func newServer() *server {
	return &server{
		workers:  newWorkerPool(),
		hellosMu: new(sync.RWMutex),
	}
}

func (s *server) Done(
	ctx context.Context,
	req *connect.Request[distributed.DoneRequest],
) (*connect.Response[distributed.DoneResponse], error) {
	slog.Debug("received done request", "worker_id", req.Msg.Id)

	w := s.workers.get(uint8(req.Msg.Id))
	if w == nil {
		slog.Error("worker not found", "worker_id", req.Msg.Id)
		return nil, errors.New("worker not found")
	}

	w.rewards = req.Msg.RewardBatch
	slog.Debug("updated worker rewards", "worker_id", req.Msg.Id)

	if s.workers.done() {
		slog.Debug("all worker rewards received, triggering next epoch...")
		s.workers.resetRewards()
		s.workers.broadcast(&distributed.SubscribeResponse{
			Type: distributed.ServerEventType_OPTIM_STEP,
			Event: &distributed.SubscribeResponse_OptimStep{
				OptimStep: &distributed.OptimStepEvent{
					Rewards: s.workers.rewards(),
				},
			},
		})
		slog.Debug("broadcasted next epoch event to all workers", "workers", len(s.workers.workers))
	}

	return connect.NewResponse(&distributed.DoneResponse{}), nil
}

func (s *server) Heartbeat(
	ctx context.Context,
	req *connect.Request[distributed.HeartbeatRequest],
) (*connect.Response[distributed.HeartbeatResponse], error) {
	now := time.Now()
	slog.Debug("received heartbeat request", "timestamp", now)

	w := s.workers.get(uint8(req.Msg.Id))
	if w == nil {
		slog.Error("worker not found", "worker_id", req.Msg.Id)
		return nil, errors.New("worker not found")
	}

	w.lastHeartBeat = now

	slog.Debug("acknowledged worker heartbeat", "worker_id", req.Msg.Id)
	return connect.NewResponse(&distributed.HeartbeatResponse{Ok: true}), nil
}

func (s *server) SendState(
	ctx context.Context,
	req *connect.Request[distributed.SendStateRequest],
) (*connect.Response[distributed.SendStateResponse], error) {
	slog.Debug("received send state request", "worker_id", req.Msg.Id)

	w := s.workers.get(uint8(req.Msg.Id))
	if w == nil {
		slog.Error("worker not found", "worker_id", req.Msg.Id)
		return nil, errors.New("worker not found")
	}

	s.hellosMu.Lock()
	for id, hello := range s.hellos {
		evt := &distributed.SubscribeResponse{
			Type: distributed.ServerEventType_HELLO,
			Event: &distributed.SubscribeResponse_Hello{
				Hello: &distributed.HelloEvent{
					Id:        int32(id),
					InitState: req.Msg.State,
				},
			},
		}
		hello.events <- evt
	}
	s.hellos = nil
	s.hellosMu.Unlock()

	slog.Debug("sent state update to new workers", "worker_id", req.Msg.Id)
	return connect.NewResponse(&distributed.SendStateResponse{}), nil
}

func (s *server) Subscribe(
	ctx context.Context,
	req *connect.Request[distributed.SubscribeRequest],
	stream *connect.ServerStream[distributed.SubscribeResponse],
) error {
	slog.Debug("received subscribe request", "num_cpus", req.Msg.NumCpus)

	w := newWorker(uint8(req.Msg.NumCpus))
	id := s.workers.add(w)

	go func() {
		for {
			select {
			case <-ctx.Done():
				slog.Debug("received cancel signal", "worker_id", id)
				s.workers.remove(id)
				break

			case evt := <-w.events: // TODO: satisfy govet
				if err := stream.Send(evt); err != nil {
					slog.Error("failed to send event", "err", err, "worker_id", id)
				}
				slog.Debug("sent event", "worker_id", id, "event", evt.Type.String())
			}
		}
	}()

	s.hellosMu.Lock()
	s.hellos = append(s.hellos, w)
	s.hellosMu.Unlock()

	trustedID, trusted := s.workers.trusted(id)
	if trusted != nil {
		slog.Debug("requesting state from trusted worker", "worker_id", id, "trusted_worker_id", trustedID)
		trusted.events <- &distributed.SubscribeResponse{
			Type: distributed.ServerEventType_SEND_STATE,
			Event: &distributed.SubscribeResponse_SendState{
				SendState: new(distributed.SendStateEvent),
			},
		}
	} else {
		slog.Debug("no trusted worker found, sending hello directly", "worker_id", id)
		evt := &distributed.SubscribeResponse{
			Type: distributed.ServerEventType_HELLO,
			Event: &distributed.SubscribeResponse_Hello{
				Hello: &distributed.HelloEvent{
					Id:        int32(id),
					InitState: nil,
				},
			},
		}
		w.events <- evt
	}

	slog.Debug("subscribed to worker events", "worker_id", id)
	return nil
}
