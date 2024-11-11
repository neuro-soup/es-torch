package main

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/bufbuild/connect-go"
	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed"
)

func (s *server) Subscribe(
	ctx context.Context,
	req *connect.Request[distributed.SubscribeRequest],
	stream *connect.ServerStream[distributed.SubscribeResponse],
) error {
	slog.Debug("received subscribe request", "num_cpus", req.Msg.NumCpus)

	if !s.params.initialized {
		s.handleFirstSubscription(req)
	}

	if err := s.validateSubscription(req); err != nil {
		slog.Error("failed to validate subscription", "err", err)
		return err
	}

	w := newWorker(uint8(req.Msg.NumCpus))
	id := s.workers.add(w)

	go s.streamWorkerEvents(ctx, stream, id, w)

	s.hellosMu.Lock()
	s.hellos = append(s.hellos, w)
	s.hellosMu.Unlock()

	trustedID, trusted := s.workers.trusted(id)
	if trusted != nil {
		// requesting state from trusted worker
		slog.Debug("requesting state from trusted worker", "worker_id", id, "trusted_worker_id", trustedID)
		trusted.events <- &distributed.SubscribeResponse{
			Type: distributed.ServerEventType_SEND_STATE,
			Event: &distributed.SubscribeResponse_SendState{
				SendState: new(distributed.SendStateEvent),
			},
		}
	} else {
		// first subscription or no trusted worker found, sending hello directly
		// without initial state
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

// validateSubscription validates the subscription request. Parameters, such asa
// number of population, must be equal across all workers.
func (s *server) validateSubscription(req *connect.Request[distributed.SubscribeRequest]) error {
	if s.params.numPop != uint32(req.Msg.NumPop) {
		return fmt.Errorf("number of population mismatch (expected %d, got %d)", s.params.numPop, req.Msg.NumPop)
	}
	return nil
}

// streamWorkerEvents streams worker events from the worker's event channel to
// over the GRPC stream to the worker client.
func (s *server) streamWorkerEvents(
	ctx context.Context,
	stream *connect.ServerStream[distributed.SubscribeResponse],
	id uint8, w *worker,
) {
	for {
		select {
		case <-ctx.Done():
			slog.Debug("received cancel signal", "worker_id", id)
			s.disconnect(id, w)
			break

		case <-w.disconnect:
			slog.Debug("received disconnect signal", "worker_id", id)
			s.disconnect(id, w)
			break

		case evt := <-w.events: // TODO: satisfy govet
			if err := stream.Send(evt); err != nil {
				slog.Error("failed to send event", "err", err, "worker_id", id)
			}
			slog.Debug("sent event", "worker_id", id, "event", evt.Type.String())
		}
	}
}

// handleFirstSubscription initializes the server state when the first
// worker subscribes to the server.
func (s *server) handleFirstSubscription(req *connect.Request[distributed.SubscribeRequest]) {
	slog.Debug("received subscribe request", "num_cpus", req.Msg.NumCpus)

	s.params.Lock()
	defer s.params.Unlock()
	s.params.initialized = true
	s.params.numPop = uint32(req.Msg.NumPop)

	s.rewardsMu.Lock()
	defer s.rewardsMu.Unlock()
	s.rewards = make([][]byte, req.Msg.NumPop)
}
