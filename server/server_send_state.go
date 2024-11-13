package main

import (
	"context"
	"errors"
	"log/slog"

	"github.com/bufbuild/connect-go"
	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed"
)

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
		hello.events <- &distributed.SubscribeResponse{
			Type: distributed.ServerEventType_HELLO,
			Event: &distributed.SubscribeResponse_Hello{
				Hello: &distributed.HelloEvent{
					Id:        int32(id),
					InitState: req.Msg.State,
				},
			},
		}

		if next := s.slices.assign(uint8(id), hello); next != nil {
			slog.Debug("sending worker next batch", "worker_id", id, "slice", next)
			hello.events <- &distributed.SubscribeResponse{
				Type: distributed.ServerEventType_EVALUATE_BATCH,
				Event: &distributed.SubscribeResponse_EvaluateBatch{
					EvaluateBatch: &distributed.EvaluateBatchEvent{
						PopSlice: &distributed.Slice{
							Start: int32(next.start),
							End:   int32(next.end),
						},
					},
				},
			}
		}
	}
	clear(s.hellos)
	s.hellosMu.Unlock()

	slog.Debug("sent state update to new workers", "worker_id", req.Msg.Id)
	return connect.NewResponse(&distributed.SendStateResponse{}), nil
}
