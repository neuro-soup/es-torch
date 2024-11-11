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
