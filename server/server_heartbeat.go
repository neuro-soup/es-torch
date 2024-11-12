package main

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/bufbuild/connect-go"
	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed"
)

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
	w.ping = time.Since(req.Msg.Timestamp.AsTime())

	if w.ping > heartbeatTimeout {
		// disconnect worker if the heartbeat was transmitted more than 1 minute ago
		slog.Error("worker ping timeout", "worker_id", req.Msg.Id)
		s.clean(uint8(req.Msg.Id), w)
		return nil, errors.New("heartbeat was too slow (>=1 minute ago)")
	}

	slog.Debug("acknowledged worker heartbeat", "worker_id", req.Msg.Id)
	return connect.NewResponse(&distributed.HeartbeatResponse{Ok: true}), nil
}
