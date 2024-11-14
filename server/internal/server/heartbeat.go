package server

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/bufbuild/connect-go"
	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed"
)

func (h *Handler) Heartbeat(ctx context.Context, req *heartbeatRequest) (*heartbeatResponse, error) {
	now := time.Now()
	slog.Debug("received heartbeat request", "timestamp", now)

	w := h.workers.Get(uint8(req.Msg.Id))
	if w == nil {
		slog.Error("heartbeat request for unknown worker", "id", req.Msg.Id)
		return nil, connect.NewError(connect.CodeNotFound, fmt.Errorf(
			"handler: heartbeat: worker not found: %d",
			req.Msg.Id,
		))
	}

	if err := w.ReceiveHeartbeat(req.Msg.Timestamp.AsTime()); err != nil {
		slog.Error("failed to receive heartbeat", "err", err)
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	slog.Debug("acknowledged worker heartbeat", "worker_id", req.Msg.Id)
	return connect.NewResponse(&distributed.HeartbeatResponse{Ok: true}), nil
}
