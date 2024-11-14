package server

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/bufbuild/connect-go"
	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed"
)

func (h *Handler) SendState(
	ctx context.Context,
	req *connect.Request[distributed.SendStateRequest],
) (*connect.Response[distributed.SendStateResponse], error) {
	slog.Debug("received send state request", "id", req.Msg.Id)

	w := h.workers.Get(uint8(req.Msg.Id))
	if w == nil {
		slog.Error("worker not found", "id", req.Msg.Id)
		return nil, connect.NewError(connect.CodeNotFound, fmt.Errorf(
			"handler: send state: worker %d not found",
			req.Msg.Id,
		))
	}

	w.ReceiveState(req.Msg.State)

	slog.Debug("sent state update to new workers", "id", req.Msg.Id)
	return connect.NewResponse(new(distributed.SendStateResponse)), nil
}
