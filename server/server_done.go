package main

import (
	"context"
	"log/slog"

	"github.com/bufbuild/connect-go"
	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed"
)

func (s *server) Done(
	ctx context.Context,
	req *connect.Request[distributed.DoneRequest],
) (*connect.Response[distributed.DoneResponse], error) {
	slog.Debug("received done request", "worker_id", req.Msg.Id)

	return connect.NewResponse(&distributed.DoneResponse{}), nil
}
