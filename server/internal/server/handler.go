package server

import (
	"sync"

	"github.com/bufbuild/connect-go"
	"github.com/neuro-soup/es-torch/server/internal/epoch"
	"github.com/neuro-soup/es-torch/server/internal/worker"
	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed"
	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed/distributedconnect"
)

type Handler struct {
	workers *worker.Pool

	initParamsSubscription sync.Once
	population             uint32

	epoch *epoch.Epoch
}

var _ distributedconnect.ESServiceHandler = (*Handler)(nil)

// New creates a new handler.
func New(workers *worker.Pool) *Handler {
	return &Handler{workers: workers}
}

type (
	heartbeatRequest      = connect.Request[distributed.HeartbeatRequest]
	heartbeatResponse     = connect.Response[distributed.HeartbeatResponse]
	subscribeRequest      = connect.Request[distributed.SubscribeRequest]
	subscribeResponse     = connect.Response[distributed.SubscribeResponse]
	subscribeServerStream = connect.ServerStream[distributed.SubscribeResponse]
	doneRequest           = connect.Request[distributed.DoneRequest]
	doneResponse          = connect.Response[distributed.DoneResponse]
)
