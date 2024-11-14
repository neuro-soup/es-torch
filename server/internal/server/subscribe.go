package server

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/bufbuild/connect-go"
	"github.com/neuro-soup/es-torch/server/internal/epoch"
	"github.com/neuro-soup/es-torch/server/internal/worker"
)

func (h *Handler) Subscribe(ctx context.Context, req *subscribeRequest, stream *subscribeServerStream) error {
	slog.Debug("received subscribe request",
		"num_cpus", req.Msg.NumCpus,
		"device", req.Msg.Device,
	)

	// initialise handler state
	h.initParamsSubscription.Do(func() { h.handleFirstSubscription(req) })

	cfg, err := h.workerCfgFromReq(req)
	if err != nil {
		return connect.NewError(connect.CodeInvalidArgument, err)
	}

	joined := h.workers.Create(cfg)
	trusted := h.workers.LowestPing(func(w *worker.Worker) bool { return w.Participates() })

	if trusted != nil {
		slog.Debug("found trusted worker to get state from", "trusted", trusted.ID)
		trusted.SendStateRequest(cfg.Device)

		select {
		case <-trusted.Disconnects():
			slog.Debug("trusted worker who was requested to send state disconnected", "trusted", trusted.ID)
			// TODO: re-elect trusted worker

		case <-joined.Disconnects():
			slog.Debug("worker who waited for state from trusted worker disconnected", "trusted", trusted.ID)
			joined.Destroy()
			return connect.NewError(connect.CodeUnavailable, fmt.Errorf("worker %d disconnected", trusted.ID))

		case state := <-trusted.States():
			slog.Debug("trusted worker sent state", "trusted", trusted.ID)
			joined.SendHello(state)
		}
	} else {
		slog.Debug("no trusted worker found, sending hello directly", "joined", joined.ID)
		joined.SendHello(nil)
	}

	slog.Debug("subscribed to worker events", "joined", joined.ID)

	if next := h.epoch.Assign(joined); next != nil {
		slog.Debug("sending worker next batch", "joined", joined.ID, "slice", next)
		joined.SendEvaluate(*next)
	} else {
		slog.Debug("no slices available for new worker", "joined", joined.ID)
	}

	for {
		select {
		case <-joined.Disconnects():
			// server-side cancellation
			slog.Debug("received server-side disconnect signal", "joined", joined.ID)

			if sl := joined.Evaluating(); sl != nil {
				// client didn't finish evaluating the slice, so we need to
				// send a done event to the client
			}

			joined.Destroy()
			return nil

		case evt := <-joined.Events():
			// send event to client
			slog.Debug("sending event", "joined", joined.ID, "event", evt.Type.String())
			if err := stream.Send(evt); err != nil {
				slog.Error("failed to send event", "err", err, "joined", joined.ID)
				joined.Disconnect()
				joined.Destroy()
				return err
			}
		}
	}
}

// handleFirstSubscription initialises the handler state from the first
// subscription request.
func (h *Handler) handleFirstSubscription(req *subscribeRequest) {
	slog.Info("obtaining parameters from first subscription", "num_pop", req.Msg.NumPop)
	h.population = uint32(req.Msg.NumPop)
	h.epoch = epoch.New(h.population) // initialise first epoch
}

// workerCfgFromReq returns the worker configuration from the given request. If
// the client-side configuration is not compatible with the server-side
// configuration, an error is returned.
func (h *Handler) workerCfgFromReq(req *subscribeRequest) (worker.Config, error) {
	if req.Msg.NumPop != int32(h.population) {
		return worker.Config{}, fmt.Errorf("handler: subscribe: population mismatch: expected %d, got %d", h.population, req.Msg.NumPop)
	}
	return worker.Config{
		NumCPUs: uint8(req.Msg.NumCpus),
		Device:  req.Msg.Device,
	}, nil
}
