package server

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/bufbuild/connect-go"
	"github.com/neuro-soup/es-torch/server/internal/worker"
	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed"
)

func (h *Handler) Done(
	ctx context.Context,
	req *connect.Request[distributed.DoneRequest],
) (*connect.Response[distributed.DoneResponse], error) {
	slog.Debug("received done request",
		"id", req.Msg.Id,
		"slice", req.Msg.Slice,
		"rewards", len(req.Msg.BatchRewards),
	)

	w := h.workers.Get(uint8(req.Msg.Id))
	if w == nil {
		slog.Error("worker not found", "id", req.Msg.Id)
		return nil, connect.NewError(connect.CodeNotFound, fmt.Errorf(
			"handler: done: worker %d not found",
			req.Msg.Id,
		))
	}

	sl := worker.Slice{
		Start: uint32(req.Msg.Slice.Start),
		End:   uint32(req.Msg.Slice.End),
	}
	w.ReceiveDone(sl)

	if err := h.epoch.Reward(sl, req.Msg.BatchRewards); err != nil {
		slog.Error("failed to reward", "id", req.Msg.Id, "err", err)
		return nil, connect.NewError(connect.CodeInternal, fmt.Errorf(
			"handler: done: failed to reward worker %d: %w",
			req.Msg.Id, err,
		))
	}

	if h.epoch.Done() {
		return h.handleEpochDone(req)
	}

	if next := h.epoch.Assign(w); next != nil {
		slog.Debug("sending worker next batch", "id", req.Msg.Id, "slice", next)
		w.SendEvaluate(*next)
		return connect.NewResponse(new(distributed.DoneResponse)), nil
	}

	slog.Debug("no more slices, waiting for other workers to finish", "id", req.Msg.Id)
	return connect.NewResponse(new(distributed.DoneResponse)), nil
}

func (h *Handler) handleEpochDone(req *doneRequest) (*doneResponse, error) {
	slog.Debug("epoch done", "id", req.Msg.Id)

	// elect logger
	logger := h.workers.Oldest(func(w *worker.Worker) bool {
		return w.Participates()
	})

	// let all participating workers optimise
	for w := range h.workers.Participating() {
		w.SendOptimStep(w.ID == logger.ID, h.epoch.Rewards())
	}

	// advance epoch
	h.epoch.Next(h.population)

	// broadcast new evaluations
	for w := range h.workers.Iter(nil) {
		if next := h.epoch.Assign(w); next != nil {
			slog.Debug("sending worker next batch", "id", req.Msg.Id, "slice", next)
			w.SendEvaluate(*next)
		}
	}

	slog.Debug("handled done", "id", req.Msg.Id)
	return connect.NewResponse(new(distributed.DoneResponse)), nil
}
