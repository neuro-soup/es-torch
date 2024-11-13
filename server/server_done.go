package main

import (
	"context"
	"errors"
	"log/slog"

	"github.com/bufbuild/connect-go"
	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed"
)

func (s *server) Done(
	ctx context.Context,
	req *connect.Request[distributed.DoneRequest],
) (*connect.Response[distributed.DoneResponse], error) {
	slog.Debug("received done request", "worker_id", req.Msg.Id)

	w := s.workers.get(uint8(req.Msg.Id))
	if w == nil {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("invalid worker id"))
	}

	prev := s.slices.find(uint32(req.Msg.Slice.Start), uint32(req.Msg.Slice.End))
	if prev == nil {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("invalid slice"))
	}
	prev.rewards = req.Msg.BatchRewards

	if next := s.slices.assign(uint8(req.Msg.Id), w); next != nil {
		slog.Debug("sending worker next batch", "worker_id", req.Msg.Id, "slice", next)
		w.events <- &distributed.SubscribeResponse{
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
		return connect.NewResponse(&distributed.DoneResponse{}), nil
	}

	if s.slices.isEpochDone() {
        rewards := s.slices.rewards()
        loggingWorkerID := s.workers.oldestWorker()
		for id, w := range s.workers.iter() {
			w.events <- &distributed.SubscribeResponse{
				Type: distributed.ServerEventType_OPTIM_STEP,
				Event: &distributed.SubscribeResponse_OptimStep{
					OptimStep: &distributed.OptimStepEvent{
						Logging: id == loggingWorkerID,
						Rewards: rewards,
					},
				},
			}
		}
		s.params.RLock()
		s.slices.reset(s.params.numPop)
		s.params.RUnlock()
		for id, w := range s.workers.iter() {
			sl := s.slices.assign(id, w)
			if sl == nil {
				continue
			}
			w.events <- &distributed.SubscribeResponse{
				Type: distributed.ServerEventType_EVALUATE_BATCH,
				Event: &distributed.SubscribeResponse_EvaluateBatch{
					EvaluateBatch: &distributed.EvaluateBatchEvent{
						PopSlice: &distributed.Slice{
							Start: int32(sl.start),
							End:   int32(sl.end),
						},
					},
				},
			}
		}

		return connect.NewResponse(&distributed.DoneResponse{}), nil
	}

	slog.Debug("received done request, waiting for other workers to finish", "worker_id", req.Msg.Id, "slice", prev)
	return connect.NewResponse(&distributed.DoneResponse{}), nil
}
