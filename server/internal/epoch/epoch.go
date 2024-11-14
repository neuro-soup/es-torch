package epoch

import (
	"fmt"
	"log/slog"
	"sync"

	"github.com/neuro-soup/es-torch/server/internal/worker"
)

type Epoch struct {
	// id is the current epoch id.
	id uint32

	// population is the number of populations in the current epoch.
	population uint32

	// unassigned is the list of unassigned slices.
	unassigned   []worker.Slice
	unassignedMu sync.RWMutex

	// rewards is the list of rewards for each slice.
	rewards   [][]byte
	rewardsMu sync.RWMutex
}

// New returns a new epoch
func New(population uint32) *Epoch {
	e := &Epoch{population: population}
	e.Next(1)
	return e
}

// Next goes to the next epoch.
func (e *Epoch) Next(population uint32) {
	e.id++
	slog.Debug("resetting epoch", "id", e.id)

	e.population = population

	e.unassignedMu.Lock()
	e.unassigned = []worker.Slice{
		{
			Start: 0,
			End:   uint32(population),
		},
	}
	e.unassignedMu.Unlock()

	e.rewardsMu.Lock()
	e.rewards = make([][]byte, population)
	e.rewardsMu.Unlock()
}

func (e *Epoch) Assign(w *worker.Worker) *worker.Slice {
	e.unassignedMu.Lock()
	defer e.unassignedMu.Unlock()

	slog.Debug("assigning slice to worker...", "worker_id", w.ID)

	for i, sl := range e.unassigned {
		diff := sl.Width()
		if diff <= uint32(w.Config.NumCPUs) {
			// claim existing entire slice fits on worker's CPUs
			return &sl
		}

		sub := worker.Slice{
			Start: sl.Start,
			End:   sl.Start + uint32(w.Config.NumCPUs),
		}

		sl.Start = sub.End
		if sl.Start != sl.End {
			e.unassigned[i] = sl
		} else {
			e.unassigned = append([]worker.Slice{sub}, e.unassigned[i+1:]...)
		}

		return &sub
	}

	return nil
}

// Reward rewards a slice with a list of rewards.
func (e *Epoch) Reward(sl worker.Slice, rewards [][]byte) error {
	slog.Debug("rewarding slice", "slice", sl)

	if sl.Width() != uint32(len(rewards)) {
		return fmt.Errorf("epoch: reward: slice width mismatch: expected %d, got %d",
			sl.Width(), len(rewards),
		)
	}

	e.rewardsMu.Lock()
	for i := sl.Start; i < sl.End; i++ {
		e.rewards[i] = rewards[i-sl.Start]
	}
	e.rewardsMu.Unlock()

	return nil
}

func (e *Epoch) Done() bool {
	e.rewardsMu.RLock()
	defer e.rewardsMu.RUnlock()

	for _, r := range e.rewards {
		if r == nil {
			return false
		}
	}
	return true
}

func (e *Epoch) Rewards() [][]byte {
	e.rewardsMu.RLock()
	defer e.rewardsMu.RUnlock()

	rewards := make([][]byte, len(e.rewards))
	copy(rewards, e.rewards)
	return rewards
}
