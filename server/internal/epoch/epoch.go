package epoch

import (
	"fmt"
	"log/slog"
	"sync"

	"github.com/neuro-soup/es-torch/server/internal/worker"
)

type Epoch struct {
	mu sync.RWMutex

	// id is the current epoch id.
	id uint32

	// population is the number of populations in the current epoch.
	population uint32

	// unassigned is the list of unassigned slices.
	unassigned []worker.Slice

	// rewards is the list of rewards for each slice.
	rewards [][]byte
}

// New returns a new epoch
func New(population uint32) *Epoch {
	e := &Epoch{population: population}
	e.Next(1)
	return e
}

// Next goes to the next epoch.
func (e *Epoch) Next(population uint32) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.id++
	slog.Debug("resetting epoch", "id", e.id)

	e.population = population

	e.unassigned = []worker.Slice{
		{
			Start: 0,
			End:   uint32(population),
		},
	}
	e.rewards = make([][]byte, population)
}

func (e *Epoch) Assign(w *worker.Worker) *worker.Slice {
	e.mu.Lock()
	defer e.mu.Unlock()

	slog.Debug("assigning slice to worker...", "worker_id", w.ID)

	if len(e.unassigned) == 0 {
		// no slices left
		slog.Debug("no slices left for worker", "worker", w.ID)
		return nil
	}

	next := e.unassigned[0]
	diff := next.Width()

	if diff <= uint32(w.Config.NumCPUs) {
		// claim existing entire slice fits on worker's CPUs
		e.unassigned = e.unassigned[1:]
		return &next
	}

	origEnd := next.End
	next.End = next.Start + uint32(w.Config.NumCPUs)
	split := worker.Slice{
		Start: next.Start + uint32(w.Config.NumCPUs),
		End:   origEnd,
	}

	e.unassigned = append(e.unassigned[1:], split)
	return &next
}

func (e *Epoch) Unassign(sl worker.Slice) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.unassigned = append(e.unassigned, sl)
}

// Reward rewards a slice with a list of rewards.
func (e *Epoch) Reward(sl worker.Slice, rewards [][]byte) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	slog.Debug("rewarding slice", "slice", sl)

	if sl.Width() != uint32(len(rewards)) {
		return fmt.Errorf("epoch: reward: slice width mismatch: expected %d, got %d",
			sl.Width(), len(rewards),
		)
	}

	for i := sl.Start; i < sl.End; i++ {
		e.rewards[i] = rewards[i-sl.Start]
	}
	return nil
}

func (e *Epoch) Done() bool {
	e.mu.RLock()
	defer e.mu.RUnlock()

	for _, r := range e.rewards {
		if r == nil {
			return false
		}
	}
	return true
}

func (e *Epoch) Rewards() [][]byte {
	e.mu.RLock()
	defer e.mu.RUnlock()

	rewards := make([][]byte, len(e.rewards))
	copy(rewards, e.rewards)
	return rewards
}
