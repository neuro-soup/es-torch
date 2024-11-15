package worker

import (
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed"
)

// heartbeatTimeout is the timeout for heartbeats.
const heartbeatTimeout = 20 * time.Second

// eventBufferSize is the size of the event buffer.
const eventBufferSize = 15

// Status is the status of the worker.
type Status string

const (
	// StatusAwaitsHello is the status of the worker when it is awaiting the
	// hello message.
	StatusAwaitsHello = "awaits-hello"

	// StatusEvaluating is the status of the worker when it is evaluating a slice.
	StatusEvaluating = "evaluating"

	// StatusOptimizing is the status of the worker when it is making an optimisation
	// step.
	StatusOptimizing = "optimizing"

	// StatusDisconnecting is the status of the worker when it is disconnecting.
	StatusDisconnecting = "disconnecting"

	// StatusIdling is the status of the worker when it is idling.
	StatusIdling = "idling"
)

// Config is the client-side configuration of the worker.
type Config struct {
	// NumCPUs is the number of CPUs available to the worker.
	NumCPUs uint8

	// Device is used to specify the acceleration device to be used by the worker.
	Device string
}

type Worker struct {
	mu sync.RWMutex

	// ID is the unique identifier of the worker.
	ID uint8

	// Config is the client-side configuration of the worker.
	Config Config

	// JoinedAt is the time when the worker joined the cluster.
	JoinedAt time.Time

	// Status is the status of the worker.
	Status Status

	// LastHeartbeat is the time when the worker last sent a heartbeat.
	LastHeartbeat time.Time

	// Ping is the last ping duration.
	Ping time.Duration

	// events is the channel used to send events to the worker.
	events chan *distributed.SubscribeResponse

	// disconnects are the channels used to send a signal to the worker to
	// disconnect. The channel is closed when the worker disconnects.
	disconnects []chan struct{}

	// evaluating is the slice that the worker is currently evaluating. If nil,
	// the worker is not evaluating a slice.
	evaluating *Slice

	// stateRequest is the channel used to request the worker's state.
	stateRequests []chan []byte

	// destroyed is true if the worker has been destroyed.
	destroyed bool
}

// new creates a new worker.
func newWorker(id uint8, cfg Config) *Worker {
	return &Worker{
		ID:       id,
		Config:   cfg,
		JoinedAt: time.Now(),
		Status:   StatusAwaitsHello,
		events:   make(chan *distributed.SubscribeResponse, eventBufferSize),
	}
}

func (w *Worker) String() string {
	return fmt.Sprintf("Worker(ID=%d, Status=%s)", w.ID, w.Status)
}

// Participates returns true if the worker is participating in the experiment.
func (w *Worker) Participates() bool {
	return w.Status != StatusAwaitsHello && w.Status != StatusDisconnecting
}

// Disconnect sends a signal to the worker to disconnect.
func (w *Worker) Disconnect() {
	slog.Debug("disconnecting worker...",
		"id", w.ID,
		"status", w.Status,
	)

	w.Status = StatusDisconnecting
	for _, ch := range w.disconnects {
		ch <- struct{}{}
	}
}

// Disconnects returns the channel used to send a signal to the worker to
// disconnect.
func (w *Worker) Disconnects() <-chan struct{} {
	w.mu.RLock()
	defer w.mu.RUnlock()

	ch := make(chan struct{}, 1)
	w.disconnects = append(w.disconnects, ch)
	return ch
}

// Destroy cleans up the worker.
func (w *Worker) Destroy() {
	slog.Debug("cleaning up worker...",
		"id", w.ID,
		"status", w.Status,
	)

	w.destroyed = true
	close(w.events)

	w.mu.Lock()
	for _, ch := range w.disconnects {
		close(ch)
	}
	w.disconnects = nil
	w.mu.Unlock()
}

// ReceiveHeartbeat marks the worker as alive.
func (w *Worker) ReceiveHeartbeat(sentAt time.Time) error {
	ping := time.Since(sentAt)

	slog.Debug("heartbeating worker...",
		"id", w.ID,
		"status", w.Status,
		"sent_at", sentAt,
		"ping", ping.String(),
	)

	w.LastHeartbeat = time.Now() // TODO: use sentAt?
	w.Ping = ping

	if w.Ping > heartbeatTimeout {
		slog.Error("worker ping timeout", "worker_id", w.ID)
		return fmt.Errorf("heartbeat was transmitted more than %s", heartbeatTimeout)
	}

	return nil
}

func (w *Worker) States() <-chan []byte {
	w.mu.Lock()
	defer w.mu.Unlock()

	ch := make(chan []byte, 1)
	w.stateRequests = append(w.stateRequests, ch)
	return ch
}

// ReceiveState needs to be called when the worker sends its updated state
// to the server.
func (w *Worker) ReceiveState(state []byte) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	var wg sync.WaitGroup
	wg.Add(len(w.stateRequests))
	for _, ch := range w.stateRequests {
		go func(state []byte) {
			ch <- state
			wg.Done()
		}(state)
	}
	wg.Wait()
}

// ReceiveDone is called when the worker has finished evaluating the given slice.
func (w *Worker) ReceiveDone(sl Slice) {
	w.mu.Lock()
	defer w.mu.Unlock()

	slog.Debug("worker finished evaluating slice",
		"id", w.ID,
		"slice", sl,
	)
	w.evaluating = nil
	w.Status = StatusIdling
}

// Events returns the channel used to send events to the worker.
func (w *Worker) Events() <-chan *distributed.SubscribeResponse {
	return w.events
}

// Evaluate requests the worker to evaluate the given slice.
func (w *Worker) SendEvaluate(slice Slice) {
	w.mu.Lock()
	defer w.mu.Unlock()

	slog.Debug("evaluating slice...",
		"id", w.ID,
		"slice", slice,
	)

	w.Status = StatusEvaluating
	w.evaluating = &slice

	w.sendEvent(&distributed.SubscribeResponse{
		Type: distributed.ServerEventType_EVALUATE_BATCH,
		Event: &distributed.SubscribeResponse_EvaluateBatch{
			EvaluateBatch: &distributed.EvaluateBatchEvent{
				PopSlice: &distributed.Slice{
					Start: int32(slice.Start),
					End:   int32(slice.End),
				},
			},
		},
	})
}

// Evaluating returns the slice that the worker is currently evaluating. If nil,
// the worker is not evaluating a slice.
func (w *Worker) Evaluating() *Slice {
	return w.evaluating
}

// SendStateRequest requests the worker to send its state for the given device.
func (w *Worker) SendStateRequest(forDevice string) <-chan []byte {
	w.mu.Lock()
	defer w.mu.Unlock()

	ch := make(chan []byte, 1)
	w.stateRequests = append(w.stateRequests, ch)

	if len(w.stateRequests) > 1 {
		slog.Debug("worker already has a state request in progress...",
			"id", w.ID,
			"device", forDevice,
		)
	} else {
		slog.Debug("requesting worker state...",
			"id", w.ID,
			"device", forDevice,
		)
		w.sendEvent(&distributed.SubscribeResponse{
			Type: distributed.ServerEventType_SEND_STATE,
			Event: &distributed.SubscribeResponse_SendState{
				SendState: &distributed.SendStateEvent{
					Device: forDevice,
				},
			},
		})
	}

	return ch
}

// SendHello sends the worker's hello message.
func (w *Worker) SendHello(state []byte) {
	slog.Debug("sending hello to worker...",
		"id", w.ID,
		"state", len(state),
	)

	w.sendEvent(&distributed.SubscribeResponse{
		Type: distributed.ServerEventType_HELLO,
		Event: &distributed.SubscribeResponse_Hello{
			Hello: &distributed.HelloEvent{
				Id:        int32(w.ID),
				InitState: state,
			},
		},
	})
}

// OptimStep requests the worker to make an optimisation step using the given
// rewards.
func (w *Worker) SendOptimStep(logging bool, rewards [][]byte) {
	slog.Debug("making an optimisation step...",
		"id", w.ID,
	)

	w.Status = StatusOptimizing

	w.sendEvent(&distributed.SubscribeResponse{
		Type: distributed.ServerEventType_OPTIM_STEP,
		Event: &distributed.SubscribeResponse_OptimStep{
			OptimStep: &distributed.OptimStepEvent{
				Logging: logging,
				Rewards: rewards,
			},
		},
	})
}

// Send sends an event to the worker.
func (w *Worker) sendEvent(msg *distributed.SubscribeResponse) {
	if w.destroyed {
		return
	}
	slog.Debug("sending event to worker...",
		"id", w.ID,
		"event", msg.Type.String(),
	)
	w.events <- msg
}
