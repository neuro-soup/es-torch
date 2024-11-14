package main

import (
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strconv"

	"github.com/neuro-soup/es-torch/server/internal/server"
	"github.com/neuro-soup/es-torch/server/internal/worker"
	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed/distributedconnect"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: logLevel(),
	}))
	slog.SetDefault(log)

	workers := worker.NewPool()
	handler := server.New(workers)

	mux := http.NewServeMux()
	mux.Handle(distributedconnect.NewESServiceHandler(handler))
	mux.Handle("/metrics", promhttp.Handler())

	p := port()
	slog.Info("starting server...", "port", p)

	err := http.ListenAndServe(
		fmt.Sprintf(":%d", p),
		h2c.NewHandler(mux, new(http2.Server)),
	)
	if err != nil {
		slog.Error("failed to start server", "err", err)
	}
}

func logLevel() slog.Level {
	level := os.Getenv("SERVER_LOG_LEVEL")
	if level == "" {
		return slog.LevelInfo
	}
	switch level {
	case "debug":
		return slog.LevelDebug
	case "info":
		return slog.LevelInfo
	case "warn":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

// port returns the port the server listens on.
func port() uint {
	port := os.Getenv("SERVER_PORT")
	if port == "" {
		return 8080
	}
	p, err := strconv.Atoi(port)
	if err != nil {
		slog.Error("failed to parse port", "err", err)
		return 8080
	}
	return uint(p)
}
