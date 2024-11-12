package main

import (
	"log/slog"
	"net/http"
	"os"
	"strconv"

	"github.com/neuro-soup/es-torch/server/pkg/proto/distributed/distributedconnect"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"
)

const (
	// serverPort is the serverPort the server listens on.
	serverPort = 8080
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	}))
	slog.SetDefault(log)

	mux := http.NewServeMux()
	mux.Handle(distributedconnect.NewESServiceHandler(newServer()))

	slog.Info("starting server...", "port", serverPort)
	err := http.ListenAndServe(":"+strconv.Itoa(serverPort), h2c.NewHandler(mux, new(http2.Server)))
	if err != nil {
		slog.Error("failed to start server", "err", err)
	}
}
