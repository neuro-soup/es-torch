package main

import (
	"log/slog"
	"net/http"
	"strconv"

	"github.com/neuro-soup/es-torch/server/pkg/proto/es/esconnect"
)

const port = 8080

func main() {
	mux := http.NewServeMux()
	mux.Handle(esconnect.NewESServiceHandler(newServer()))

	slog.Info("starting server...", "port", port)
	err := http.ListenAndServe(":"+strconv.Itoa(port), mux)
	if err != nil {
		slog.Error("failed to start server", "err", err)
	}
}
