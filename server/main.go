package main

import (
	"log/slog"
	"net/http"
	"strconv"

	"github.com/neuro-soup/es-torch/server/pkg/proto/es/esconnect"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"
)

const port = 8080

func main() {
	mux := http.NewServeMux()
	mux.Handle(esconnect.NewESServiceHandler(newServer()))

	slog.Info("starting server...", "port", port)
	err := http.ListenAndServe(":"+strconv.Itoa(port), h2c.NewHandler(mux, new(http2.Server)))
	if err != nil {
		slog.Error("failed to start server", "err", err)
	}
}
