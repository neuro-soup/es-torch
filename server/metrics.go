package main

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var eventsSent = promauto.NewCounter(prometheus.CounterOpts{
	Name: "events_sent",
	Help: "Total number of events sent",
})
