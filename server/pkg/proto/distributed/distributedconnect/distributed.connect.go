// Code generated by protoc-gen-connect-go. DO NOT EDIT.
//
// Source: distributed/distributed.proto

package distributedconnect

import (
	context "context"
	errors "errors"
	connect_go "github.com/bufbuild/connect-go"
	distributed "github.com/neuro-soup/es-torch/server/pkg/proto/distributed"
	http "net/http"
	strings "strings"
)

// This is a compile-time assertion to ensure that this generated file and the connect package are
// compatible. If you get a compiler error that this constant is not defined, this code was
// generated with a version of connect newer than the one compiled into your binary. You can fix the
// problem by either regenerating this code with an older version of connect or updating the connect
// version compiled into your binary.
const _ = connect_go.IsAtLeastVersion0_1_0

const (
	// ESServiceName is the fully-qualified name of the ESService service.
	ESServiceName = "distributed.ESService"
)

// These constants are the fully-qualified names of the RPCs defined in this package. They're
// exposed at runtime as Spec.Procedure and as the final two segments of the HTTP route.
//
// Note that these are different from the fully-qualified method names used by
// google.golang.org/protobuf/reflect/protoreflect. To convert from these constants to
// reflection-formatted method names, remove the leading slash and convert the remaining slash to a
// period.
const (
	// ESServiceDoneProcedure is the fully-qualified name of the ESService's Done RPC.
	ESServiceDoneProcedure = "/distributed.ESService/Done"
	// ESServiceHeartbeatProcedure is the fully-qualified name of the ESService's Heartbeat RPC.
	ESServiceHeartbeatProcedure = "/distributed.ESService/Heartbeat"
	// ESServiceSendStateProcedure is the fully-qualified name of the ESService's SendState RPC.
	ESServiceSendStateProcedure = "/distributed.ESService/SendState"
	// ESServiceSubscribeProcedure is the fully-qualified name of the ESService's Subscribe RPC.
	ESServiceSubscribeProcedure = "/distributed.ESService/Subscribe"
)

// ESServiceClient is a client for the distributed.ESService service.
type ESServiceClient interface {
	// epoch is done
	Done(context.Context, *connect_go.Request[distributed.DoneRequest]) (*connect_go.Response[distributed.DoneResponse], error)
	// worker heartbeat
	Heartbeat(context.Context, *connect_go.Request[distributed.HeartbeatRequest]) (*connect_go.Response[distributed.HeartbeatResponse], error)
	// send state if server demands it
	SendState(context.Context, *connect_go.Request[distributed.SendStateRequest]) (*connect_go.Response[distributed.SendStateResponse], error)
	// subscribe to server events
	Subscribe(context.Context, *connect_go.Request[distributed.SubscribeRequest]) (*connect_go.ServerStreamForClient[distributed.SubscribeResponse], error)
}

// NewESServiceClient constructs a client for the distributed.ESService service. By default, it uses
// the Connect protocol with the binary Protobuf Codec, asks for gzipped responses, and sends
// uncompressed requests. To use the gRPC or gRPC-Web protocols, supply the connect.WithGRPC() or
// connect.WithGRPCWeb() options.
//
// The URL supplied here should be the base URL for the Connect or gRPC server (for example,
// http://api.acme.com or https://acme.com/grpc).
func NewESServiceClient(httpClient connect_go.HTTPClient, baseURL string, opts ...connect_go.ClientOption) ESServiceClient {
	baseURL = strings.TrimRight(baseURL, "/")
	return &eSServiceClient{
		done: connect_go.NewClient[distributed.DoneRequest, distributed.DoneResponse](
			httpClient,
			baseURL+ESServiceDoneProcedure,
			opts...,
		),
		heartbeat: connect_go.NewClient[distributed.HeartbeatRequest, distributed.HeartbeatResponse](
			httpClient,
			baseURL+ESServiceHeartbeatProcedure,
			opts...,
		),
		sendState: connect_go.NewClient[distributed.SendStateRequest, distributed.SendStateResponse](
			httpClient,
			baseURL+ESServiceSendStateProcedure,
			opts...,
		),
		subscribe: connect_go.NewClient[distributed.SubscribeRequest, distributed.SubscribeResponse](
			httpClient,
			baseURL+ESServiceSubscribeProcedure,
			opts...,
		),
	}
}

// eSServiceClient implements ESServiceClient.
type eSServiceClient struct {
	done      *connect_go.Client[distributed.DoneRequest, distributed.DoneResponse]
	heartbeat *connect_go.Client[distributed.HeartbeatRequest, distributed.HeartbeatResponse]
	sendState *connect_go.Client[distributed.SendStateRequest, distributed.SendStateResponse]
	subscribe *connect_go.Client[distributed.SubscribeRequest, distributed.SubscribeResponse]
}

// Done calls distributed.ESService.Done.
func (c *eSServiceClient) Done(ctx context.Context, req *connect_go.Request[distributed.DoneRequest]) (*connect_go.Response[distributed.DoneResponse], error) {
	return c.done.CallUnary(ctx, req)
}

// Heartbeat calls distributed.ESService.Heartbeat.
func (c *eSServiceClient) Heartbeat(ctx context.Context, req *connect_go.Request[distributed.HeartbeatRequest]) (*connect_go.Response[distributed.HeartbeatResponse], error) {
	return c.heartbeat.CallUnary(ctx, req)
}

// SendState calls distributed.ESService.SendState.
func (c *eSServiceClient) SendState(ctx context.Context, req *connect_go.Request[distributed.SendStateRequest]) (*connect_go.Response[distributed.SendStateResponse], error) {
	return c.sendState.CallUnary(ctx, req)
}

// Subscribe calls distributed.ESService.Subscribe.
func (c *eSServiceClient) Subscribe(ctx context.Context, req *connect_go.Request[distributed.SubscribeRequest]) (*connect_go.ServerStreamForClient[distributed.SubscribeResponse], error) {
	return c.subscribe.CallServerStream(ctx, req)
}

// ESServiceHandler is an implementation of the distributed.ESService service.
type ESServiceHandler interface {
	// epoch is done
	Done(context.Context, *connect_go.Request[distributed.DoneRequest]) (*connect_go.Response[distributed.DoneResponse], error)
	// worker heartbeat
	Heartbeat(context.Context, *connect_go.Request[distributed.HeartbeatRequest]) (*connect_go.Response[distributed.HeartbeatResponse], error)
	// send state if server demands it
	SendState(context.Context, *connect_go.Request[distributed.SendStateRequest]) (*connect_go.Response[distributed.SendStateResponse], error)
	// subscribe to server events
	Subscribe(context.Context, *connect_go.Request[distributed.SubscribeRequest], *connect_go.ServerStream[distributed.SubscribeResponse]) error
}

// NewESServiceHandler builds an HTTP handler from the service implementation. It returns the path
// on which to mount the handler and the handler itself.
//
// By default, handlers support the Connect, gRPC, and gRPC-Web protocols with the binary Protobuf
// and JSON codecs. They also support gzip compression.
func NewESServiceHandler(svc ESServiceHandler, opts ...connect_go.HandlerOption) (string, http.Handler) {
	mux := http.NewServeMux()
	mux.Handle(ESServiceDoneProcedure, connect_go.NewUnaryHandler(
		ESServiceDoneProcedure,
		svc.Done,
		opts...,
	))
	mux.Handle(ESServiceHeartbeatProcedure, connect_go.NewUnaryHandler(
		ESServiceHeartbeatProcedure,
		svc.Heartbeat,
		opts...,
	))
	mux.Handle(ESServiceSendStateProcedure, connect_go.NewUnaryHandler(
		ESServiceSendStateProcedure,
		svc.SendState,
		opts...,
	))
	mux.Handle(ESServiceSubscribeProcedure, connect_go.NewServerStreamHandler(
		ESServiceSubscribeProcedure,
		svc.Subscribe,
		opts...,
	))
	return "/distributed.ESService/", mux
}

// UnimplementedESServiceHandler returns CodeUnimplemented from all methods.
type UnimplementedESServiceHandler struct{}

func (UnimplementedESServiceHandler) Done(context.Context, *connect_go.Request[distributed.DoneRequest]) (*connect_go.Response[distributed.DoneResponse], error) {
	return nil, connect_go.NewError(connect_go.CodeUnimplemented, errors.New("distributed.ESService.Done is not implemented"))
}

func (UnimplementedESServiceHandler) Heartbeat(context.Context, *connect_go.Request[distributed.HeartbeatRequest]) (*connect_go.Response[distributed.HeartbeatResponse], error) {
	return nil, connect_go.NewError(connect_go.CodeUnimplemented, errors.New("distributed.ESService.Heartbeat is not implemented"))
}

func (UnimplementedESServiceHandler) SendState(context.Context, *connect_go.Request[distributed.SendStateRequest]) (*connect_go.Response[distributed.SendStateResponse], error) {
	return nil, connect_go.NewError(connect_go.CodeUnimplemented, errors.New("distributed.ESService.SendState is not implemented"))
}

func (UnimplementedESServiceHandler) Subscribe(context.Context, *connect_go.Request[distributed.SubscribeRequest], *connect_go.ServerStream[distributed.SubscribeResponse]) error {
	return connect_go.NewError(connect_go.CodeUnimplemented, errors.New("distributed.ESService.Subscribe is not implemented"))
}
