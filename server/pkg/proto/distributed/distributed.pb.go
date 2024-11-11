// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.1
// 	protoc        (unknown)
// source: distributed/distributed.proto

package distributed

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	timestamppb "google.golang.org/protobuf/types/known/timestamppb"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type ServerEventType int32

const (
	ServerEventType_UNKNOWN        ServerEventType = 0
	ServerEventType_SEND_STATE     ServerEventType = 1
	ServerEventType_EVALUATE_BATCH ServerEventType = 2
	ServerEventType_STATE_UPDATE   ServerEventType = 3
	ServerEventType_OPTIM_STEP     ServerEventType = 4
	ServerEventType_HELLO          ServerEventType = 5
)

// Enum value maps for ServerEventType.
var (
	ServerEventType_name = map[int32]string{
		0: "UNKNOWN",
		1: "SEND_STATE",
		2: "EVALUATE_BATCH",
		3: "STATE_UPDATE",
		4: "OPTIM_STEP",
		5: "HELLO",
	}
	ServerEventType_value = map[string]int32{
		"UNKNOWN":        0,
		"SEND_STATE":     1,
		"EVALUATE_BATCH": 2,
		"STATE_UPDATE":   3,
		"OPTIM_STEP":     4,
		"HELLO":          5,
	}
)

func (x ServerEventType) Enum() *ServerEventType {
	p := new(ServerEventType)
	*p = x
	return p
}

func (x ServerEventType) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ServerEventType) Descriptor() protoreflect.EnumDescriptor {
	return file_distributed_distributed_proto_enumTypes[0].Descriptor()
}

func (ServerEventType) Type() protoreflect.EnumType {
	return &file_distributed_distributed_proto_enumTypes[0]
}

func (x ServerEventType) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ServerEventType.Descriptor instead.
func (ServerEventType) EnumDescriptor() ([]byte, []int) {
	return file_distributed_distributed_proto_rawDescGZIP(), []int{0}
}

type Slice struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Start int32 `protobuf:"varint,1,opt,name=start,proto3" json:"start,omitempty"`
	End   int32 `protobuf:"varint,2,opt,name=end,proto3" json:"end,omitempty"`
}

func (x *Slice) Reset() {
	*x = Slice{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_distributed_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Slice) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Slice) ProtoMessage() {}

func (x *Slice) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_distributed_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Slice.ProtoReflect.Descriptor instead.
func (*Slice) Descriptor() ([]byte, []int) {
	return file_distributed_distributed_proto_rawDescGZIP(), []int{0}
}

func (x *Slice) GetStart() int32 {
	if x != nil {
		return x.Start
	}
	return 0
}

func (x *Slice) GetEnd() int32 {
	if x != nil {
		return x.End
	}
	return 0
}

type HeartbeatRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Id        int32                  `protobuf:"varint,1,opt,name=id,proto3" json:"id,omitempty"`
	Timestamp *timestamppb.Timestamp `protobuf:"bytes,2,opt,name=timestamp,proto3" json:"timestamp,omitempty"`
}

func (x *HeartbeatRequest) Reset() {
	*x = HeartbeatRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_distributed_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *HeartbeatRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*HeartbeatRequest) ProtoMessage() {}

func (x *HeartbeatRequest) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_distributed_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use HeartbeatRequest.ProtoReflect.Descriptor instead.
func (*HeartbeatRequest) Descriptor() ([]byte, []int) {
	return file_distributed_distributed_proto_rawDescGZIP(), []int{1}
}

func (x *HeartbeatRequest) GetId() int32 {
	if x != nil {
		return x.Id
	}
	return 0
}

func (x *HeartbeatRequest) GetTimestamp() *timestamppb.Timestamp {
	if x != nil {
		return x.Timestamp
	}
	return nil
}

type HeartbeatResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Ok bool `protobuf:"varint,1,opt,name=ok,proto3" json:"ok,omitempty"`
}

func (x *HeartbeatResponse) Reset() {
	*x = HeartbeatResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_distributed_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *HeartbeatResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*HeartbeatResponse) ProtoMessage() {}

func (x *HeartbeatResponse) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_distributed_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use HeartbeatResponse.ProtoReflect.Descriptor instead.
func (*HeartbeatResponse) Descriptor() ([]byte, []int) {
	return file_distributed_distributed_proto_rawDescGZIP(), []int{2}
}

func (x *HeartbeatResponse) GetOk() bool {
	if x != nil {
		return x.Ok
	}
	return false
}

type DoneRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Id           int32    `protobuf:"varint,1,opt,name=id,proto3" json:"id,omitempty"`
	Slice        *Slice   `protobuf:"bytes,2,opt,name=slice,proto3" json:"slice,omitempty"`
	BatchRewards [][]byte `protobuf:"bytes,3,rep,name=batch_rewards,json=batchRewards,proto3" json:"batch_rewards,omitempty"`
}

func (x *DoneRequest) Reset() {
	*x = DoneRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_distributed_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *DoneRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DoneRequest) ProtoMessage() {}

func (x *DoneRequest) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_distributed_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DoneRequest.ProtoReflect.Descriptor instead.
func (*DoneRequest) Descriptor() ([]byte, []int) {
	return file_distributed_distributed_proto_rawDescGZIP(), []int{3}
}

func (x *DoneRequest) GetId() int32 {
	if x != nil {
		return x.Id
	}
	return 0
}

func (x *DoneRequest) GetSlice() *Slice {
	if x != nil {
		return x.Slice
	}
	return nil
}

func (x *DoneRequest) GetBatchRewards() [][]byte {
	if x != nil {
		return x.BatchRewards
	}
	return nil
}

type DoneResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *DoneResponse) Reset() {
	*x = DoneResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_distributed_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *DoneResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DoneResponse) ProtoMessage() {}

func (x *DoneResponse) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_distributed_proto_msgTypes[4]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DoneResponse.ProtoReflect.Descriptor instead.
func (*DoneResponse) Descriptor() ([]byte, []int) {
	return file_distributed_distributed_proto_rawDescGZIP(), []int{4}
}

type SendStateRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Id    int32  `protobuf:"varint,1,opt,name=id,proto3" json:"id,omitempty"`
	State []byte `protobuf:"bytes,2,opt,name=state,proto3" json:"state,omitempty"`
}

func (x *SendStateRequest) Reset() {
	*x = SendStateRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_distributed_proto_msgTypes[5]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SendStateRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SendStateRequest) ProtoMessage() {}

func (x *SendStateRequest) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_distributed_proto_msgTypes[5]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SendStateRequest.ProtoReflect.Descriptor instead.
func (*SendStateRequest) Descriptor() ([]byte, []int) {
	return file_distributed_distributed_proto_rawDescGZIP(), []int{5}
}

func (x *SendStateRequest) GetId() int32 {
	if x != nil {
		return x.Id
	}
	return 0
}

func (x *SendStateRequest) GetState() []byte {
	if x != nil {
		return x.State
	}
	return nil
}

type SendStateResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *SendStateResponse) Reset() {
	*x = SendStateResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_distributed_proto_msgTypes[6]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SendStateResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SendStateResponse) ProtoMessage() {}

func (x *SendStateResponse) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_distributed_proto_msgTypes[6]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SendStateResponse.ProtoReflect.Descriptor instead.
func (*SendStateResponse) Descriptor() ([]byte, []int) {
	return file_distributed_distributed_proto_rawDescGZIP(), []int{6}
}

type SubscribeRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// number of CPUs available on the node
	NumCpus int32 `protobuf:"varint,1,opt,name=num_cpus,json=numCpus,proto3" json:"num_cpus,omitempty"`
	// number of population to be evaluated
	NumPop int32 `protobuf:"varint,2,opt,name=num_pop,json=numPop,proto3" json:"num_pop,omitempty"`
}

func (x *SubscribeRequest) Reset() {
	*x = SubscribeRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_distributed_proto_msgTypes[7]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SubscribeRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SubscribeRequest) ProtoMessage() {}

func (x *SubscribeRequest) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_distributed_proto_msgTypes[7]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SubscribeRequest.ProtoReflect.Descriptor instead.
func (*SubscribeRequest) Descriptor() ([]byte, []int) {
	return file_distributed_distributed_proto_rawDescGZIP(), []int{7}
}

func (x *SubscribeRequest) GetNumCpus() int32 {
	if x != nil {
		return x.NumCpus
	}
	return 0
}

func (x *SubscribeRequest) GetNumPop() int32 {
	if x != nil {
		return x.NumPop
	}
	return 0
}

type SendStateEvent struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *SendStateEvent) Reset() {
	*x = SendStateEvent{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_distributed_proto_msgTypes[8]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SendStateEvent) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SendStateEvent) ProtoMessage() {}

func (x *SendStateEvent) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_distributed_proto_msgTypes[8]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SendStateEvent.ProtoReflect.Descriptor instead.
func (*SendStateEvent) Descriptor() ([]byte, []int) {
	return file_distributed_distributed_proto_rawDescGZIP(), []int{8}
}

type EvaluateBatchEvent struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	PopSlice *Slice `protobuf:"bytes,1,opt,name=pop_slice,json=popSlice,proto3" json:"pop_slice,omitempty"`
}

func (x *EvaluateBatchEvent) Reset() {
	*x = EvaluateBatchEvent{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_distributed_proto_msgTypes[9]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *EvaluateBatchEvent) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*EvaluateBatchEvent) ProtoMessage() {}

func (x *EvaluateBatchEvent) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_distributed_proto_msgTypes[9]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use EvaluateBatchEvent.ProtoReflect.Descriptor instead.
func (*EvaluateBatchEvent) Descriptor() ([]byte, []int) {
	return file_distributed_distributed_proto_rawDescGZIP(), []int{9}
}

func (x *EvaluateBatchEvent) GetPopSlice() *Slice {
	if x != nil {
		return x.PopSlice
	}
	return nil
}

type OptimStepEvent struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Logging bool     `protobuf:"varint,1,opt,name=logging,proto3" json:"logging,omitempty"`
	Rewards [][]byte `protobuf:"bytes,2,rep,name=rewards,proto3" json:"rewards,omitempty"` // n_pop floats
}

func (x *OptimStepEvent) Reset() {
	*x = OptimStepEvent{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_distributed_proto_msgTypes[10]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *OptimStepEvent) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OptimStepEvent) ProtoMessage() {}

func (x *OptimStepEvent) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_distributed_proto_msgTypes[10]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use OptimStepEvent.ProtoReflect.Descriptor instead.
func (*OptimStepEvent) Descriptor() ([]byte, []int) {
	return file_distributed_distributed_proto_rawDescGZIP(), []int{10}
}

func (x *OptimStepEvent) GetLogging() bool {
	if x != nil {
		return x.Logging
	}
	return false
}

func (x *OptimStepEvent) GetRewards() [][]byte {
	if x != nil {
		return x.Rewards
	}
	return nil
}

type HelloEvent struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Id        int32  `protobuf:"varint,1,opt,name=id,proto3" json:"id,omitempty"`
	InitState []byte `protobuf:"bytes,2,opt,name=init_state,json=initState,proto3,oneof" json:"init_state,omitempty"`
}

func (x *HelloEvent) Reset() {
	*x = HelloEvent{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_distributed_proto_msgTypes[11]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *HelloEvent) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*HelloEvent) ProtoMessage() {}

func (x *HelloEvent) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_distributed_proto_msgTypes[11]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use HelloEvent.ProtoReflect.Descriptor instead.
func (*HelloEvent) Descriptor() ([]byte, []int) {
	return file_distributed_distributed_proto_rawDescGZIP(), []int{11}
}

func (x *HelloEvent) GetId() int32 {
	if x != nil {
		return x.Id
	}
	return 0
}

func (x *HelloEvent) GetInitState() []byte {
	if x != nil {
		return x.InitState
	}
	return nil
}

type SubscribeResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Type ServerEventType `protobuf:"varint,1,opt,name=type,proto3,enum=distributed.ServerEventType" json:"type,omitempty"`
	// Types that are assignable to Event:
	//
	//	*SubscribeResponse_SendState
	//	*SubscribeResponse_EvaluateBatch
	//	*SubscribeResponse_OptimStep
	//	*SubscribeResponse_Hello
	Event isSubscribeResponse_Event `protobuf_oneof:"event"`
}

func (x *SubscribeResponse) Reset() {
	*x = SubscribeResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_distributed_proto_msgTypes[12]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SubscribeResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SubscribeResponse) ProtoMessage() {}

func (x *SubscribeResponse) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_distributed_proto_msgTypes[12]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SubscribeResponse.ProtoReflect.Descriptor instead.
func (*SubscribeResponse) Descriptor() ([]byte, []int) {
	return file_distributed_distributed_proto_rawDescGZIP(), []int{12}
}

func (x *SubscribeResponse) GetType() ServerEventType {
	if x != nil {
		return x.Type
	}
	return ServerEventType_UNKNOWN
}

func (m *SubscribeResponse) GetEvent() isSubscribeResponse_Event {
	if m != nil {
		return m.Event
	}
	return nil
}

func (x *SubscribeResponse) GetSendState() *SendStateEvent {
	if x, ok := x.GetEvent().(*SubscribeResponse_SendState); ok {
		return x.SendState
	}
	return nil
}

func (x *SubscribeResponse) GetEvaluateBatch() *EvaluateBatchEvent {
	if x, ok := x.GetEvent().(*SubscribeResponse_EvaluateBatch); ok {
		return x.EvaluateBatch
	}
	return nil
}

func (x *SubscribeResponse) GetOptimStep() *OptimStepEvent {
	if x, ok := x.GetEvent().(*SubscribeResponse_OptimStep); ok {
		return x.OptimStep
	}
	return nil
}

func (x *SubscribeResponse) GetHello() *HelloEvent {
	if x, ok := x.GetEvent().(*SubscribeResponse_Hello); ok {
		return x.Hello
	}
	return nil
}

type isSubscribeResponse_Event interface {
	isSubscribeResponse_Event()
}

type SubscribeResponse_SendState struct {
	SendState *SendStateEvent `protobuf:"bytes,2,opt,name=send_state,json=sendState,proto3,oneof"`
}

type SubscribeResponse_EvaluateBatch struct {
	EvaluateBatch *EvaluateBatchEvent `protobuf:"bytes,3,opt,name=evaluate_batch,json=evaluateBatch,proto3,oneof"`
}

type SubscribeResponse_OptimStep struct {
	OptimStep *OptimStepEvent `protobuf:"bytes,4,opt,name=optim_step,json=optimStep,proto3,oneof"`
}

type SubscribeResponse_Hello struct {
	Hello *HelloEvent `protobuf:"bytes,5,opt,name=hello,proto3,oneof"`
}

func (*SubscribeResponse_SendState) isSubscribeResponse_Event() {}

func (*SubscribeResponse_EvaluateBatch) isSubscribeResponse_Event() {}

func (*SubscribeResponse_OptimStep) isSubscribeResponse_Event() {}

func (*SubscribeResponse_Hello) isSubscribeResponse_Event() {}

var File_distributed_distributed_proto protoreflect.FileDescriptor

var file_distributed_distributed_proto_rawDesc = []byte{
	0x0a, 0x1d, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x2f, 0x64, 0x69,
	0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12,
	0x0b, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x1a, 0x1f, 0x67, 0x6f,
	0x6f, 0x67, 0x6c, 0x65, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2f, 0x74, 0x69,
	0x6d, 0x65, 0x73, 0x74, 0x61, 0x6d, 0x70, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22, 0x2f, 0x0a,
	0x05, 0x53, 0x6c, 0x69, 0x63, 0x65, 0x12, 0x14, 0x0a, 0x05, 0x73, 0x74, 0x61, 0x72, 0x74, 0x18,
	0x01, 0x20, 0x01, 0x28, 0x05, 0x52, 0x05, 0x73, 0x74, 0x61, 0x72, 0x74, 0x12, 0x10, 0x0a, 0x03,
	0x65, 0x6e, 0x64, 0x18, 0x02, 0x20, 0x01, 0x28, 0x05, 0x52, 0x03, 0x65, 0x6e, 0x64, 0x22, 0x5c,
	0x0a, 0x10, 0x48, 0x65, 0x61, 0x72, 0x74, 0x62, 0x65, 0x61, 0x74, 0x52, 0x65, 0x71, 0x75, 0x65,
	0x73, 0x74, 0x12, 0x0e, 0x0a, 0x02, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x05, 0x52, 0x02,
	0x69, 0x64, 0x12, 0x38, 0x0a, 0x09, 0x74, 0x69, 0x6d, 0x65, 0x73, 0x74, 0x61, 0x6d, 0x70, 0x18,
	0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x1a, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e, 0x54, 0x69, 0x6d, 0x65, 0x73, 0x74, 0x61, 0x6d,
	0x70, 0x52, 0x09, 0x74, 0x69, 0x6d, 0x65, 0x73, 0x74, 0x61, 0x6d, 0x70, 0x22, 0x23, 0x0a, 0x11,
	0x48, 0x65, 0x61, 0x72, 0x74, 0x62, 0x65, 0x61, 0x74, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73,
	0x65, 0x12, 0x0e, 0x0a, 0x02, 0x6f, 0x6b, 0x18, 0x01, 0x20, 0x01, 0x28, 0x08, 0x52, 0x02, 0x6f,
	0x6b, 0x22, 0x6c, 0x0a, 0x0b, 0x44, 0x6f, 0x6e, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74,
	0x12, 0x0e, 0x0a, 0x02, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x05, 0x52, 0x02, 0x69, 0x64,
	0x12, 0x28, 0x0a, 0x05, 0x73, 0x6c, 0x69, 0x63, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32,
	0x12, 0x2e, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x2e, 0x53, 0x6c,
	0x69, 0x63, 0x65, 0x52, 0x05, 0x73, 0x6c, 0x69, 0x63, 0x65, 0x12, 0x23, 0x0a, 0x0d, 0x62, 0x61,
	0x74, 0x63, 0x68, 0x5f, 0x72, 0x65, 0x77, 0x61, 0x72, 0x64, 0x73, 0x18, 0x03, 0x20, 0x03, 0x28,
	0x0c, 0x52, 0x0c, 0x62, 0x61, 0x74, 0x63, 0x68, 0x52, 0x65, 0x77, 0x61, 0x72, 0x64, 0x73, 0x22,
	0x0e, 0x0a, 0x0c, 0x44, 0x6f, 0x6e, 0x65, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22,
	0x38, 0x0a, 0x10, 0x53, 0x65, 0x6e, 0x64, 0x53, 0x74, 0x61, 0x74, 0x65, 0x52, 0x65, 0x71, 0x75,
	0x65, 0x73, 0x74, 0x12, 0x0e, 0x0a, 0x02, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x05, 0x52,
	0x02, 0x69, 0x64, 0x12, 0x14, 0x0a, 0x05, 0x73, 0x74, 0x61, 0x74, 0x65, 0x18, 0x02, 0x20, 0x01,
	0x28, 0x0c, 0x52, 0x05, 0x73, 0x74, 0x61, 0x74, 0x65, 0x22, 0x13, 0x0a, 0x11, 0x53, 0x65, 0x6e,
	0x64, 0x53, 0x74, 0x61, 0x74, 0x65, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x46,
	0x0a, 0x10, 0x53, 0x75, 0x62, 0x73, 0x63, 0x72, 0x69, 0x62, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65,
	0x73, 0x74, 0x12, 0x19, 0x0a, 0x08, 0x6e, 0x75, 0x6d, 0x5f, 0x63, 0x70, 0x75, 0x73, 0x18, 0x01,
	0x20, 0x01, 0x28, 0x05, 0x52, 0x07, 0x6e, 0x75, 0x6d, 0x43, 0x70, 0x75, 0x73, 0x12, 0x17, 0x0a,
	0x07, 0x6e, 0x75, 0x6d, 0x5f, 0x70, 0x6f, 0x70, 0x18, 0x02, 0x20, 0x01, 0x28, 0x05, 0x52, 0x06,
	0x6e, 0x75, 0x6d, 0x50, 0x6f, 0x70, 0x22, 0x10, 0x0a, 0x0e, 0x53, 0x65, 0x6e, 0x64, 0x53, 0x74,
	0x61, 0x74, 0x65, 0x45, 0x76, 0x65, 0x6e, 0x74, 0x22, 0x45, 0x0a, 0x12, 0x45, 0x76, 0x61, 0x6c,
	0x75, 0x61, 0x74, 0x65, 0x42, 0x61, 0x74, 0x63, 0x68, 0x45, 0x76, 0x65, 0x6e, 0x74, 0x12, 0x2f,
	0x0a, 0x09, 0x70, 0x6f, 0x70, 0x5f, 0x73, 0x6c, 0x69, 0x63, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28,
	0x0b, 0x32, 0x12, 0x2e, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x2e,
	0x53, 0x6c, 0x69, 0x63, 0x65, 0x52, 0x08, 0x70, 0x6f, 0x70, 0x53, 0x6c, 0x69, 0x63, 0x65, 0x22,
	0x44, 0x0a, 0x0e, 0x4f, 0x70, 0x74, 0x69, 0x6d, 0x53, 0x74, 0x65, 0x70, 0x45, 0x76, 0x65, 0x6e,
	0x74, 0x12, 0x18, 0x0a, 0x07, 0x6c, 0x6f, 0x67, 0x67, 0x69, 0x6e, 0x67, 0x18, 0x01, 0x20, 0x01,
	0x28, 0x08, 0x52, 0x07, 0x6c, 0x6f, 0x67, 0x67, 0x69, 0x6e, 0x67, 0x12, 0x18, 0x0a, 0x07, 0x72,
	0x65, 0x77, 0x61, 0x72, 0x64, 0x73, 0x18, 0x02, 0x20, 0x03, 0x28, 0x0c, 0x52, 0x07, 0x72, 0x65,
	0x77, 0x61, 0x72, 0x64, 0x73, 0x22, 0x4f, 0x0a, 0x0a, 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x45, 0x76,
	0x65, 0x6e, 0x74, 0x12, 0x0e, 0x0a, 0x02, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x05, 0x52,
	0x02, 0x69, 0x64, 0x12, 0x22, 0x0a, 0x0a, 0x69, 0x6e, 0x69, 0x74, 0x5f, 0x73, 0x74, 0x61, 0x74,
	0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0c, 0x48, 0x00, 0x52, 0x09, 0x69, 0x6e, 0x69, 0x74, 0x53,
	0x74, 0x61, 0x74, 0x65, 0x88, 0x01, 0x01, 0x42, 0x0d, 0x0a, 0x0b, 0x5f, 0x69, 0x6e, 0x69, 0x74,
	0x5f, 0x73, 0x74, 0x61, 0x74, 0x65, 0x22, 0xc5, 0x02, 0x0a, 0x11, 0x53, 0x75, 0x62, 0x73, 0x63,
	0x72, 0x69, 0x62, 0x65, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x30, 0x0a, 0x04,
	0x74, 0x79, 0x70, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0e, 0x32, 0x1c, 0x2e, 0x64, 0x69, 0x73,
	0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x2e, 0x53, 0x65, 0x72, 0x76, 0x65, 0x72, 0x45,
	0x76, 0x65, 0x6e, 0x74, 0x54, 0x79, 0x70, 0x65, 0x52, 0x04, 0x74, 0x79, 0x70, 0x65, 0x12, 0x3c,
	0x0a, 0x0a, 0x73, 0x65, 0x6e, 0x64, 0x5f, 0x73, 0x74, 0x61, 0x74, 0x65, 0x18, 0x02, 0x20, 0x01,
	0x28, 0x0b, 0x32, 0x1b, 0x2e, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64,
	0x2e, 0x53, 0x65, 0x6e, 0x64, 0x53, 0x74, 0x61, 0x74, 0x65, 0x45, 0x76, 0x65, 0x6e, 0x74, 0x48,
	0x00, 0x52, 0x09, 0x73, 0x65, 0x6e, 0x64, 0x53, 0x74, 0x61, 0x74, 0x65, 0x12, 0x48, 0x0a, 0x0e,
	0x65, 0x76, 0x61, 0x6c, 0x75, 0x61, 0x74, 0x65, 0x5f, 0x62, 0x61, 0x74, 0x63, 0x68, 0x18, 0x03,
	0x20, 0x01, 0x28, 0x0b, 0x32, 0x1f, 0x2e, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74,
	0x65, 0x64, 0x2e, 0x45, 0x76, 0x61, 0x6c, 0x75, 0x61, 0x74, 0x65, 0x42, 0x61, 0x74, 0x63, 0x68,
	0x45, 0x76, 0x65, 0x6e, 0x74, 0x48, 0x00, 0x52, 0x0d, 0x65, 0x76, 0x61, 0x6c, 0x75, 0x61, 0x74,
	0x65, 0x42, 0x61, 0x74, 0x63, 0x68, 0x12, 0x3c, 0x0a, 0x0a, 0x6f, 0x70, 0x74, 0x69, 0x6d, 0x5f,
	0x73, 0x74, 0x65, 0x70, 0x18, 0x04, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x1b, 0x2e, 0x64, 0x69, 0x73,
	0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x2e, 0x4f, 0x70, 0x74, 0x69, 0x6d, 0x53, 0x74,
	0x65, 0x70, 0x45, 0x76, 0x65, 0x6e, 0x74, 0x48, 0x00, 0x52, 0x09, 0x6f, 0x70, 0x74, 0x69, 0x6d,
	0x53, 0x74, 0x65, 0x70, 0x12, 0x2f, 0x0a, 0x05, 0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x18, 0x05, 0x20,
	0x01, 0x28, 0x0b, 0x32, 0x17, 0x2e, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65,
	0x64, 0x2e, 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x45, 0x76, 0x65, 0x6e, 0x74, 0x48, 0x00, 0x52, 0x05,
	0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x42, 0x07, 0x0a, 0x05, 0x65, 0x76, 0x65, 0x6e, 0x74, 0x2a, 0x6f,
	0x0a, 0x0f, 0x53, 0x65, 0x72, 0x76, 0x65, 0x72, 0x45, 0x76, 0x65, 0x6e, 0x74, 0x54, 0x79, 0x70,
	0x65, 0x12, 0x0b, 0x0a, 0x07, 0x55, 0x4e, 0x4b, 0x4e, 0x4f, 0x57, 0x4e, 0x10, 0x00, 0x12, 0x0e,
	0x0a, 0x0a, 0x53, 0x45, 0x4e, 0x44, 0x5f, 0x53, 0x54, 0x41, 0x54, 0x45, 0x10, 0x01, 0x12, 0x12,
	0x0a, 0x0e, 0x45, 0x56, 0x41, 0x4c, 0x55, 0x41, 0x54, 0x45, 0x5f, 0x42, 0x41, 0x54, 0x43, 0x48,
	0x10, 0x02, 0x12, 0x10, 0x0a, 0x0c, 0x53, 0x54, 0x41, 0x54, 0x45, 0x5f, 0x55, 0x50, 0x44, 0x41,
	0x54, 0x45, 0x10, 0x03, 0x12, 0x0e, 0x0a, 0x0a, 0x4f, 0x50, 0x54, 0x49, 0x4d, 0x5f, 0x53, 0x54,
	0x45, 0x50, 0x10, 0x04, 0x12, 0x09, 0x0a, 0x05, 0x48, 0x45, 0x4c, 0x4c, 0x4f, 0x10, 0x05, 0x32,
	0xb6, 0x02, 0x0a, 0x09, 0x45, 0x53, 0x53, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x12, 0x3d, 0x0a,
	0x04, 0x44, 0x6f, 0x6e, 0x65, 0x12, 0x18, 0x2e, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75,
	0x74, 0x65, 0x64, 0x2e, 0x44, 0x6f, 0x6e, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a,
	0x19, 0x2e, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x2e, 0x44, 0x6f,
	0x6e, 0x65, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00, 0x12, 0x4c, 0x0a, 0x09,
	0x48, 0x65, 0x61, 0x72, 0x74, 0x62, 0x65, 0x61, 0x74, 0x12, 0x1d, 0x2e, 0x64, 0x69, 0x73, 0x74,
	0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x2e, 0x48, 0x65, 0x61, 0x72, 0x74, 0x62, 0x65, 0x61,
	0x74, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1e, 0x2e, 0x64, 0x69, 0x73, 0x74, 0x72,
	0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x2e, 0x48, 0x65, 0x61, 0x72, 0x74, 0x62, 0x65, 0x61, 0x74,
	0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00, 0x12, 0x4c, 0x0a, 0x09, 0x53, 0x65,
	0x6e, 0x64, 0x53, 0x74, 0x61, 0x74, 0x65, 0x12, 0x1d, 0x2e, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69,
	0x62, 0x75, 0x74, 0x65, 0x64, 0x2e, 0x53, 0x65, 0x6e, 0x64, 0x53, 0x74, 0x61, 0x74, 0x65, 0x52,
	0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1e, 0x2e, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62,
	0x75, 0x74, 0x65, 0x64, 0x2e, 0x53, 0x65, 0x6e, 0x64, 0x53, 0x74, 0x61, 0x74, 0x65, 0x52, 0x65,
	0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00, 0x12, 0x4e, 0x0a, 0x09, 0x53, 0x75, 0x62, 0x73,
	0x63, 0x72, 0x69, 0x62, 0x65, 0x12, 0x1d, 0x2e, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75,
	0x74, 0x65, 0x64, 0x2e, 0x53, 0x75, 0x62, 0x73, 0x63, 0x72, 0x69, 0x62, 0x65, 0x52, 0x65, 0x71,
	0x75, 0x65, 0x73, 0x74, 0x1a, 0x1e, 0x2e, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74,
	0x65, 0x64, 0x2e, 0x53, 0x75, 0x62, 0x73, 0x63, 0x72, 0x69, 0x62, 0x65, 0x52, 0x65, 0x73, 0x70,
	0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00, 0x30, 0x01, 0x42, 0xac, 0x01, 0x0a, 0x0f, 0x63, 0x6f, 0x6d,
	0x2e, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x42, 0x10, 0x44, 0x69,
	0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x50, 0x01,
	0x5a, 0x3b, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x6e, 0x65, 0x75,
	0x72, 0x6f, 0x2d, 0x73, 0x6f, 0x75, 0x70, 0x2f, 0x65, 0x73, 0x2d, 0x74, 0x6f, 0x72, 0x63, 0x68,
	0x2f, 0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2f, 0x70, 0x6b, 0x67, 0x2f, 0x70, 0x72, 0x6f, 0x74,
	0x6f, 0x2f, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0xa2, 0x02, 0x03,
	0x44, 0x58, 0x58, 0xaa, 0x02, 0x0b, 0x44, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65,
	0x64, 0xca, 0x02, 0x0b, 0x44, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0xe2,
	0x02, 0x17, 0x44, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x5c, 0x47, 0x50,
	0x42, 0x4d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0xea, 0x02, 0x0b, 0x44, 0x69, 0x73, 0x74,
	0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_distributed_distributed_proto_rawDescOnce sync.Once
	file_distributed_distributed_proto_rawDescData = file_distributed_distributed_proto_rawDesc
)

func file_distributed_distributed_proto_rawDescGZIP() []byte {
	file_distributed_distributed_proto_rawDescOnce.Do(func() {
		file_distributed_distributed_proto_rawDescData = protoimpl.X.CompressGZIP(file_distributed_distributed_proto_rawDescData)
	})
	return file_distributed_distributed_proto_rawDescData
}

var file_distributed_distributed_proto_enumTypes = make([]protoimpl.EnumInfo, 1)
var file_distributed_distributed_proto_msgTypes = make([]protoimpl.MessageInfo, 13)
var file_distributed_distributed_proto_goTypes = []interface{}{
	(ServerEventType)(0),          // 0: distributed.ServerEventType
	(*Slice)(nil),                 // 1: distributed.Slice
	(*HeartbeatRequest)(nil),      // 2: distributed.HeartbeatRequest
	(*HeartbeatResponse)(nil),     // 3: distributed.HeartbeatResponse
	(*DoneRequest)(nil),           // 4: distributed.DoneRequest
	(*DoneResponse)(nil),          // 5: distributed.DoneResponse
	(*SendStateRequest)(nil),      // 6: distributed.SendStateRequest
	(*SendStateResponse)(nil),     // 7: distributed.SendStateResponse
	(*SubscribeRequest)(nil),      // 8: distributed.SubscribeRequest
	(*SendStateEvent)(nil),        // 9: distributed.SendStateEvent
	(*EvaluateBatchEvent)(nil),    // 10: distributed.EvaluateBatchEvent
	(*OptimStepEvent)(nil),        // 11: distributed.OptimStepEvent
	(*HelloEvent)(nil),            // 12: distributed.HelloEvent
	(*SubscribeResponse)(nil),     // 13: distributed.SubscribeResponse
	(*timestamppb.Timestamp)(nil), // 14: google.protobuf.Timestamp
}
var file_distributed_distributed_proto_depIdxs = []int32{
	14, // 0: distributed.HeartbeatRequest.timestamp:type_name -> google.protobuf.Timestamp
	1,  // 1: distributed.DoneRequest.slice:type_name -> distributed.Slice
	1,  // 2: distributed.EvaluateBatchEvent.pop_slice:type_name -> distributed.Slice
	0,  // 3: distributed.SubscribeResponse.type:type_name -> distributed.ServerEventType
	9,  // 4: distributed.SubscribeResponse.send_state:type_name -> distributed.SendStateEvent
	10, // 5: distributed.SubscribeResponse.evaluate_batch:type_name -> distributed.EvaluateBatchEvent
	11, // 6: distributed.SubscribeResponse.optim_step:type_name -> distributed.OptimStepEvent
	12, // 7: distributed.SubscribeResponse.hello:type_name -> distributed.HelloEvent
	4,  // 8: distributed.ESService.Done:input_type -> distributed.DoneRequest
	2,  // 9: distributed.ESService.Heartbeat:input_type -> distributed.HeartbeatRequest
	6,  // 10: distributed.ESService.SendState:input_type -> distributed.SendStateRequest
	8,  // 11: distributed.ESService.Subscribe:input_type -> distributed.SubscribeRequest
	5,  // 12: distributed.ESService.Done:output_type -> distributed.DoneResponse
	3,  // 13: distributed.ESService.Heartbeat:output_type -> distributed.HeartbeatResponse
	7,  // 14: distributed.ESService.SendState:output_type -> distributed.SendStateResponse
	13, // 15: distributed.ESService.Subscribe:output_type -> distributed.SubscribeResponse
	12, // [12:16] is the sub-list for method output_type
	8,  // [8:12] is the sub-list for method input_type
	8,  // [8:8] is the sub-list for extension type_name
	8,  // [8:8] is the sub-list for extension extendee
	0,  // [0:8] is the sub-list for field type_name
}

func init() { file_distributed_distributed_proto_init() }
func file_distributed_distributed_proto_init() {
	if File_distributed_distributed_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_distributed_distributed_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Slice); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_distributed_distributed_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*HeartbeatRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_distributed_distributed_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*HeartbeatResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_distributed_distributed_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*DoneRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_distributed_distributed_proto_msgTypes[4].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*DoneResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_distributed_distributed_proto_msgTypes[5].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SendStateRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_distributed_distributed_proto_msgTypes[6].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SendStateResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_distributed_distributed_proto_msgTypes[7].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SubscribeRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_distributed_distributed_proto_msgTypes[8].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SendStateEvent); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_distributed_distributed_proto_msgTypes[9].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*EvaluateBatchEvent); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_distributed_distributed_proto_msgTypes[10].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*OptimStepEvent); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_distributed_distributed_proto_msgTypes[11].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*HelloEvent); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_distributed_distributed_proto_msgTypes[12].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SubscribeResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	file_distributed_distributed_proto_msgTypes[11].OneofWrappers = []interface{}{}
	file_distributed_distributed_proto_msgTypes[12].OneofWrappers = []interface{}{
		(*SubscribeResponse_SendState)(nil),
		(*SubscribeResponse_EvaluateBatch)(nil),
		(*SubscribeResponse_OptimStep)(nil),
		(*SubscribeResponse_Hello)(nil),
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_distributed_distributed_proto_rawDesc,
			NumEnums:      1,
			NumMessages:   13,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_distributed_distributed_proto_goTypes,
		DependencyIndexes: file_distributed_distributed_proto_depIdxs,
		EnumInfos:         file_distributed_distributed_proto_enumTypes,
		MessageInfos:      file_distributed_distributed_proto_msgTypes,
	}.Build()
	File_distributed_distributed_proto = out.File
	file_distributed_distributed_proto_rawDesc = nil
	file_distributed_distributed_proto_goTypes = nil
	file_distributed_distributed_proto_depIdxs = nil
}
