# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: distributed/distributed.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'distributed/distributed.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1d\x64istributed/distributed.proto\x12\x0b\x64istributed\x1a\x1fgoogle/protobuf/timestamp.proto\"#\n\x05Slice\x12\r\n\x05start\x18\x01 \x01(\x05\x12\x0b\n\x03\x65nd\x18\x02 \x01(\x05\" \n\x0cHelloRequest\x12\x10\n\x08num_cpus\x18\x01 \x01(\x05\"\x1b\n\rHelloResponse\x12\n\n\x02id\x18\x01 \x01(\x05\"M\n\x10HeartbeatRequest\x12\n\n\x02id\x18\x01 \x01(\x05\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"\x1f\n\x11HeartbeatResponse\x12\n\n\x02ok\x18\x01 \x01(\x08\"/\n\x0b\x44oneRequest\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x14\n\x0creward_batch\x18\x02 \x01(\x0c\"\x0e\n\x0c\x44oneResponse\"-\n\x10SendStateRequest\x12\n\n\x02id\x18\x01 \x01(\x05\x12\r\n\x05state\x18\x02 \x01(\x0c\"\x13\n\x11SendStateResponse\"\x1e\n\x10SubscribeRequest\x12\n\n\x02id\x18\x01 \x01(\x05\"\xaf\x01\n\x11SubscribeResponse\x12*\n\x04type\x18\x01 \x01(\x0e\x32\x1c.distributed.ServerEventType\x12\x15\n\roptim_rewards\x18\x02 \x03(\x0c\x12+\n\x0f\x65val_pop_slices\x18\x03 \x03(\x0b\x32\x12.distributed.Slice\x12\x19\n\x0cupdate_state\x18\x04 \x01(\x0cH\x00\x88\x01\x01\x42\x0f\n\r_update_state*d\n\x0fServerEventType\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x0e\n\nSEND_STATE\x10\x01\x12\x12\n\x0e\x45VALUATE_BATCH\x10\x02\x12\x10\n\x0cSTATE_UPDATE\x10\x03\x12\x0e\n\nOPTIM_STEP\x10\x04\x32\xf8\x02\n\tESService\x12@\n\x05Hello\x12\x19.distributed.HelloRequest\x1a\x1a.distributed.HelloResponse\"\x00\x12=\n\x04\x44one\x12\x18.distributed.DoneRequest\x1a\x19.distributed.DoneResponse\"\x00\x12L\n\tHeartbeat\x12\x1d.distributed.HeartbeatRequest\x1a\x1e.distributed.HeartbeatResponse\"\x00\x12L\n\tSendState\x12\x1d.distributed.SendStateRequest\x1a\x1e.distributed.SendStateResponse\"\x00\x12N\n\tSubscribe\x12\x1d.distributed.SubscribeRequest\x1a\x1e.distributed.SubscribeResponse\"\x00\x30\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'distributed.distributed_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_SERVEREVENTTYPE']._serialized_start=634
  _globals['_SERVEREVENTTYPE']._serialized_end=734
  _globals['_SLICE']._serialized_start=79
  _globals['_SLICE']._serialized_end=114
  _globals['_HELLOREQUEST']._serialized_start=116
  _globals['_HELLOREQUEST']._serialized_end=148
  _globals['_HELLORESPONSE']._serialized_start=150
  _globals['_HELLORESPONSE']._serialized_end=177
  _globals['_HEARTBEATREQUEST']._serialized_start=179
  _globals['_HEARTBEATREQUEST']._serialized_end=256
  _globals['_HEARTBEATRESPONSE']._serialized_start=258
  _globals['_HEARTBEATRESPONSE']._serialized_end=289
  _globals['_DONEREQUEST']._serialized_start=291
  _globals['_DONEREQUEST']._serialized_end=338
  _globals['_DONERESPONSE']._serialized_start=340
  _globals['_DONERESPONSE']._serialized_end=354
  _globals['_SENDSTATEREQUEST']._serialized_start=356
  _globals['_SENDSTATEREQUEST']._serialized_end=401
  _globals['_SENDSTATERESPONSE']._serialized_start=403
  _globals['_SENDSTATERESPONSE']._serialized_end=422
  _globals['_SUBSCRIBEREQUEST']._serialized_start=424
  _globals['_SUBSCRIBEREQUEST']._serialized_end=454
  _globals['_SUBSCRIBERESPONSE']._serialized_start=457
  _globals['_SUBSCRIBERESPONSE']._serialized_end=632
  _globals['_ESSERVICE']._serialized_start=737
  _globals['_ESSERVICE']._serialized_end=1113
# @@protoc_insertion_point(module_scope)
