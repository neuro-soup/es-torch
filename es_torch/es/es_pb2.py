# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: es/es.proto
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
    'es/es.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0b\x65s/es.proto\x12\x02\x65s\x1a\x1fgoogle/protobuf/timestamp.proto\" \n\x0cHelloRequest\x12\x10\n\x08num_cpus\x18\x01 \x01(\x05\"*\n\rHelloResponse\x12\n\n\x02id\x18\x01 \x01(\x05\x12\r\n\x05state\x18\x02 \x01(\x0c\"M\n\x10HeartbeatRequest\x12\n\n\x02id\x18\x01 \x01(\x05\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"\x1f\n\x11HeartbeatResponse\x12\n\n\x02ok\x18\x01 \x01(\x08\")\n\x0b\x44oneRequest\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x0e\n\x06reward\x18\x02 \x01(\x0c\"\x0e\n\x0c\x44oneResponse\"-\n\x10SendStateRequest\x12\n\n\x02id\x18\x01 \x01(\x05\x12\r\n\x05state\x18\x02 \x01(\x0c\"\x13\n\x11SendStateResponse\"\x1e\n\x10SubscribeRequest\x12\n\n\x02id\x18\x01 \x01(\x05\"u\n\x11SubscribeResponse\x12!\n\x04type\x18\x01 \x01(\x0e\x32\x13.es.ServerEventType\x12\x0f\n\x07rewards\x18\x02 \x03(\x0c\x12\x1a\n\rupdated_state\x18\x03 \x01(\x0cH\x00\x88\x01\x01\x42\x10\n\x0e_updated_state*P\n\x0fServerEventType\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x0e\n\nSEND_STATE\x10\x01\x12\x0e\n\nNEXT_EPOCH\x10\x02\x12\x10\n\x0cSTATE_UPDATE\x10\x03\x32\x9e\x02\n\tESService\x12.\n\x05Hello\x12\x10.es.HelloRequest\x1a\x11.es.HelloResponse\"\x00\x12+\n\x04\x44one\x12\x0f.es.DoneRequest\x1a\x10.es.DoneResponse\"\x00\x12:\n\tHeartbeat\x12\x14.es.HeartbeatRequest\x1a\x15.es.HeartbeatResponse\"\x00\x12:\n\tSendState\x12\x14.es.SendStateRequest\x1a\x15.es.SendStateResponse\"\x00\x12<\n\tSubscribe\x12\x14.es.SubscribeRequest\x1a\x15.es.SubscribeResponse\"\x00\x30\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'es.es_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_SERVEREVENTTYPE']._serialized_start=520
  _globals['_SERVEREVENTTYPE']._serialized_end=600
  _globals['_HELLOREQUEST']._serialized_start=52
  _globals['_HELLOREQUEST']._serialized_end=84
  _globals['_HELLORESPONSE']._serialized_start=86
  _globals['_HELLORESPONSE']._serialized_end=128
  _globals['_HEARTBEATREQUEST']._serialized_start=130
  _globals['_HEARTBEATREQUEST']._serialized_end=207
  _globals['_HEARTBEATRESPONSE']._serialized_start=209
  _globals['_HEARTBEATRESPONSE']._serialized_end=240
  _globals['_DONEREQUEST']._serialized_start=242
  _globals['_DONEREQUEST']._serialized_end=283
  _globals['_DONERESPONSE']._serialized_start=285
  _globals['_DONERESPONSE']._serialized_end=299
  _globals['_SENDSTATEREQUEST']._serialized_start=301
  _globals['_SENDSTATEREQUEST']._serialized_end=346
  _globals['_SENDSTATERESPONSE']._serialized_start=348
  _globals['_SENDSTATERESPONSE']._serialized_end=367
  _globals['_SUBSCRIBEREQUEST']._serialized_start=369
  _globals['_SUBSCRIBEREQUEST']._serialized_end=399
  _globals['_SUBSCRIBERESPONSE']._serialized_start=401
  _globals['_SUBSCRIBERESPONSE']._serialized_end=518
  _globals['_ESSERVICE']._serialized_start=603
  _globals['_ESSERVICE']._serialized_end=889
# @@protoc_insertion_point(module_scope)