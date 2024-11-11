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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1d\x64istributed/distributed.proto\x12\x0b\x64istributed\x1a\x1fgoogle/protobuf/timestamp.proto\"#\n\x05Slice\x12\r\n\x05start\x18\x01 \x01(\x05\x12\x0b\n\x03\x65nd\x18\x02 \x01(\x05\"M\n\x10HeartbeatRequest\x12\n\n\x02id\x18\x01 \x01(\x05\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"\x1f\n\x11HeartbeatResponse\x12\n\n\x02ok\x18\x01 \x01(\x08\"M\n\x0b\x44oneRequest\x12\n\n\x02id\x18\x01 \x01(\x05\x12!\n\x05slice\x18\x02 \x01(\x0b\x32\x12.distributed.Slice\x12\x0f\n\x07rewards\x18\x03 \x03(\x0c\"\x0e\n\x0c\x44oneResponse\"-\n\x10SendStateRequest\x12\n\n\x02id\x18\x01 \x01(\x05\x12\r\n\x05state\x18\x02 \x01(\x0c\"\x13\n\x11SendStateResponse\"5\n\x10SubscribeRequest\x12\x10\n\x08num_cpus\x18\x01 \x01(\x05\x12\x0f\n\x07num_pop\x18\x02 \x01(\x05\"\x10\n\x0eSendStateEvent\";\n\x12\x45valuateBatchEvent\x12%\n\tpop_slice\x18\x01 \x01(\x0b\x32\x12.distributed.Slice\"2\n\x0eOptimStepEvent\x12\x0f\n\x07logging\x18\x01 \x01(\x08\x12\x0f\n\x07rewards\x18\x02 \x03(\x0c\"@\n\nHelloEvent\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x17\n\ninit_state\x18\x02 \x01(\x0cH\x00\x88\x01\x01\x42\r\n\x0b_init_state\"\x93\x02\n\x11SubscribeResponse\x12*\n\x04type\x18\x01 \x01(\x0e\x32\x1c.distributed.ServerEventType\x12\x31\n\nsend_state\x18\x02 \x01(\x0b\x32\x1b.distributed.SendStateEventH\x00\x12\x39\n\x0e\x65valuate_batch\x18\x03 \x01(\x0b\x32\x1f.distributed.EvaluateBatchEventH\x00\x12\x31\n\noptim_step\x18\x04 \x01(\x0b\x32\x1b.distributed.OptimStepEventH\x00\x12(\n\x05hello\x18\x05 \x01(\x0b\x32\x17.distributed.HelloEventH\x00\x42\x07\n\x05\x65vent*o\n\x0fServerEventType\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x0e\n\nSEND_STATE\x10\x01\x12\x12\n\x0e\x45VALUATE_BATCH\x10\x02\x12\x10\n\x0cSTATE_UPDATE\x10\x03\x12\x0e\n\nOPTIM_STEP\x10\x04\x12\t\n\x05HELLO\x10\x05\x32\xb6\x02\n\tESService\x12=\n\x04\x44one\x12\x18.distributed.DoneRequest\x1a\x19.distributed.DoneResponse\"\x00\x12L\n\tHeartbeat\x12\x1d.distributed.HeartbeatRequest\x1a\x1e.distributed.HeartbeatResponse\"\x00\x12L\n\tSendState\x12\x1d.distributed.SendStateRequest\x1a\x1e.distributed.SendStateResponse\"\x00\x12N\n\tSubscribe\x12\x1d.distributed.SubscribeRequest\x1a\x1e.distributed.SubscribeResponse\"\x00\x30\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'distributed.distributed_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_SERVEREVENTTYPE']._serialized_start=921
  _globals['_SERVEREVENTTYPE']._serialized_end=1032
  _globals['_SLICE']._serialized_start=79
  _globals['_SLICE']._serialized_end=114
  _globals['_HEARTBEATREQUEST']._serialized_start=116
  _globals['_HEARTBEATREQUEST']._serialized_end=193
  _globals['_HEARTBEATRESPONSE']._serialized_start=195
  _globals['_HEARTBEATRESPONSE']._serialized_end=226
  _globals['_DONEREQUEST']._serialized_start=228
  _globals['_DONEREQUEST']._serialized_end=305
  _globals['_DONERESPONSE']._serialized_start=307
  _globals['_DONERESPONSE']._serialized_end=321
  _globals['_SENDSTATEREQUEST']._serialized_start=323
  _globals['_SENDSTATEREQUEST']._serialized_end=368
  _globals['_SENDSTATERESPONSE']._serialized_start=370
  _globals['_SENDSTATERESPONSE']._serialized_end=389
  _globals['_SUBSCRIBEREQUEST']._serialized_start=391
  _globals['_SUBSCRIBEREQUEST']._serialized_end=444
  _globals['_SENDSTATEEVENT']._serialized_start=446
  _globals['_SENDSTATEEVENT']._serialized_end=462
  _globals['_EVALUATEBATCHEVENT']._serialized_start=464
  _globals['_EVALUATEBATCHEVENT']._serialized_end=523
  _globals['_OPTIMSTEPEVENT']._serialized_start=525
  _globals['_OPTIMSTEPEVENT']._serialized_end=575
  _globals['_HELLOEVENT']._serialized_start=577
  _globals['_HELLOEVENT']._serialized_end=641
  _globals['_SUBSCRIBERESPONSE']._serialized_start=644
  _globals['_SUBSCRIBERESPONSE']._serialized_end=919
  _globals['_ESSERVICE']._serialized_start=1035
  _globals['_ESSERVICE']._serialized_end=1345
# @@protoc_insertion_point(module_scope)
