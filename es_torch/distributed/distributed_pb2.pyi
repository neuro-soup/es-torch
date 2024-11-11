from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ServerEventType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN: _ClassVar[ServerEventType]
    SEND_STATE: _ClassVar[ServerEventType]
    EVALUATE_BATCH: _ClassVar[ServerEventType]
    STATE_UPDATE: _ClassVar[ServerEventType]
    OPTIM_STEP: _ClassVar[ServerEventType]
UNKNOWN: ServerEventType
SEND_STATE: ServerEventType
EVALUATE_BATCH: ServerEventType
STATE_UPDATE: ServerEventType
OPTIM_STEP: ServerEventType

class Slice(_message.Message):
    __slots__ = ("start", "end")
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    start: int
    end: int
    def __init__(self, start: _Optional[int] = ..., end: _Optional[int] = ...) -> None: ...

class HelloRequest(_message.Message):
    __slots__ = ("num_cpus",)
    NUM_CPUS_FIELD_NUMBER: _ClassVar[int]
    num_cpus: int
    def __init__(self, num_cpus: _Optional[int] = ...) -> None: ...

class HelloResponse(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("id", "timestamp")
    ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    id: int
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, id: _Optional[int] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ("ok",)
    OK_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    def __init__(self, ok: bool = ...) -> None: ...

class DoneRequest(_message.Message):
    __slots__ = ("id", "reward_batch")
    ID_FIELD_NUMBER: _ClassVar[int]
    REWARD_BATCH_FIELD_NUMBER: _ClassVar[int]
    id: int
    reward_batch: bytes
    def __init__(self, id: _Optional[int] = ..., reward_batch: _Optional[bytes] = ...) -> None: ...

class DoneResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SendStateRequest(_message.Message):
    __slots__ = ("id", "state")
    ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    id: int
    state: bytes
    def __init__(self, id: _Optional[int] = ..., state: _Optional[bytes] = ...) -> None: ...

class SendStateResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SubscribeRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class SubscribeResponse(_message.Message):
    __slots__ = ("type", "optim_rewards", "eval_pop_slices", "update_state")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    OPTIM_REWARDS_FIELD_NUMBER: _ClassVar[int]
    EVAL_POP_SLICES_FIELD_NUMBER: _ClassVar[int]
    UPDATE_STATE_FIELD_NUMBER: _ClassVar[int]
    type: ServerEventType
    optim_rewards: _containers.RepeatedScalarFieldContainer[bytes]
    eval_pop_slices: _containers.RepeatedCompositeFieldContainer[Slice]
    update_state: bytes
    def __init__(self, type: _Optional[_Union[ServerEventType, str]] = ..., optim_rewards: _Optional[_Iterable[bytes]] = ..., eval_pop_slices: _Optional[_Iterable[_Union[Slice, _Mapping]]] = ..., update_state: _Optional[bytes] = ...) -> None: ...
