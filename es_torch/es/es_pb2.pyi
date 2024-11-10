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
    NEXT_EPOCH: _ClassVar[ServerEventType]
    STATE_UPDATE: _ClassVar[ServerEventType]
UNKNOWN: ServerEventType
SEND_STATE: ServerEventType
NEXT_EPOCH: ServerEventType
STATE_UPDATE: ServerEventType

class HelloRequest(_message.Message):
    __slots__ = ("num_cpus",)
    NUM_CPUS_FIELD_NUMBER: _ClassVar[int]
    num_cpus: int
    def __init__(self, num_cpus: _Optional[int] = ...) -> None: ...

class HelloResponse(_message.Message):
    __slots__ = ("id", "state")
    ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    id: int
    state: bytes
    def __init__(self, id: _Optional[int] = ..., state: _Optional[bytes] = ...) -> None: ...

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
    __slots__ = ("id", "reward")
    ID_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    id: int
    reward: bytes
    def __init__(self, id: _Optional[int] = ..., reward: _Optional[bytes] = ...) -> None: ...

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
    __slots__ = ("type", "rewards", "updated_state")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    REWARDS_FIELD_NUMBER: _ClassVar[int]
    UPDATED_STATE_FIELD_NUMBER: _ClassVar[int]
    type: ServerEventType
    rewards: _containers.RepeatedScalarFieldContainer[bytes]
    updated_state: bytes
    def __init__(self, type: _Optional[_Union[ServerEventType, str]] = ..., rewards: _Optional[_Iterable[bytes]] = ..., updated_state: _Optional[bytes] = ...) -> None: ...
