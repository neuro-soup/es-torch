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
    HELLO: _ClassVar[ServerEventType]
UNKNOWN: ServerEventType
SEND_STATE: ServerEventType
EVALUATE_BATCH: ServerEventType
STATE_UPDATE: ServerEventType
OPTIM_STEP: ServerEventType
HELLO: ServerEventType

class Slice(_message.Message):
    __slots__ = ("start", "end")
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    start: int
    end: int
    def __init__(self, start: _Optional[int] = ..., end: _Optional[int] = ...) -> None: ...

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

class SendWandbRunRequest(_message.Message):
    __slots__ = ("id", "wandb_run")
    ID_FIELD_NUMBER: _ClassVar[int]
    WANDB_RUN_FIELD_NUMBER: _ClassVar[int]
    id: int
    wandb_run: bytes
    def __init__(self, id: _Optional[int] = ..., wandb_run: _Optional[bytes] = ...) -> None: ...

class SendWandbRunResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SubscribeRequest(_message.Message):
    __slots__ = ("num_cpus",)
    NUM_CPUS_FIELD_NUMBER: _ClassVar[int]
    num_cpus: int
    def __init__(self, num_cpus: _Optional[int] = ...) -> None: ...

class SendStateEvent(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class EvaluateBatchEvent(_message.Message):
    __slots__ = ("pop_slice",)
    POP_SLICE_FIELD_NUMBER: _ClassVar[int]
    pop_slice: Slice
    def __init__(self, pop_slice: _Optional[_Union[Slice, _Mapping]] = ...) -> None: ...

class OptimStepEvent(_message.Message):
    __slots__ = ("logging", "rewards")
    LOGGING_FIELD_NUMBER: _ClassVar[int]
    REWARDS_FIELD_NUMBER: _ClassVar[int]
    logging: bool
    rewards: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, logging: bool = ..., rewards: _Optional[_Iterable[bytes]] = ...) -> None: ...

class HelloEvent(_message.Message):
    __slots__ = ("id", "init_state")
    ID_FIELD_NUMBER: _ClassVar[int]
    INIT_STATE_FIELD_NUMBER: _ClassVar[int]
    id: int
    init_state: bytes
    def __init__(self, id: _Optional[int] = ..., init_state: _Optional[bytes] = ...) -> None: ...

class SubscribeResponse(_message.Message):
    __slots__ = ("type", "send_state", "evaluate_batch", "optim_step", "hello")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    SEND_STATE_FIELD_NUMBER: _ClassVar[int]
    EVALUATE_BATCH_FIELD_NUMBER: _ClassVar[int]
    OPTIM_STEP_FIELD_NUMBER: _ClassVar[int]
    HELLO_FIELD_NUMBER: _ClassVar[int]
    type: ServerEventType
    send_state: SendStateEvent
    evaluate_batch: EvaluateBatchEvent
    optim_step: OptimStepEvent
    hello: HelloEvent
    def __init__(self, type: _Optional[_Union[ServerEventType, str]] = ..., send_state: _Optional[_Union[SendStateEvent, _Mapping]] = ..., evaluate_batch: _Optional[_Union[EvaluateBatchEvent, _Mapping]] = ..., optim_step: _Optional[_Union[OptimStepEvent, _Mapping]] = ..., hello: _Optional[_Union[HelloEvent, _Mapping]] = ...) -> None: ...
