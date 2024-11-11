proto-py:
    python -m grpc_tools.protoc -I=proto --python_out=es_torch --pyi_out=es_torch --grpc_python_out=es_torch ./proto/distributed/distributed.proto
