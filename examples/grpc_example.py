import grpc
import torch

import es_torch.es.es_pb2 as es_pb2
import es_torch.es.es_pb2_grpc as es_pb2_grpc

channel = grpc.insecure_channel("localhost:8080")
stub = es_pb2_grpc.ESServiceStub(channel)

resp: es_pb2.HelloResponse = stub.Hello(es_pb2.HelloRequest(num_cpus=4))
print(resp)
resp: es_pb2.DoneResponse = stub.Done(
    es_pb2.DoneRequest(
        id=resp.id,
        reward=torch.tensor([1, 2, 3]).numpy().tobytes(),
    )
)
print(resp)
