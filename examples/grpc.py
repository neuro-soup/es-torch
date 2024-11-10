import grpc
import es_torch.es.es_pb2 as es_pb2
import es_torch.es.es_pb2_grpc as es_pb2_grpc


channel = grpc.insecure_channel('localhost:8080')
stub = es_pb2_grpc.ESServiceStub(channel)

hello_request = es_pb2.HelloRequest(num_cpus=4)
hello_response = stub.Hello(hello_request)
print("Hello Response:", hello_response.id, hello_response.state)
