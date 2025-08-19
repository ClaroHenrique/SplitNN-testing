import grpc
import greeter_pb2
import greeter_pb2_grpc


def run():
    # substitua pelo IP real do servidor
    server_ip = "11.11.11.11:50051"
    with grpc.insecure_channel(server_ip) as channel:
        stub = greeter_pb2_grpc.GreeterStub(channel)
        response = stub.SayHello(greeter_pb2.HelloRequest(name="Cliente remoto"))
    print("Resposta do servidor:", response.message)


if __name__ == "__main__":
    run()
