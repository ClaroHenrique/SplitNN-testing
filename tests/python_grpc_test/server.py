import grpc
from concurrent import futures
import time

import greeter_pb2
import greeter_pb2_grpc


class GreeterServicer(greeter_pb2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        return greeter_pb2.HelloReply(message=f"Olá, {request.name}!")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    greeter_pb2_grpc.add_GreeterServicer_to_server(GreeterServicer(), server)
    # escuta em todas as interfaces de rede
    server.add_insecure_port("0.0.0.0:50051")
    server.start()
    print("Servidor gRPC rodando na porta 50051...")
    input("Pressione ENTER para encerrar o servidor.\n")
    server.stop(0)


if __name__ == "__main__":
    serve()
