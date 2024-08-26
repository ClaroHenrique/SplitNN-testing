import socket

server_host = '127.0.0.1'
server_port = 8092
num_clients = 2

serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind((server_host, server_port))
serversocket.listen(5) # become a server socket, maximum 5 connections

while True:
    print("👀")
    connections = []
    for i in range(num_clients):
        print(f"try connection {i}")
        connection, address = serversocket.accept()
        connections.append(connection)

    for conn in connections:
        print(f"try send to {i}")
        conn.send(b'hello world from server!')
        print(f"try recv from {i}")
        buf = conn.recv(1024)
        print(buf)
