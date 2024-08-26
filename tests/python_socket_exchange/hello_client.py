import socket

server_host = '127.0.0.1'
server_port = 8092

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect((server_host, server_port))

buf = clientsocket.recv(1024)
print(buf)
clientsocket.send(buf)
print("sent")
clientsocket.send(bytes(str(buf) + " take this thing back!", 'UTF-8'))


