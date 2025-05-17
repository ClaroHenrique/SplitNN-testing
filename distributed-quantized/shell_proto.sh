python3 -m grpc_tools.protoc ./proto/distributed_learning.proto \
  --proto_path=.              \
  --python_out=.              \
  --grpc_python_out=.
