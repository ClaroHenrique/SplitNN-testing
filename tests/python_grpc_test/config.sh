apt install python3.12-venv
apt install pip3

python3 -m pip install --upgrade pip
python3 -m venv .venv

source .venv/bin/activate
pip3 install grpcio grpcio-tools
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. greeter.proto
