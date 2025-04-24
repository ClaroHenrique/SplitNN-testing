
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
pip3 install python-dotenv
pip3 install torch
pip3 install torchvision
pip3 install grpcio grpcio-tools
pip3 install --upgrade google-api-python-client
pip3 install flwr-datasets[vision]

source .venv/bin/activate
python3 sfl_client.py --client_id=0
python3 sfl_client.py --client_id=1
python3 sfl_server.py

