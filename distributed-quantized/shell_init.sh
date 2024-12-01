
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip3 install dontenv
pip3 install torch
pip3 install torchvision
pip3 install grpcio
pip3 install --upgrade google-api-python-client

source .venv/bin/activate
python3 sfl_client.py --client_id=0
python3 sfl_client.py --client_id=1
python3 sfl_server.py

