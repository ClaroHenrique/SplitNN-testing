import pickle
import torch

t = torch.rand((2,2))
st = pickle.dumps(t)
st

dt = pickle.loads(st)
t
dt

type(st)
