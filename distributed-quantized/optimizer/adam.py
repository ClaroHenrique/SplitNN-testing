 
import torch.optim as optim

def create_optimizer(params, learning_rate):
    return optim.Adam(params, lr=learning_rate)