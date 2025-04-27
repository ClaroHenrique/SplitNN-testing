 
import torch.optim as optim

def create_optimizer(params, learning_rate):
    optimizer = optim.Adam(params, lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    return optimizer, scheduler
