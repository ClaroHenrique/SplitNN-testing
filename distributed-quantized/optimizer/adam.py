 
import torch.optim as optim

def create_optimizer(params, learning_rate):
    optimizer = optim.Adam(params, lr=learning_rate)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0) # No learning rate decay
    return optimizer, scheduler
