 
import torch.optim as optim

def create_optimizer(params, learning_rate):
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    return optimizer, scheduler

