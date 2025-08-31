 
import torch.optim as optim

def create_optimizer(params, learning_rate):
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0) # No learning rate decay
    return optimizer, scheduler

