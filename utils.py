# from debug.debugger import *

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
    
    #@profile_memory_and_time
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    #@profile_memory_and_time
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
