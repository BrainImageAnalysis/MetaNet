import torch

def mydice(seg, gt):
    alpha = 0.00001
    
    dice_score = torch.sum(seg[gt==1.])*2. / (torch.sum(seg) + torch.sum(gt) + alpha)
    
    #print(dice_score)
    return dice_score