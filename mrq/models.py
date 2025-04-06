#rough outline 
''' 
Define neural net archs AND losses! Defining losses is different than MrQ implementationo
Encoder
Policy
Value
Standalone functions for computing loss 
Agent class? 
'''
import torch 
import torch.nn as nn
import torch.nn.functional as F 

class StateEncoder(nn.Module): 
    ''' 
    f(s) --> z_(s)
    convert raw state obs into a latent embedding 
    '''

