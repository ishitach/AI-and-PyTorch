import torch

def activation(x):
    return 1/(1+torch.exp(-x))
    
torch.manual_seed(7)
features = torch.randn((1, 3))
n_input = features.shape[1]
n_hidden =2
n_output = 1
W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))


first = activation(torch.mm(features, W1)+B1)
print(activation(torch.mm(first,W2)+B2))
