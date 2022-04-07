import torch
import torch.nn as nn
from itertools import tee
from torch import nn, optim
from PIL import Image
from torchvision.transforms import PILToTensor
import matplotlib.pyplot as plt
from einops import rearrange

def block(in_neurons, out_neurons, activation_fn):
    return nn.Sequential(
        nn.Linear(in_neurons, out_neurons),
        activation_fn()
    )

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class MLP(torch.nn.Module):
    def __init__(self, activation_fn, neurons) -> None:
        super(MLP, self).__init__()
        self.activation_fn = activation_fn
        self.neurons = neurons
        self.net = nn.Sequential(*[block(i,o,self.activation_fn) for i,o in pairwise(self.neurons)])    

    def forward(self, x):
        return self.net(x)


def main():
    num_epochs = 1000
    layers = [2, 256, 256, 256, 256, 256, 3]
    model = MLP(nn.ReLU, layers)
    model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss().cuda()

    rgb_gt = PILToTensor()(Image.open('lydia.jpeg'))
    plt.imshow(rgb_gt.permute(1,2,0))
    plt.axis('off')
    plt.show()
    for epoch in range(num_epochs):
        print(f"{epoch=}")
        rgb_gt = rgb_gt.cuda()
        rgb_predict = model(rgb_gt)
        loss = criterion(rgb_gt, rgb_predict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

if __name__ == "__main__":
    main()