import torch
import torch.nn as nn
from itertools import tee
from torch import nn, optim
from PIL import Image
from torchvision.transforms import PILToTensor, CenterCrop, Resize
import matplotlib.pyplot as plt
from einops import rearrange
import re

def block(in_neurons, out_neurons, activation_fn):
    return nn.Sequential(
        nn.Linear(in_neurons, out_neurons),
        activation_fn()
    )

def pos_encoding(x):
    return torch.cat([torch.sin(10*x),torch.sin(100*x),torch.sin(150*x), torch.sin(450*x), torch.sin(2*x)], dim=1)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class MLP(torch.nn.Module):
    def __init__(self, activation_fn, neurons, name, preprocess_fn) -> None:
        torch.manual_seed(69)
        super(MLP, self).__init__()
        self.activation_fn = activation_fn
        self.neurons = neurons
        self.name = name
        self.preprocess_fn = preprocess_fn
        self.net = nn.Sequential(*[block(i,o,self.activation_fn) for i,o in pairwise(self.neurons)])    

    def forward(self, x):
        return self.net(self.preprocess_fn(x))

def get_imgs(image_paths):
    l = [Resize(130)
        (CenterCrop(250)
        (PILToTensor()
        (Image.open(path).convert('RGB')))) for path in image_paths]
    return torch.cat(l, dim=1)

def main():
    criterion = nn.MSELoss().cuda()
    rgb_gt = get_imgs(['toy/lydia.png', 'toy/markus.png'])

    c, h, w = rgb_gt.shape
    rgb_gt = rearrange(rgb_gt, 'c h w -> (h w) c')

    y_grid, x_grid  = torch.meshgrid([torch.arange(w), torch.arange(h)])
    grid = torch.stack([y_grid/h, x_grid/w], dim=2)

    grid = rearrange(grid, 'h w c -> (h w) c').cuda()
    rgb_gt = rgb_gt.cuda()/255.   

    layers = [2, 256, 256, 256, 256, 3]
    pelyer = [10, 256, 256, 256, 256, 3]
    models = [
                MLP(nn.ReLU,    layers, name='ReLU',          preprocess_fn=nn.Identity()).cuda(), 
                MLP(nn.Sigmoid, layers, name='Sigmoid',       preprocess_fn=nn.Identity()).cuda(),
                MLP(nn.Tanh,    layers, name='Tanh',          preprocess_fn=nn.Identity()).cuda(),
                MLP(nn.ReLU,    pelyer, name='ReLU Pos.Embd', preprocess_fn=pos_encoding).cuda(),
            ]
    optimizers = [optim.AdamW(m.parameters(), lr=3e-3) for m in models]

    fig, axes = plt.subplots(1,len(models)+1)
    
    [a.set_axis_off() for a in axes]
    axes = axes.ravel()
    axes[0].imshow(rgb_gt.cpu().reshape(h,w,c))
    
    im_axes = []
    for ax, m in zip(axes[1:].ravel(), models):
        im_axes.append(ax.imshow(rgb_gt.cpu().reshape(h,w,c)))
        ax.set_title(m.name)

    plt.show(block=False)

    epoch = 0
    while(True):
        for model, optims, im_ax in zip(models, optimizers, im_axes):
            
            rgb_predict = model(grid)
            loss = criterion(rgb_gt, rgb_predict)

            optims.zero_grad()
            loss.backward()
            optims.step()

            im_ax.set_data(rgb_predict.detach().cpu().reshape(h,w,c).clip(0,1))
            im_ax.axes.figure.canvas.draw()

            is_print = (epoch % 100 == 0)
            if is_print:
                print(f'{loss:.2E}\t \t', end='')

        if is_print:                
            print(f'{epoch=}')
        epoch += 1

        plt.pause(0.000000000000001)

    

if __name__ == "__main__":
    main()