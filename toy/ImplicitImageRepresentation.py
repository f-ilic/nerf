import torch
import torch.nn as nn
from itertools import tee
from torch import nn, optim
from PIL import Image
from torchvision.transforms import PILToTensor, CenterCrop, Resize, Normalize
import matplotlib.pyplot as plt
import matplotlib as mpl
from einops import rearrange
import re
import matplotlib.animation as animation
import numpy as np

torch.pi = torch.acos(torch.zeros(1)).item() * 2

def block(in_neurons, out_neurons, activation_fn):
    return nn.Sequential(
        nn.Linear(in_neurons, out_neurons, bias=False),
        activation_fn()
    )

def sine_embedding(num_basis):
    def f(x):
        return torch.cat([torch.sin((2**i)*x*torch.pi) for i in range(num_basis)], dim=1)
    return f

def sine_cosine_embedding(num_basis):
    def f(x):
        s = torch.cat([torch.sin((2**i)*x*torch.pi) for i in range(num_basis//2)], dim=1)
        c = torch.cat([torch.cos((2**i)*x*torch.pi) for i in range(num_basis//2)], dim=1)
        return torch.cat([s,c], dim=1)
    return f

def replicate_inputs_sanity_check(num_basis):
    def f(x):
        return torch.cat([x for i in range(num_basis)], dim=1)
    return f

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def inv_normalize(mean, std):
    return Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])


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
    l = [
        (Resize(130)
        (CenterCrop(250)
        (PILToTensor()
        (Image.open(path).convert('RGB'))))) for path in image_paths]

    l = torch.cat(l, dim=1).float()
    return l
    

def main():
    criterion = nn.MSELoss().cuda()
    rgb_gt = get_imgs([
                    #    'toy/text.png',
                        # 'toy/bin.png',
                    #    'toy/markus.png',
                    #    'toy/lydia.png',
                       'toy/dominik.png',
                    # 'toy/lion.png'
                       ])

    mean = torch.mean(rgb_gt, dim=(1,2))
    std = torch.std(rgb_gt, dim=(1,2))
    rgb_gt = Normalize(mean, std)(rgb_gt) # 3, h, w

    embedding_dims = 10
    use_subset = False

    c, h, w = rgb_gt.shape
    rgb_gt = rearrange(rgb_gt, 'c h w -> (h w) c')

    y_grid, x_grid  = torch.meshgrid([torch.arange(w), torch.arange(h)])
    grid = torch.stack([y_grid/h, x_grid/w], dim=2)

    grid = rearrange(grid, 'h w c -> (h w) c').cuda()
    rgb_gt = rgb_gt.cuda()#/255.   
    print(f'Image has {rearrange(rgb_gt, "a c -> (a c)").size(0)} Values')

    layers = [2, 256, 256, 256, 256, 3]
    pelyer = [2*embedding_dims, 256, 256, 256, 256, 3]
    models = [
                MLP(nn.ReLU,    layers, name=f'ReLU',              preprocess_fn=nn.Identity()                                 ).cuda(), 
                # MLP(nn.Sigmoid, layers, name=f'Sigmoid',           preprocess_fn=nn.Identity()                                 ).cuda(),
                MLP(nn.Tanh,    layers, name=f'Tanh',              preprocess_fn=nn.Identity()                                 ).cuda(),
                MLP(nn.ReLU,    pelyer, name=f'ReLU pe sin',       preprocess_fn=sine_embedding(embedding_dims)                ).cuda(),
                MLP(nn.ReLU,    pelyer, name=f'ReLU pe sin+cos',   preprocess_fn=sine_cosine_embedding(embedding_dims)         ).cuda(),
                # MLP(nn.ReLU,    pelyer, name=f'ReLU sanity check', preprocess_fn=replicate_inputs_sanity_check(embedding_dims) ).cuda(),
            ]
    optimizers = [optim.AdamW(m.parameters(), lr=3e-3) for m in models]

    # print(f'Image has {rgb_gt.all.}')

    print("number of parameters:")
    for m in models:
        print(f'{m.name}: {sum(p.numel() for p in m.parameters() if p.requires_grad)}', end='\t')
    print("")

    
    fig, axes = plt.subplots(1,len(models)+2, figsize=(16,5))
    fig.tight_layout()
    
    
    [a.set_axis_off() for a in axes]
    axes = axes.ravel()
    inv_T = inv_normalize(mean, std)
    I = rgb_gt.cpu().reshape(h,w,c).permute(2,0,1)
    axes[0].imshow(inv_T(I).permute(1,2,0)/255.)

    im_axes = []
    for ax, m in zip(axes[2:].ravel(), models):
        im_axes.append(ax.imshow(np.zeros((h,w,c))))
        ax.set_title(m.name)

    
    epoch = 0
    batchsize = rgb_gt.size(0)
    perm = torch.randperm(batchsize)
    if use_subset:
        indices = perm[:batchsize//2]
        unused_indices = perm[batchsize//2:]
        rgb_gt[unused_indices] = 0
        grid_input = grid[indices]
        rgb_gt_input = rgb_gt[indices]
    else:
        grid_input = grid
        rgb_gt_input = rgb_gt


    I = rgb_gt.cpu().reshape(h,w,c).permute(2,0,1)
    axes[1].imshow(inv_T(I).permute(1,2,0)/255.)
    
    plt.show(block=False)

    while(True):
        for model, optims, im_ax in zip(models, optimizers, im_axes):
            rgb_predict = model(grid_input)
            loss = criterion(rgb_gt_input, rgb_predict)

            optims.zero_grad()
            loss.backward()
            optims.step()

            reco_image = model(grid).detach().cpu().reshape(h,w,c).clip(0,1).permute(2,0,1)
            reco_image = inv_T(reco_image).permute(1,2,0)/255.
            im_ax.set_data(reco_image)
            im_ax.axes.figure.canvas.draw()

            is_print = (epoch % 100 == 0)
            if is_print:
                print(f'{loss:.2E}\t', end='')

        if is_print:                
            print(f'Epoch: {epoch:.2E}')
        epoch += 1
        fig.canvas.draw()
        fig.canvas.flush_events()

    

if __name__ == "__main__":
    main()