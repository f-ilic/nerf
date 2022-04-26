from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
from itertools import tee
from torch import nn, optim
from PIL import Image
from torchvision.transforms import PILToTensor, CenterCrop, Resize, Normalize, Compose
import matplotlib.pyplot as plt
import matplotlib as mpl
from einops import rearrange
import re
import matplotlib.animation as animation
import numpy as np

torch.pi = torch.acos(torch.zeros(1)).item() * 2

def block(in_neurons, out_neurons, activation_fn):
    return nn.Sequential(
        nn.Linear(in_neurons, out_neurons),
        activation_fn()
    )

def sine_embedding(num_basis):
    def f(x):
        return torch.cat([torch.sin((2**i)*x*torch.pi) for i in range(num_basis)], dim=1)
    return f

def sine_cosine_embedding(num_basis):
    def f(x):
        if num_basis % 2 == 1:
            raise ValueError("num_basis must be even")

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
    def __init__(self, activation_fn, neurons, name, embedding_fn) -> None:
        torch.manual_seed(69)
        super(MLP, self).__init__()
        self.activation_fn = activation_fn
        self.name = name
        self.neurons = neurons
        self.embedding_fn = embedding_fn
        self.net = nn.Sequential(*[block(i,o,self.activation_fn) for i,o in pairwise(self.neurons)]) 
        self.num_params =  sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        y = self.embedding_fn(x)
        x = self.net(y)
        return x, y

def get_imgs(image_paths):
    T = Compose([
                PILToTensor(),
                CenterCrop(550),
                Resize(120)
                ])
    l = [ T(Image.open(path).convert('RGB')) for path in image_paths ]
    return torch.cat(l, dim=1).float()
    

def main():
    criterion = nn.MSELoss().cuda()
    rgb_gt = get_imgs([
# 'toy/input_images/text.png',
# 'toy/input_images/bin.png',
# 'toy/input_images/lydia.png',
# 'toy/input_images/markus.png',
# 'toy/input_images/color.png',
# 'toy/input_images/dominik.png',
# 'toy/input_images/lion.png',
# 'toy/input_images/matador.png',
'toy/input_images/2dsine.png'
                       ])

    num_basis = 12
    embedding_dims = 2 * num_basis
    use_subset = True
    subset_percent = 0.5
    lmda = 0.5

    c, h, w = rgb_gt.shape

    y_grid, x_grid  = torch.meshgrid([torch.arange(w), torch.arange(h)])
    grid = torch.stack([x_grid/w, y_grid/h], dim=2)

    xylayer= [2, 256, 256, 256, 3]
    layers = [embedding_dims, 256, 256, 256, 3]
    pelyer = [embedding_dims, 256, 256, 256, 3]
 
    non_lin_embedding_neurons = [2, embedding_dims,embedding_dims,embedding_dims]
    non_lin_embedding = nn.Sequential(*[block(i,o,nn.LeakyReLU) for i,o in pairwise(non_lin_embedding_neurons)]) 
    models = [
                MLP(nn.ReLU,    layers, name=f'LeakyReLU learn embed', embedding_fn=non_lin_embedding        ).cuda(), 
            ]
    optimizers = [optim.AdamW(m.parameters(), lr=3e-3) for m in models]


    grid   = rearrange(grid, 'w h c -> (h w) c').cuda()
    rgb_gt = rearrange(rgb_gt, 'c h w -> (h w) c').cuda()/255.

    #  --- Train with whole data or with subset which is randomly sampled once.
    batchsize = rgb_gt.size(0)
    perm = torch.randperm(batchsize)
    if use_subset:
        indices, unused = perm[:int(batchsize*(subset_percent))], perm[int(batchsize*(subset_percent)):]
        rgb_gt_input = rgb_gt.clone()
        rgb_gt_input = rgb_gt_input[indices]
        grid_input = grid[indices]
        

        rgb_gt_input_displayable = rgb_gt.clone()
        rgb_gt_input_displayable[unused]=0
        rgb_gt_input_displayable = rgb_gt_input_displayable.reshape(h,w,c)
    else:
        grid_input = grid
        rgb_gt_input = rgb_gt
        rgb_gt_input_displayable = rgb_gt

    #  --- Sanity check plots of input data
    fig, axes = plt.subplots(1,len(models)+2, figsize=(16,5), sharex=True, sharey=True)    
    [a.set_axis_off() for a in axes]
    axes = axes.ravel()

    im_axes = []
    for ax, m in zip(axes[2:].ravel(), models):
        im_axes.append(ax.imshow(np.zeros((h,w,c))))
        ax.set_title(f'{m.name}\n{m.embedding_fn}\n{m.neurons}\n#Params: {m.num_params}', fontsize=8)

    axes[0].imshow(rgb_gt.cpu().reshape(h,w,c))
    axes[0].set_title(f'#Pixels: {rgb_gt.reshape(-1).size(0)}', fontsize=8)
    axes[1].imshow(rgb_gt_input_displayable.cpu().reshape(h,w,c))
    axes[1].set_title(f'#Pixels: {rgb_gt_input.reshape(-1).size(0)}', fontsize=8)

    fig.tight_layout()
    plt.show(block=False)


    print(f'Image has {rgb_gt.reshape(-1).size(0)} values')
    print("number of parameters:")
    for m in models:
        print(f'{m.name}: {sum(p.numel() for p in m.parameters() if p.requires_grad)}', end='\t')
    print("")

    epoch = 0
    fig.suptitle(f'Lambda: {lmda}')

    while(True):
        for model, optims, im_ax in zip(models, optimizers, im_axes):
            rgb_predict, basis = model(grid_input)
            rgb_loss = criterion(rgb_gt_input, rgb_predict)

            orthogonal_embedding_loss = torch.sum(basis[:,num_basis:] * basis[:,:num_basis], dim=1).mean()

            # orthogonal_embedding_loss = (1/2)*(basis.T * basis)**2

            loss = lmda     * rgb_loss + \
                   (1-lmda) * orthogonal_embedding_loss


            optims.zero_grad()
            loss.backward()
            optims.step()

            reco_image = model(grid)[0].detach().cpu().reshape(h,w,c).clip(0,1)
            im_ax.set_data(reco_image)
            im_ax.axes.figure.canvas.draw()

            is_print = (epoch % 100 == 0)
            if is_print:
                print(f'{loss:.2E}\t', end='')

        if is_print:                
            print(f'Epoch: {epoch:.2E}')

        # if epoch % 10 == 0:
        #     fig.savefig(f'toy/output/{epoch:05d}.png', format='png')
        #     if epoch==4090:
        #         break;

        epoch += 1
        fig.canvas.draw()
        fig.canvas.flush_events()

    

if __name__ == "__main__":
    main()