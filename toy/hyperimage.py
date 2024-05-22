from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
from itertools import tee
from torch import nn, optim
from PIL import Image
from torchvision.transforms import (
    PILToTensor,
    CenterCrop,
    Resize,
    Normalize,
    Compose,
    GaussianBlur,
)
import matplotlib.pyplot as plt
import matplotlib as mpl
from einops import rearrange
import re
import matplotlib.animation as animation
import numpy as np
import torch
from hypnettorch.mnets import MLP
from hypnettorch.hnets import HMLP

torch.pi = torch.acos(torch.zeros(1)).item() * 2
device = torch.device("mps")


def sine_embedding(num_basis):
    def f(x):
        return torch.cat(
            [torch.sin((2**i) * x * torch.pi) for i in range(num_basis)], dim=1
        )

    return f


def sine_cosine_embedding(num_basis):
    def f(x):
        if num_basis % 2 == 1:
            raise ValueError("num_basis must be even")

        s = torch.cat(
            [torch.sin((2**i) * x * torch.pi) for i in range(num_basis // 2)], dim=1
        )
        c = torch.cat(
            [torch.cos((2**i) * x * torch.pi) for i in range(num_basis // 2)], dim=1
        )
        return torch.cat([s, c], dim=1)

    return f


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_imgs(image_paths):
    T = Compose(
        [
            PILToTensor(),
            # CenterCrop(550),
            Resize(64),
            GaussianBlur(kernel_size=3, sigma=1),
        ]
    )
    l = [T(Image.open(path).convert("RGB")) for path in image_paths]
    return torch.cat(l, dim=1).float()


def main():
    criterion = nn.MSELoss().to(device)
    rgb_gt = get_imgs(
        [
            #    'toy/input_images//text.png',
            # 'toy/input_images/bin.png',
            #    'toy/input_images/lydia.png',
            #    'toy/input_images/markus.png',
            # 'toy/input_images/color.png'
            #    'toy/input_images/dominik.png',
            "toy/input_images/lion.png"
            # "toy/input_images/matador.png"
            # "toy/input_images/2dsine.png"
            # "ipn/1CM1_1_R_#218_000036.jpg"
        ]
    )
    c, h, w = rgb_gt.shape

    y_grid, x_grid = torch.meshgrid([torch.arange(w), torch.arange(h)])
    grid = torch.stack([x_grid / w, y_grid / h], dim=2)

    grid = rearrange(grid, "w h c -> (h w) c").to(device)
    rgb_gt = rearrange(rgb_gt, "c h w -> (h w) c").to(device) / 255.0

    # num_basis = 12
    # emb = sine_cosine_embedding(num_basis)
    model = MLP(
        2,
        n_out=3,
        no_weights=True,
        hidden_layers=(4, 4),
    ).to(device)
    hnet = HMLP(model.param_shapes).to(device)
    optimizer = optim.Adam(hnet.parameters(), lr=1e-3)
    epoch = 0
    weights = hnet.forward(cond_id=0)
    tmp = model.forward(grid, weights=weights)
    fig, ax = plt.subplots(1, 2)
    ax = ax.flatten()
    ax[0].imshow(rgb_gt.cpu().reshape(h, w, c).clip(0, 1).detach().numpy())
    im_ax = ax[1].imshow(rgb_gt.cpu().reshape(h, w, c).clip(0, 1).detach().numpy())
    plt.show(block=False)
    # plt.show()

    while True:
        weights = hnet.forward(cond_id=0)
        rgb_predict = model.forward(grid, weights=weights)

        loss = criterion(rgb_gt, rgb_predict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reco_image = rgb_predict.cpu().reshape(h, w, c).clip(0, 1).detach().numpy()

        im_ax.set_data(reco_image)
        im_ax.axes.figure.canvas.draw()

        fig.canvas.draw()
        fig.canvas.flush_events()
        # is_print = epoch % 100 == 0
        # if is_print:
        # print(f"{loss:.2E}\t", end="")
        print(f"Epoch: {epoch:07d}")
        epoch += 1

    if is_print:
        print(f"Epoch: {epoch:.2E}")

    # if epoch % 10 == 0:
    #     fig.savefig(f'toy/output/{epoch:05d}.png', format='png')
    #     if epoch==4090:
    #         break;

    epoch += 1
    fig.canvas.draw()
    fig.canvas.flush_events()


if __name__ == "__main__":
    main()
