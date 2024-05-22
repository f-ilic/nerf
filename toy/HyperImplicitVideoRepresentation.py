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
from torchvision.utils import make_grid
from glob import glob
from hypnettorch.mnets import MLP
from hypnettorch.hnets import HMLP

torch.pi = torch.acos(torch.zeros(1)).item() * 2
device = torch.device("mps")


def sine_embedding(num_basis):
    def f(x):
        return torch.cat(
            [torch.sin((2**i) * x * torch.pi) for i in range(num_basis)],
            dim=1,
        )

    return f


def sine_cosine_embedding(num_basis):
    def f(x):
        if num_basis % 2 == 1:
            raise ValueError("num_basis must be even")

        s = torch.cat(
            [torch.sin((2**i) * x * torch.pi) for i in range(num_basis // 3)], dim=1
        )
        c = torch.cat(
            [torch.cos((2**i) * x * torch.pi) for i in range(num_basis // 3)], dim=1
        )
        return torch.cat([s, c], dim=1)

    return f


def get_video(image_paths):
    T = Compose([PILToTensor(), Resize(64), GaussianBlur(kernel_size=3, sigma=1)])
    l = [T(Image.open(path).convert("RGB")) for path in image_paths]
    return torch.stack(l, dim=0).float()


def video2grid(video):  # T C H W -> C H W*T
    return make_grid(video, padding=0)


def psnr(x, y):
    return torch.mean(20 * torch.log10(1.0 / torch.sqrt(torch.mean((x - y) ** 2))))


def main():
    criterion = nn.MSELoss().to(device)
    rgb_gt = get_video(sorted(glob("ipn/*.jpg")))

    num_basis = 12
    t, c, h, w = rgb_gt.shape

    y_grid, x_grid, t_grid = torch.meshgrid(
        [torch.arange(w), torch.arange(h), torch.arange(t)],
        indexing="ij",
    )

    grid = torch.stack(
        [
            x_grid / w,
            y_grid / h,
            t_grid / t,
        ],
        dim=2,
    )

    grid = rearrange(grid, "w h c t-> (h w t) c").to(device)
    rgb_gt = rearrange(rgb_gt, "t c h w -> (h w t) c").to(device) / 255.0

    batchsize = rgb_gt.size(0)
    perm = torch.randperm(batchsize)
    use_subset = True
    subset_percent = 0.001

    if use_subset:
        indices, unused = (
            perm[: int(batchsize * (subset_percent))],
            perm[int(batchsize * (subset_percent)) :],
        )
        rgb_gt_input = rgb_gt.clone()
        rgb_gt_input = rgb_gt_input[indices]
        grid_input = grid[indices]

        rgb_gt_input_displayable = rgb_gt.clone()
        rgb_gt_input_displayable[unused] = 0
    else:
        grid_input = grid
        rgb_gt_input = rgb_gt
        rgb_gt_input_displayable = rgb_gt

    model = MLP(
        n_in=3,  # num_basis * 2,
        n_out=3,
        no_weights=True,
        hidden_layers=(10, 10),
    ).to(device)
    hnet = HMLP(model.param_shapes).to(device)
    optimizer = optim.Adam(hnet.parameters(), lr=1e-3)

    weights = hnet.forward(cond_id=0)
    # tmp = model.forward(grid, weights=weights)
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    ax = ax.flatten()

    # tmp = rearrange(rgb_gt, "(h w t) c -> h (t w) c", h=h, w=w, t=t, c=c).cpu()
    input_plotable = rearrange(
        rgb_gt_input_displayable,
        "(h w t) c -> h (t w) c",
        h=h,
        w=w,
        t=t,
        c=c,
    )
    ax[0].imshow(input_plotable.cpu().clip(0, 1).detach().numpy())
    im_ax = ax[1].imshow(input_plotable.cpu().clip(0, 1).detach().numpy())
    plt.show(block=False)

    epoch = 0
    emb = lambda x: x  # sine_cosine_embedding(num_basis)
    while True:
        weights = hnet.forward(cond_id=0)
        rgb_predict = model.forward(emb(rgb_gt_input), weights=weights)
        print(rgb_gt_input.shape)

        loss = criterion(rgb_gt_input, rgb_predict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # full grid
        full = model.forward(emb(rgb_gt), weights=weights)
        reco_image = rearrange(
            full,
            "(h w t) c -> h (t w) c",
            h=h,
            t=t,
            c=c,
        )

        im_ax.set_data(reco_image.detach().cpu().clip(0, 1).numpy())
        im_ax.axes.figure.canvas.draw()

        fig.canvas.draw()
        fig.canvas.flush_events()
        print(f"Epoch: {epoch:07d}")

        epoch += 1


if __name__ == "__main__":
    main()
