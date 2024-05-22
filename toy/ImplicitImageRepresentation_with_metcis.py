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
from kornia.losses import ssim_loss

torch.pi = torch.acos(torch.zeros(1)).item() * 2


def block(in_neurons, out_neurons, activation_fn):
    return nn.Sequential(nn.Linear(in_neurons, out_neurons), activation_fn())


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
        self.net = nn.Sequential(
            *[block(i, o, self.activation_fn) for i, o in pairwise(self.neurons)]
        )
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.embedding_fn(x)
        x = self.net(x)
        return x


def get_imgs(image_paths):
    T = Compose(
        [
            PILToTensor(),
            # CenterCrop(250),
            Resize(120),
        ]
    )
    l = [T(Image.open(path).convert("RGB")).float() / 255.0 for path in image_paths]
    return torch.cat(l, dim=1)


def psnr(x, y):
    return torch.mean(20 * torch.log10(1.0 / torch.sqrt(torch.mean((x - y) ** 2))))


def main():
    criterion = nn.MSELoss().cuda()
    rgb_gt = get_imgs(
        [
            # 'toy/input_images/text.png',
            # 'toy/input_images/bin.png',
            # 'toy/input_images/lydia.png',
            # 'toy/input_images/markus.png',
            # 'toy/input_images/color.png',
            # 'toy/input_images/dominik.png',
            "toy/input_images/lion.png"
            # 'toy/input_images/matador.png'
            # 'toy/input_images/2dsine.png'
        ]
    )
    num_basis = 12
    embedding_dims = 2 * num_basis
    use_subset = False
    subset_percent = 0.5

    c, h, w = rgb_gt.shape

    y_grid, x_grid = torch.meshgrid([torch.arange(w), torch.arange(h)])
    grid = torch.stack([x_grid / w, y_grid / h], dim=2)

    xylayer = [2, 64, 64, 64, 64, 3]
    layers = [embedding_dims, 64, 64, 64, 64, 3]

    models = [
        MLP(
            nn.Tanh,
            layers,
            name=f"Tanh",
            embedding_fn=block(2, embedding_dims, nn.Tanh),
        ).cuda(),
        MLP(
            nn.LeakyReLU, xylayer, name=f"LeakyReLU", embedding_fn=nn.Identity()
        ).cuda(),
        # MLP(nn.ReLU,    xylayer, name=f'ReLU pure', embedding_fn=nn.Identity()            ).cuda(),
        # MLP(nn.ReLU, layers, name=f'ReLU learn embed', embedding_fn=block(2, embedding_dims, nn.ReLU)).cuda(),
        # MLP(nn.LeakyReLU,    xylayer, name=f'LeakyReLU pure', embedding_fn=nn.Identity()            ).cuda(),
        # MLP(nn.LeakyReLU,    layers, name=f'LeakyReLU learn embed', embedding_fn=block(2, embedding_dims, nn.LeakyReLU)            ).cuda(),
        # MLP(nn.Sigmoid, layers, name=f'Sigmoid',              embedding_fn=block(2, embedding_dims, nn.Sigmoid)         ).cuda(),
        # MLP(nn.LeakyReLU,    layers, name=f'LeakyReLU only xy coords',  embedding_fn=replicate_inputs_sanity_check(num_basis)).cuda(),
        # MLP(nn.LeakyReLU,    layers, name=f'LeakyReLU pe sin', embedding_fn=sine_embedding(num_basis)).cuda(),
        MLP(
            nn.LeakyReLU,
            layers,
            name=f"LeakyReLU pe sin+cos",
            embedding_fn=sine_cosine_embedding(num_basis),
        ).cuda(),
    ]

    optimizers = [optim.AdamW(m.parameters(), lr=3e-3) for m in models]
    # optimizers = [optim.SGD(m.parameters(), lr=3e-2) for m in models]

    grid = rearrange(grid, "w h c -> (h w) c").cuda()
    rgb_gt = rearrange(rgb_gt, "c h w -> (h w) c").cuda()

    #  --- Train with whole data or with subset which is randomly sampled once.
    batchsize = rgb_gt.size(0)
    perm = torch.randperm(batchsize)
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
        rgb_gt_input_displayable = rgb_gt_input_displayable.reshape(h, w, c)
    else:
        grid_input = grid
        rgb_gt_input = rgb_gt
        rgb_gt_input_displayable = rgb_gt

    #  --- Sanity check plots of input data
    metric_fig, metric_ax = plt.subplots(1, 3, figsize=(16, 4))
    metric_ax = metric_ax.ravel()
    loss_plot = metric_ax[0].plot(
        np.empty((0, len(models))),
        np.empty((0, len(models))),
        label=[m.name for m in models],
    )
    psnr_plot = metric_ax[1].plot(
        np.empty((0, len(models))),
        np.empty((0, len(models))),
        label=[m.name for m in models],
    )
    ssim_plot = metric_ax[2].plot(
        np.empty((0, len(models))),
        np.empty((0, len(models))),
        label=[m.name for m in models],
    )

    metric_ax[0].set_title("Loss")
    metric_ax[0].set_yscale("log")
    metric_ax[1].set_title("PSNR")
    metric_ax[2].set_title("SSIM")
    [ma.legend(loc="best") for ma in metric_ax]

    fig, axes = plt.subplots(
        1, len(models) + 2, figsize=(16, 5), sharex=True, sharey=True
    )
    [a.set_axis_off() for a in axes]
    axes = axes.ravel()

    im_axes = []
    for ax, m in zip(axes[2:].ravel(), models):
        im_axes.append(ax.imshow(np.zeros((h, w, c))))
        ax.set_title(f"{m.name}\n{m.neurons}\n#Params: {m.num_params}", fontsize=8)

    axes[0].imshow(rgb_gt.cpu().reshape(h, w, c))
    axes[0].set_title(f"#Pixels: {rgb_gt.reshape(-1).size(0)}", fontsize=8)
    axes[1].imshow(rgb_gt_input_displayable.cpu().reshape(h, w, c))
    axes[1].set_title(f"#Pixels: {rgb_gt_input.reshape(-1).size(0)}", fontsize=8)

    fig.tight_layout()
    [ma.grid(True, which="both", ls="-", alpha=0.3) for ma in metric_ax]

    plt.show(block=False)

    print(f"Image has {rgb_gt.reshape(-1).size(0)} values")
    print("number of parameters:")
    for m in models:
        print(
            f"{m.name}: {sum(p.numel() for p in m.parameters() if p.requires_grad)}",
            end="\t",
        )
    print("")

    lines = {m.name: {i: [] for i in ["epoch", "loss", "psnr", "ssim"]} for m in models}
    epoch = 0
    while True:
        for idx, (model, optims, im_ax) in enumerate(zip(models, optimizers, im_axes)):
            rgb_predict = model(grid_input)
            loss = criterion(rgb_gt_input, rgb_predict)

            optims.zero_grad()
            loss.backward()
            optims.step()

            reco_image = model(grid).detach().cpu().reshape(h, w, c).clip(0, 1)
            im_ax.set_data(reco_image)
            im_ax.axes.figure.canvas.draw()

            is_print = epoch % 100 == 0
            if is_print:
                print(f"{loss:.2E}\t", end="")

            # fig, ax = plt.subplots(1,2)
            # ax[0].imshow(rgb_gt.reshape(h,w,c).cpu())
            # ax[1].imshow(reco_image)
            # plt.show()
            peak_signal_to_noise = psnr(rgb_gt.reshape(h, w, c).cpu(), reco_image)
            structural_sim = ssim_loss(
                rgb_gt.reshape(h, w, c).cpu().permute(2, 0, 1).unsqueeze(0),
                reco_image.permute(2, 0, 1).unsqueeze(0),
                window_size=7,
            )
            lines[model.name]["epoch"].append(epoch)
            lines[model.name]["loss"].append(loss.item())
            lines[model.name]["psnr"].append(peak_signal_to_noise.item())
            lines[model.name]["ssim"].append(1 - structural_sim.item())

            loss_plot[idx].set_data(
                lines[model.name]["epoch"], lines[model.name]["loss"]
            )
            psnr_plot[idx].set_data(
                lines[model.name]["epoch"], lines[model.name]["psnr"]
            )
            ssim_plot[idx].set_data(
                lines[model.name]["epoch"], lines[model.name]["ssim"]
            )

        for ma in metric_ax:
            ma.relim()
            ma.autoscale_view()

        if is_print:
            print(f"Epoch: {epoch:.2E}")

        fig.canvas.draw()
        fig.canvas.flush_events()

        metric_fig.canvas.draw()
        metric_fig.canvas.flush_events()

        if epoch % 15 == 0:
            fig.savefig(f"toy/output/lion/imgs/{epoch:06d}.png", format="png")
            metric_fig.savefig(f"toy/output/lion/metrics/{epoch:06d}.png", format="png")
            if epoch == 50000:
                break
        epoch += 1


if __name__ == "__main__":
    main()
