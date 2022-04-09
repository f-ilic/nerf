# Setup

## Anaconda first time setup
```
conda env create -f environment.yml
conda activate nerf
```

### Updating existing
```
conda env export > environment.yaml
conda env update --file environment.yaml --prune
```


## 2d toy problem:
imgs -> gif:
> ```cat *.png | ffmpeg -f image2pipe -stream_loop -1 -i - output.mp4```

from mp4 -> gif:
> ```for f in *.mp4; do ffmpeg -i $f -vf "fps=10,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 "${f%.mp4}.gif"; done```

### Reconstruction with fewer parameters than image
![1](toy/output_videos/smaller_net____100percent.gif)
![2](toy/output_videos/smaller_net____050percent.gif)
![3](toy/output_videos/smaller_net____010percent.gif)

LeakyReLU fixes saturation issues (see red images). The networks shown next have almost NO capacity:
`[20 24 24 24 3] -> 1800ish weights`. They have **10x** fewer parameters than all the networks above.
![4](toy/output_videos/tiny_net____010percent_leakyReLU.gif)
It seems to come at a cost of reconstruction fidelity however.
Comapare the following plos that shows ReLU vs LeakyRelU.

Another Relu vs leakyRelu comparison where capacity of network much higher.
![5](toy/output_videos/relu_vs_leakyrelu.gif)

### Learned embeddings
Also lets look at the embedding of 2->2 that is learned in both of these cases.

### In loss have coarse and fine network

# Resources


* [Yannic Kilcher](https://www.youtube.com/watch?v=CRlN-cYFxTk) - Original NeRF paper
* [Overview NeRF from one of the Authors (Barron)](https://www.youtube.com/watch?v=HfJpQCBTqZs)  - highly rewatchable good talk


# My NERF reading group slides
* [Slides1](https://files.icg.tugraz.at/f/8e753b931694420f9115/)


# Related literature
* [BlockNerf](https://arxiv.org/pdf/2202.05263.pdf)
* [Flame-In-Nerf](https://arxiv.org/pdf/2108.04913.pdf)

# githubs 

* https://github.com/ashawkey/torch-ngp instant ngp in pytorch reimpl.


# Matt positional encoding stuff


https://arxiv.org/abs/2108.07884

https://twitter.com/ducha_aiki/status/1428324571992711175

https://twitter.com/kleptid/status/1428345522797088772

https://arxiv.org/abs/2101.12322


# Literature
