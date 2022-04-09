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
### Reconstruction with fewer parameters than image
[1](toy/output_videos/smaller_net____100percent.mp4)
[2](toy/output_videos/smaller_net____050percent.mp4)
[3](toy/output_videos/smaller_net____010percent.mp4)
[4](toy/output_videos/tiny_net____010percent_leakyReLU.mp4)



# Resources


* [Yannic Kilcher](https://www.youtube.com/watch?v=CRlN-cYFxTk) - Original NeRF paper
* [Overview NeRF from one of the Authors (Barron)](https://www.youtube.com/watch?v=HfJpQCBTqZs)  - highly rewatchable good talk


# My summary slides
* [Problem statement, related problems, and original NeRF paper](https://googlelink)


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
