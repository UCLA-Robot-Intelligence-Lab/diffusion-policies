# diffusion-policies
Diffusion Policy and other generative model policies
## Getting Started
At the root level, first create the environment:
```console
conda env create -f environment.yaml
```
and activate it:
```console
conda activate diffpolicy
```
Run setup:
```console
python setup.py && pip install -e .
```
For training data, you can download training data from the original
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
repository. Below we use PushT:
```console
cd data && wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip && unzip pusht.zip && rm -f pusht.zip && cd ..
```
Navigate to `diffusion_policy/config`, select the config for
architecture and modality you want to train for, and change
the config as necessary (i.e., task). Make sure to `wandb login` if
you have not already. Then, navigate back up to the
network architecture folder that you chose and run the
training script, for example
```console
python train_unet_image_policy.py
```

### Credit
The implementations here are not necessarily original. Please reference the original repositories (linked below) literature (cited at the bottom)!
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy): see the `README.md` under `diffusion_policy/` for more details on the practical implementation in this repository.

### Shape Suffixes and Dimension  Key
This repository annotates tensors with shape suffixes, which simply indicate what you would expect to receive after calling `.shape` on the tensor. To read about why, see [this blog post](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd) from Noam Shazeer. In this repository, we use the following dimension key:
```python
"""
Dimension key:

B: batch size
T: prediction horizon
    To: observation horizon
    Ta: action horizon
F: feature dimension
    Fo: observation feature dimension
    Fa: action feature dimension
D: embedding dimension
C: (generic) conditioning dimension, sometimes channel dimension
G: global conditioning dimension
L: local conditioning dimension
I: (conv, generic) input channel dimension
O: (conv, generic) output channel dimension
H: (image) height
W: (image) width

Tensors are denoted with brackets [ ], i.e., [ B, T, L ] and suffixed,
i.e., x_BOT. If we are just using the variable as a dimension (int),
no brackets are present, and the name will additionally be suffixed
with _dim_, i.e., inp_dim_F.

To read tensors, i.e., cond_BTL, apply the last dimension as a prefix
when applicable (L, G). Our example would read "local conditioning
tensor of shape (B, T, L)". Otherwise, just read the tensor directly,
adding the shape as a suffix.
"""
```

### Citations
```bibtex
@inproceedings{chi2023diffusionpolicy,
	title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
	author={Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
	booktitle={Proceedings of Robotics: Science and Systems (RSS)},
	year={2023}
}
```
