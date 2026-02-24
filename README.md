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
pip install -e .
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
## Training
To train, first create the environment:
```console
uv venv
```
and activate it:
```console
source .venv/bin/activate
```
then install requirements:
```console
uv pip install -r requirements-train.txt
```
make sure wandb is enabled:
```console
wandb login
```
you can try to train with the following command, for example:
```console
CUDA_VISIBLE_DEVICES=1 \
python diffusion_policy/unet/train_unet_image_policy.py \
  --config-name=train_unet_image_merlin_policy \
  logging.mode=online
```
change the device, and config accordingly. You can also override the Hydra config with specific parameters by adding more arguments (e.g. `logging.mode=online` here).

### Credit
The implementations here are not necessarily original. Please reference the original repositories (linked below) literature (cited at the bottom)!
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy): see the `README.md` under `diffusion_policy/` for more details on the practical implementation in this repository.
- [Vision in Action](vision-in-action.github.io): ViT backbone

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
@misc{xiong2025visionactionlearningactive,
      title={Vision in Action: Learning Active Perception from Human Demonstrations}, 
      author={Haoyu Xiong and Xiaomeng Xu and Jimmy Wu and Yifan Hou and Jeannette Bohg and Shuran Song},
      year={2025},
      eprint={2506.15666},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.15666}, 
}
```

