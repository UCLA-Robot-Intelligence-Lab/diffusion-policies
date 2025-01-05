# diffusion-policies
Diffusion Policy and other generative model policies

### Credit
The implementations here are not necessarily original. Please reference the original repositories (linked below) literature (cited at the bottom)!
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)

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
C: conditioning (observation) dimension
G: global conditioning dimension
L: local conditioning dimension
I: (conv, generic) input channel dimension
O: (conv, generic) output channel dimension
H: (image) height
W: (image) width

Tensors are denoted with brackets [ ], i.e., [ B, T, L ].
If we are just using the variable as a dimension (int),
no brackets are present.
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
