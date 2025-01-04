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
B: batch size / batch diimension
T: horizon / sequence length
L: action feature dimension / low dimensional feature (low_dim in original)
O: observation feature dimension
F: conditioning feature dimension
D: embedding dimension
C: (image) channel dimension
H: (image) height
W: (image) width


"""
```
