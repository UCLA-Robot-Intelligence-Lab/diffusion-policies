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
C: conditioning (observation) dimension
G: global conditioning dimension
L: local conditioning dimension
I: (conv) input channel dimension
O: (conv) output channel dimension
D: embedding dimension
H: (image) height
W: (image) width
"""
```
