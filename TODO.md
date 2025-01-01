# TODO

IMPORTANT:
- Make all the dimension keys consistent, update README once and use
  consistently
- Update README with explanation; add explanations and consistency for
  dictionaries and other definitions
- Move tests to a tests folder and actually use a testing framework
  - What's the point of these? Maintain consistency among changes?
- ~~Make weights folder consistent whenever used~~
  - Update and clean encoders files, create default resnet option
    - For training, find a good set of hyperparameters; get full
      IMAGENET
          - Update to FSDP instead of DDP?
          - wandb and omegaconf?
          - Find a unified approach to encoders section
          - ~~Add crop randomization~~ and make it easy to create a pipeline
- JAXtyping, type annotations, type hints, assertions, documentation
  (params and args) for every important function, etc
- Check what dimensions work for the unet, maybe re-write so it can accept (relatively) arbitrary dimensions
- Test local_cond_dim on image policy
- Make sure to credit code properly. For the code you credit, review
  what's needed and potentially minimize.

(from readme)
TODO:
- clean up and update the visual encoders section
  - shape annotations, good hyperparameters for the training script,
    easy logging, and more
- add shape annotations to conv1d_components; clean up dimension key
  and shape testing
