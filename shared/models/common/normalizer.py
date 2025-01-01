# The core functionality is from the Diffusion Policy repository
# I have simply re-written some of it in my own preferred way


import torch
import numpy as np

from typing import Union, Dict


class LinearNormalizer:
    def __init__(self, mode="limits", output_min=-1.0, output_max=1.0, range_eps=1e-4):
        """
        A simple linear normalizer for scaling data to a specified range.

        Args:
            mode (str): Normalization mode, "limits" (default) or "gaussian".
            output_min (float): Minimum value of the output range.
            output_max (float): Maximum value of the output range.
            range_eps (float): Small epsilon to handle zero range.
        """
        assert mode in [
            "limits",
            "gaussian",
        ], "Invalid mode: choose 'limits' or 'gaussian'"
        self.mode = mode
        self.output_min = output_min
        self.output_max = output_max
        self.range_eps = range_eps
        self.params = {}

    def fit(
        self,
        data: Union[
            torch.Tensor, np.ndarray, Dict[str, Union[torch.Tensor, np.ndarray]]
        ],
        **kwargs,
    ):
        """
        Fits the normalizer parameters to the data.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                self.params[key] = self._compute_params(value, **kwargs)
            # Ensure "_default" is always set and consistent
            if "_default" not in self.params or not self.params["_default"]:
                self.params["_default"] = self._compute_params(
                    next(iter(data.values())), **kwargs
                )
        else:
            # For non-dictionary data, directly compute and assign "_default"
            self.params["_default"] = self._compute_params(data, **kwargs)

    def normalize(
        self,
        data: Union[
            torch.Tensor, np.ndarray, Dict[str, Union[torch.Tensor, np.ndarray]]
        ],
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(data, dict):
            return {
                key: self._apply_normalization(value, self.params[key], forward=True)
                for key, value in data.items()
            }
        return self._apply_normalization(data, self.params["_default"], forward=True)

    def unnormalize(
        self,
        data: Union[
            torch.Tensor, np.ndarray, Dict[str, Union[torch.Tensor, np.ndarray]]
        ],
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(data, dict):
            return {
                key: self._apply_normalization(value, self.params[key], forward=False)
                for key, value in data.items()
            }
        return self._apply_normalization(data, self.params["_default"], forward=False)

    def _compute_params(self, data, fit_offset=True, last_n_dims=1):
        data = self._to_tensor(data)

        # Flatten last n_dims for per-element stats
        shape = data.shape
        flatten_dims = (-1,) if last_n_dims == 0 else (-1, *shape[-last_n_dims:])
        data = data.reshape(flatten_dims)

        input_min = data.min(dim=0).values
        input_max = data.max(dim=0).values
        input_mean = data.mean(dim=0)
        input_std = data.std(dim=0)

        if self.mode == "limits":
            scale = (self.output_max - self.output_min) / (
                input_max - input_min + self.range_eps
            )
            offset = (
                self.output_min - scale * input_min
                if fit_offset
                else torch.zeros_like(input_mean)
            )
        elif self.mode == "gaussian":
            scale = 1 / (input_std + self.range_eps)
            offset = -input_mean * scale if fit_offset else torch.zeros_like(input_mean)

        return {
            "scale": scale,
            "offset": offset,
            "input_stats": {
                "min": input_min,
                "max": input_max,
                "mean": input_mean,
                "std": input_std,
            },
        }

    def _apply_normalization(self, data, params, forward=True):
        data = self._to_tensor(data)
        scale = params["scale"]
        offset = params["offset"]

        shape = data.shape
        data = data.reshape(-1, *scale.shape)

        if forward:
            data = data * scale + offset
        else:
            data = (data - offset) / scale

        return data.reshape(shape)

    @staticmethod
    def _to_tensor(data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data

    def get_input_stats(self) -> Dict:
        return {key: params["input_stats"] for key, params in self.params.items()}


class SingleFieldLinearNormalizer:
    def __init__(self, mode="limits", output_min=-1.0, output_max=1.0, range_eps=1e-4):
        """
        A normalizer for a single field with configurable scaling and offset.

        Args:
            mode (str): Normalization mode, "limits" or "gaussian".
            output_min (float): Minimum value of the output range.
            output_max (float): Maximum value of the output range.
            range_eps (float): Small epsilon to handle zero range.
        """
        assert mode in [
            "limits",
            "gaussian",
        ], "Invalid mode: choose 'limits' or 'gaussian'"
        self.mode = mode
        self.output_min = output_min
        self.output_max = output_max
        self.range_eps = range_eps
        self.params = None

    def fit(
        self, data: Union[torch.Tensor, np.ndarray], last_n_dims=1, fit_offset=True
    ):
        """
        Compute and store the normalization parameters based on the data.

        Args:
            data (torch.Tensor or np.ndarray): Input data to fit the normalizer.
            last_n_dims (int): Number of dimensions to treat as feature dimensions.
            fit_offset (bool): Whether to compute and use an offset.
        """
        data = self._to_tensor(data)
        shape = data.shape
        flatten_dims = (-1,) if last_n_dims == 0 else (-1, *shape[-last_n_dims:])
        data = data.reshape(flatten_dims)

        input_min = data.min(dim=0).values
        input_max = data.max(dim=0).values
        input_mean = data.mean(dim=0)
        input_std = data.std(dim=0)

        if self.mode == "limits":
            scale = (self.output_max - self.output_min) / (
                input_max - input_min + self.range_eps
            )
            offset = (
                self.output_min - scale * input_min
                if fit_offset
                else torch.zeros_like(input_mean)
            )
        elif self.mode == "gaussian":
            scale = 1 / (input_std + self.range_eps)
            offset = -input_mean * scale if fit_offset else torch.zeros_like(input_mean)

        self.params = {
            "scale": scale,
            "offset": offset,
            "input_stats": {
                "min": input_min,
                "max": input_max,
                "mean": input_mean,
                "std": input_std,
            },
        }

    def normalize(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._apply_normalization(data, forward=True)

    def unnormalize(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._apply_normalization(data, forward=False)

    def get_input_stats(self) -> Dict:
        if self.params is None:
            raise RuntimeError(
                "Normalizer parameters not initialized. Call 'fit' first."
            )
        return self.params["input_stats"]

    def get_output_stats(self) -> Dict:
        if self.params is None:
            raise RuntimeError(
                "Normalizer parameters not initialized. Call 'fit' first."
            )
        return {
            key: self.normalize(value.unsqueeze(0)).squeeze(0)
            for key, value in self.params["input_stats"].items()
        }

    def _apply_normalization(self, data, forward=True):
        if self.params is None:
            raise RuntimeError(
                "Normalizer parameters not initialized. Call 'fit' first."
            )
        data = self._to_tensor(data)
        scale = self.params["scale"]
        offset = self.params["offset"]

        shape = data.shape
        data = data.reshape(-1, *scale.shape)

        if forward:
            data = data * scale + offset
        else:
            data = (data - offset) / scale

        return data.reshape(shape)

    @staticmethod
    def _to_tensor(data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data


# ====== TEST FUNCTION ======
# These tests are largely taken from the Diffusion Policy repository
# However, the precision is slightly lower... not sure why?
def test_linear_normalizer():
    # Test 1: Basic normalization
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    data[..., 0, 0] = 0

    normalizer = LinearNormalizer(mode="limits")
    normalizer.fit(data)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.isclose(datan.max().item(), 1.0, atol=1e-3)
    assert np.isclose(datan.min().item(), -1.0, atol=1e-3)

    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    # Test 2: Gaussian normalization
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    normalizer = LinearNormalizer(mode="gaussian")
    normalizer.fit(data)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.isclose(datan.mean().item(), 0.0, atol=1e-3)
    assert np.isclose(datan.std().item(), 1.0, atol=1e-3)

    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-6)

    # Test 3: Dictionary normalization
    data = {
        "obs": torch.zeros((1000, 128, 9, 2)).uniform_() * 512,
        "action": torch.zeros((1000, 128, 2)).uniform_() * 512,
    }
    normalizer = LinearNormalizer()
    normalizer.fit(data)
    datan = normalizer.normalize(data)
    dataun = normalizer.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-5)

    print("LinearNormalizer tests passed!")


def test_single_field_linear_normalizer():
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    data[..., 0, 0] = 0

    normalizer = SingleFieldLinearNormalizer(mode="limits")
    normalizer.fit(data, last_n_dims=2)

    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.isclose(datan.max().item(), 1.0, atol=1e-3)
    assert np.isclose(datan.min().item(), -1.0, atol=1e-3)

    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    print("SingleFieldLinearNormalizer tests passed!")


if __name__ == "__main__":
    test_linear_normalizer()
    test_single_field_linear_normalizer()
    print("All tests passed!")
