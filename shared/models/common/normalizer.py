import unittest
import zarr
import numpy as np
import torch
import torch.nn as nn

from typing import Union, Dict
from shared.utils.pytorch_util import dict_apply


class LinearNormalizer(nn.Module):
    available_modes = ["limits", "gaussian"]

    def __init__(self, params_dict: nn.ParameterDict = None):
        """
        Initializes the LinearNormalizer.

        args:
            params_dict (nn.ParameterDict, optional): A ParameterDict containing
            normalization parameters. If None, an empty ParameterDict is initialized.
        """
        super().__init__()
        if params_dict is None:
            params_dict = nn.ParameterDict()
        self.params_dict = params_dict

    @property
    def device(self):
        """
        returns the device of the first parameter in the ParameterDict.
        """
        return next(iter(self.parameters())).device

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Custom method to load parameters from a state_dict.

        args:
            state_dict (dict): State dictionary containing model parameters.
            prefix (str): Prefix for parameter keys.
            local_metadata (dict): Metadata for loading.
            strict (bool): Whether to strictly enforce parameter matching.
            missing_keys (list): List of missing keys.
            unexpected_keys (list): List of unexpected keys.
            error_msgs (list): List of error messages.
        """

        def dfs_add(dest, keys, value: torch.Tensor):
            if len(keys) == 1:
                dest[keys[0]] = value
                return

            if keys[0] not in dest:
                dest[keys[0]] = nn.ParameterDict()
            dfs_add(dest[keys[0]], keys[1:], value.clone())

        def load_dict(state_dict, prefix):
            out_dict = nn.ParameterDict()
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    param_keys = key[len(prefix) :].split(".")[1:]
                    dfs_add(out_dict, param_keys, value.clone())
            return out_dict

        self.params_dict = load_dict(state_dict, prefix + "params_dict")
        self.params_dict.requires_grad_(False)
        return

    @torch.no_grad()
    def fit(
        self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims: int = 1,
        dtype: torch.dtype = torch.float32,
        mode: str = "limits",
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
    ):
        """
        Fits the normalizer parameters to the provided data.

        args:
            data (Union[Dict, torch.Tensor, np.ndarray, zarr.Array]): Input data to fit.
            last_n_dims (int, optional): Number of dimensions to consider as features.
            dtype (torch.dtype, optional): Data type for computations.
            mode (str, optional): Normalization mode ("limits" or "gaussian").
            output_max (float, optional): Maximum output value for normalization.
            output_min (float, optional): Minimum output value for normalization.
            range_eps (float, optional): Epsilon value to avoid division by zero.
            fit_offset (bool, optional): Whether to fit an offset.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] = _fit(
                    value,
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset,
                )
        else:
            self.params_dict["_default"] = _fit(
                data,
                last_n_dims=last_n_dims,
                dtype=dtype,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset,
            )

    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Allows the normalizer to be called as a function to normalize data.

        args:
            x (Union[Dict, torch.Tensor, np.ndarray]): Input data to normalize.

        returns:
            torch.Tensor: Normalized data.
        """
        return self.normalize(x)

    def __getitem__(self, key: str):
        """
        Allows accessing a specific field's normalizer.

        args:
            key (str): Key of the field.

        returns:
            SingleFieldLinearNormalizer: Normalizer for the specified field.
        """
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str, value: "SingleFieldLinearNormalizer"):
        """
        Allows setting a specific field's normalizer.

        args:
            key (str): Key of the field.
            value (SingleFieldLinearNormalizer): Normalizer to set.
        """
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward: bool = True):
        """
        Internal method to perform normalization or unnormalization.

        args:
            x: Input data.
            forward (bool, optional): If True, perform normalization; otherwise, unnormalize.

        returns:
            torch.Tensor: Transformed data.
        """
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if "_default" not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict["_default"]
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Normalizes the input data.

        args:
            x (Union[Dict, torch.Tensor, np.ndarray]): Input data to normalize.

        returns:
            torch.Tensor: Normalized data.
        """
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Unnormalizes the input data.

        args:
            x (Union[Dict, torch.Tensor, np.ndarray]): Data to unnormalize.

        returns:
            torch.Tensor: Unnormalized data.
        """
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        """
        Retrieves input statistics used for normalization.

        returns:
            Dict: Input statistics.

        Raises:
            RuntimeError: If the normalizer is not initialized.
        """
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and "_default" in self.params_dict:
            return self.params_dict["_default"]["input_stats"]

        result = dict()
        for key, value in self.params_dict.items():
            if key != "_default":
                result[key] = value["input_stats"]
        return result

    def get_output_stats(self, key: str = "_default"):
        """
        Retrieves output statistics after normalization.

        args:
            key (str, optional): Specific key to retrieve stats for.

        returns:
            Dict: Output statistics.
        """
        input_stats = self.get_input_stats()
        if "min" in input_stats:
            # No dictionary, single field
            return dict_apply(input_stats, self.normalize)

        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key: value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(nn.Module):
    available_modes = ["limits", "gaussian"]

    def __init__(self, params_dict: nn.ParameterDict = None):
        """
        Initializes the SingleFieldLinearNormalizer.

        args:
            params_dict (nn.ParameterDict, optional): A ParameterDict containing normalization parameters.
                If None, an empty ParameterDict is initialized.
        """
        super().__init__()
        if params_dict is None:
            self.params_dict = nn.ParameterDict()
        else:
            self.params_dict = params_dict

    @property
    def device(self):
        """
        returns the device of the first parameter in the ParameterDict.
        """
        return next(iter(self.parameters())).device

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Custom method to load parameters from a state_dict.

        args:
            state_dict (dict): State dictionary containing model parameters.
            prefix (str): Prefix for parameter keys.
            local_metadata (dict): Metadata for loading.
            strict (bool): Whether to strictly enforce parameter matching.
            missing_keys (list): List of missing keys.
            unexpected_keys (list): List of unexpected keys.
            error_msgs (list): List of error messages.
        """

        def dfs_add(dest, keys, value: torch.Tensor):
            if len(keys) == 1:
                dest[keys[0]] = value
                return

            if keys[0] not in dest:
                dest[keys[0]] = nn.ParameterDict()
            dfs_add(dest[keys[0]], keys[1:], value.clone())

        def load_dict(state_dict, prefix):
            out_dict = nn.ParameterDict()
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    param_keys = key[len(prefix) :].split(".")[1:]
                    dfs_add(out_dict, param_keys, value.clone())
            return out_dict

        self.params_dict = load_dict(state_dict, prefix + "params_dict")
        self.params_dict.requires_grad_(False)
        return

    @torch.no_grad()
    def fit(
        self,
        data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims: int = 1,
        dtype: torch.dtype = torch.float32,
        mode: str = "limits",
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
    ):
        """
        Fits the normalizer parameters to the provided data.

        args:
            data (Union[torch.Tensor, np.ndarray, zarr.Array]): Input data to fit.
            last_n_dims (int, optional): Number of dimensions to consider as features.
            dtype (torch.dtype, optional): Data type for computations.
            mode (str, optional): Normalization mode ("limits" or "gaussian").
            output_max (float, optional): Maximum output value for normalization.
            output_min (float, optional): Minimum output value for normalization.
            range_eps (float, optional): Epsilon value to avoid division by zero.
            fit_offset (bool, optional): Whether to fit an offset.
        """
        self.params_dict = _fit(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
        )

    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray, zarr.Array], **kwargs):
        """
        Creates a SingleFieldLinearNormalizer by fitting to the data.

        args:
            data (Union[torch.Tensor, np.ndarray, zarr.Array]): Data to fit.
            **kwargs: Additional keyword arguments for fitting.

        returns:
            SingleFieldLinearNormalizer: Fitted normalizer.
        """
        obj = cls()
        obj.fit(data, **kwargs)
        return obj

    @classmethod
    def create_manual(
        cls,
        scale: Union[torch.Tensor, np.ndarray],
        offset: Union[torch.Tensor, np.ndarray],
        input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    ):
        """
        Creates a SingleFieldLinearNormalizer with manually specified parameters.

        args:
            scale (Union[torch.Tensor, np.ndarray]): Scale factors.
            offset (Union[torch.Tensor, np.ndarray]): Offset values.
            input_stats_dict (Dict[str, Union[torch.Tensor, np.ndarray]]): Input statistics.

        returns:
            SingleFieldLinearNormalizer: Manually configured normalizer.
        """

        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            return x.flatten()

        # Validate shapes and dtypes
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape, "Shape mismatch in manual creation."
            assert x.dtype == scale.dtype, "Dtype mismatch in manual creation."

        params_dict = nn.ParameterDict(
            {
                "scale": to_tensor(scale),
                "offset": to_tensor(offset),
                "input_stats": nn.ParameterDict(
                    {k: to_tensor(v) for k, v in input_stats_dict.items()}
                ),
            }
        )
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype: torch.dtype = torch.float32):
        """
        Creates an identity SingleFieldLinearNormalizer.

        args:
            dtype (torch.dtype, optional): Data type for the parameters.

        returns:
            SingleFieldLinearNormalizer: Identity normalizer.
        """
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            "min": torch.tensor([-1], dtype=dtype),
            "max": torch.tensor([1], dtype=dtype),
            "mean": torch.tensor([0], dtype=dtype),
            "std": torch.tensor([1], dtype=dtype),
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Normalizes the input data.

        args:
            x (Union[torch.Tensor, np.ndarray]): Input data to normalize.

        returns:
            torch.Tensor: Normalized data.
        """
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Unnormalizes the input data.

        args:
            x (Union[torch.Tensor, np.ndarray]): Data to unnormalize.

        returns:
            torch.Tensor: Unnormalized data.
        """
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self) -> Dict:
        """
        Retrieves input statistics used for normalization.

        returns:
            Dict: Input statistics.
        """
        return self.params_dict["input_stats"]

    def get_output_stats(self):
        """
        Retrieves output statistics after normalization.

        returns:
            Dict: Output statistics.
        """
        return dict_apply(self.params_dict["input_stats"], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Allows the normalizer to be called as a function to normalize data.

        args:
            x (Union[torch.Tensor, np.ndarray]): Input data to normalize.

        returns:
            torch.Tensor: Normalized data.
        """
        return self.normalize(x)


def _fit(
    data: Union[torch.Tensor, np.ndarray, zarr.Array],
    last_n_dims: int = 1,
    dtype: torch.dtype = torch.float32,
    mode: str = "limits",
    output_max: float = 1.0,
    output_min: float = -1.0,
    range_eps: float = 1e-4,
    fit_offset: bool = True,
) -> nn.ParameterDict:
    """
    Computes normalization parameters based on the input data.

    args:
        data (Union[torch.Tensor, np.ndarray, zarr.Array]): Input data.
        last_n_dims (int, optional): Number of dimensions to consider as features.
        dtype (torch.dtype, optional): Data type for computations.
        mode (str, optional): Normalization mode ("limits" or "gaussian").
        output_max (float, optional): Maximum output value for normalization.
        output_min (float, optional): Minimum output value for normalization.
        range_eps (float, optional): Epsilon value to avoid division by zero.
        fit_offset (bool, optional): Whether to fit an offset.

    returns:
        nn.ParameterDict: Computed normalization parameters.
    """
    assert mode in ["limits", "gaussian"], "Mode must be 'limits' or 'gaussian'."
    assert last_n_dims >= 0, "last_n_dims must be non-negative."
    assert output_max > output_min, "output_max must be greater than output_min."

    # Convert data to torch.Tensor if necessary
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # Reshape data to combine the last `last_n_dims` dimensions
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1, dim)

    # Compute input statistics
    input_min, _ = data.min(dim=0)
    input_max, _ = data.max(dim=0)
    input_mean = data.mean(dim=0)
    input_std = data.std(dim=0)

    # Compute scale and offset based on the chosen mode
    if mode == "limits":
        if fit_offset:
            # Scale to fit within [output_min, output_max]
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        else:
            # Assume data is pre-zero-centered and scale based on absolute max
            assert (
                output_max > 0 and output_min < 0
            ), "output_max must be positive and output_min must be negative when fit_offset is False."
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == "gaussian":
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = -input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)

    # Create ParameterDict to store normalization parameters
    this_params = nn.ParameterDict(
        {
            "scale": scale,
            "offset": offset,
            "input_stats": nn.ParameterDict(
                {
                    "min": input_min,
                    "max": input_max,
                    "mean": input_mean,
                    "std": input_std,
                }
            ),
        }
    )
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params: nn.ParameterDict, forward: bool = True) -> torch.Tensor:
    """
    Applies normalization or unnormalization to the input data.

    args:
        x: Input data.
        params (nn.ParameterDict): Normalization parameters.
        forward (bool, optional): If True, perform normalization; otherwise, unnormalize.

    returns:
        torch.Tensor: Transformed data.
    """
    assert "scale" in params, "Scale parameter missing in params."
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params["scale"]
    offset = params["offset"]
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x


def test():
    """
    Unit tests for the normalizer classes.
    """
    # Test SingleFieldLinearNormalizer with limits mode
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    data[..., 0, 0] = 0

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode="limits", last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.0)
    assert np.allclose(datan.min(), -1.0)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    # Test SingleFieldLinearNormalizer without fitting offset
    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode="limits", last_n_dims=1, fit_offset=False)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.0, atol=1e-3)
    assert np.allclose(datan.min(), 0.0, atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    # Test SingleFieldLinearNormalizer with gaussian mode
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode="gaussian", last_n_dims=0)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.mean(), 0.0, atol=1e-3)
    assert np.allclose(datan.std(), 1.0, atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    # Test LinearNormalizer with dictionary input
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    data[..., 0, 0] = 0

    normalizer = LinearNormalizer()
    normalizer.fit(data, mode="limits", last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.0)
    assert np.allclose(datan.min(), -1.0)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    # Test LinearNormalizer with multiple dictionary entries
    data = {
        "obs": torch.zeros((1000, 128, 9, 2)).uniform_() * 512,
        "action": torch.zeros((1000, 128, 2)).uniform_() * 512,
    }
    normalizer = LinearNormalizer()
    normalizer.fit(data)
    datan = normalizer.normalize(data)
    dataun = normalizer.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    # Test state_dict functionality
    state_dict = normalizer.state_dict()
    n = LinearNormalizer()
    n.load_state_dict(state_dict)
    datan = n.normalize(data)
    dataun = n.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)

    print("All tests passed successfully.")


if __name__ == "__main__":
    test()
