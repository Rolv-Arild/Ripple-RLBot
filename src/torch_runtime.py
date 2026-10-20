"""Optional CPU PyTorch runtime for conversion and reference comparisons."""

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise ModuleNotFoundError(
        'PyTorch comparisons require the optional dependency: uv sync --extra torch',
        name='torch',
    ) from error

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # Another importer may already have initialized the interop pool.
