"""Interchangeable float32 CPU backends with shared action selection."""
from enum import Enum
from pathlib import Path

import numpy as np

INPUT_NAMES = ('main','others','ball','actions','boosts')


class ActionMode(str, Enum):
    STOCHASTIC = "stochastic"
    DETERMINISTIC = "deterministic"
    KL = "kl"
    INVERSE_KL = "inverse-kl"


def load_model(path):
    from torch_runtime import torch
    # Keep the returned module: freeze folds the 276 constant action embeddings.
    return torch.jit.optimize_for_inference(torch.jit.freeze(
        torch.jit.load(str(path), map_location="cpu").eval()
    ))


class TorchModel:
    def __init__(self, path):
        self.model = load_model(path)

    def __call__(self, observation):
        from torch_runtime import torch
        with torch.inference_mode():
            inputs = tuple(torch.from_numpy(np.asarray(x,dtype=np.float32)) for x in observation)
            return self.model(inputs).numpy()


class OnnxModel:
    def __init__(self, path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        options.add_session_config_entry('session.inter_op.allow_spinning','0')
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        if tuple(x.name for x in self.session.get_inputs()) != INPUT_NAMES:
            raise ValueError(f'Unexpected ONNX inputs in {path}')

    def __call__(self, observation):
        inputs = {name:np.ascontiguousarray(x,dtype=np.float32)
                  for name,x in zip(INPUT_NAMES,observation)}
        return self.session.run(['logits'],inputs)[0]


def log_softmax(logits):
    shifted = logits.astype(np.float64) - np.max(logits,axis=-1,keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1,keepdims=True))


class Agent:
    def __init__(self, model: Path, base: Path | None = None, mode=ActionMode.STOCHASTIC,
                 backend=None, seed=None, available_modes=None):
        self.mode = ActionMode(mode)
        self.available_modes = tuple(ActionMode(m) for m in available_modes) if available_modes else (self.mode,)
        backend = backend or ('onnx' if Path(model).suffix == '.onnx' else 'torch')
        factory = {'onnx':OnnxModel,'torch':TorchModel}[backend]
        needs_base = any(m in (ActionMode.KL, ActionMode.INVERSE_KL) for m in self.available_modes)
        if needs_base and base is None:
            raise ValueError("KL selection requires a base model")
        self.backend = backend
        self.model = factory(model)
        self.base = factory(base) if needs_base else None
        self.rng = np.random.default_rng(seed)

    def act(self, observation, stochastic=False):
        logits = self.model(observation)
        mode = ActionMode.STOCHASTIC if stochastic else self.mode
        if mode == ActionMode.DETERMINISTIC:
            return logits.argmax(-1)
        log_policy = log_softmax(logits)
        if mode == ActionMode.STOCHASTIC:
            cdf = np.cumsum(np.exp(log_policy),axis=-1)
            return np.minimum((self.rng.random((len(logits),1)) >= cdf).sum(axis=-1),
                              logits.shape[-1]-1)
        log_base = log_softmax(self.base(observation))
        weight = log_policy if mode == ActionMode.KL else log_base
        return (np.exp(weight) * (log_policy-log_base)).argmax(-1)

    def warmup(self, batch_size):
        obs = [np.zeros(shape, dtype=np.float32) for shape in (
            (batch_size, 32), (batch_size, 5, 21), (batch_size, 6, 9),
            (batch_size, 6, 8), (batch_size, 34, 5))]
        original = self.mode
        try:
            for mode in self.available_modes:
                self.mode = mode
                for _ in range(3):
                    self.act(obs)
        finally:
            self.mode = original
