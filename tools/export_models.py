"""Convert all supplied policies to ONNX and check against the original exports."""
import argparse
from pathlib import Path
import sys
import hashlib
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from torch_runtime import torch
from export_network import ExportNetwork
from variants import source_models

INPUT_NAMES = ['main','others','ball','actions','boosts']


def observation(batch, rng):
    obs = [rng.normal(size=shape).astype(np.float32) for shape in
           ((batch,32),(batch,5,21),(batch,6,9),(batch,6,8),(batch,34,5))]
    obs[1][:,:,15] = rng.choice([-1,0,1], size=(batch,5))
    obs[0][:,30] = rng.integers(0,2,size=batch)
    return obs


def convert(path):
    original = torch.jit.load(str(path), map_location='cpu').eval()
    exportable = ExportNetwork(original.state_dict())
    rng = np.random.default_rng(83)
    example = tuple(torch.from_numpy(x) for x in observation(2,rng))
    with torch.inference_mode():
        torch.testing.assert_close(exportable(*example), original(example), rtol=2e-4, atol=3e-5)
    output = path.with_suffix('.onnx')
    batch_dim = torch.export.Dim('batch', min=1, max=8)
    torch.onnx.export(exportable, example, str(output), input_names=INPUT_NAMES,
        output_names=['logits'], opset_version=18, dynamo=True,
        dynamic_shapes=tuple({0:batch_dim} for _ in example), external_data=False, verbose=False)
    exported = onnx.load(str(output))
    onnx.helper.set_model_props(exported, {
        'source_file':path.name,
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'torch_version':torch.__version__,
        'converter':'tools/export_models.py',
        'precision':'float32',
    })
    onnx.save(exported,str(output))
    onnx.checker.check_model(str(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output), options, providers=['CPUExecutionProvider'])
    max_error = 0.
    decisions = 0
    with torch.inference_mode():
        for batch in (1,2,3,6):
            for _ in range(25):
                obs = observation(batch,rng)
                expected = original(tuple(torch.from_numpy(x) for x in obs)).numpy()
                actual = session.run(None,dict(zip(INPUT_NAMES,obs)))[0]
                np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-4)
                np.testing.assert_array_equal(actual.argmax(-1),expected.argmax(-1))
                max_error = max(max_error,float(np.max(np.abs(actual-expected))))
                decisions += batch
    print(f'{path.name}: {decisions} matching decisions, max logit error {max_error:.8g}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    for path in ([args.model] if args.model else source_models()):
        convert(path)
