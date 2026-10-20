from pathlib import Path
import argparse
import sys
import json
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tests'))
from test_backends import observation_corpus
from agent import OnnxModel,log_softmax,TorchModel
from torch_runtime import torch
from export_network import ExportNetwork
from variants import ASSETS, VARIANTS, source_models

parser = argparse.ArgumentParser(description='Check all model backends against the original TorchScript exports')
parser.add_argument('--samples',type=int,default=50,help='Observation fixtures per match size')
parser.add_argument('--output',type=Path)
args = parser.parse_args()
report = {}
corpus = list(observation_corpus(args.samples))
for path in source_models():
    name = path.relative_to(ASSETS).with_suffix('').as_posix()
    original = torch.jit.load(str(path),map_location='cpu').eval()
    eager = ExportNetwork(original.state_dict())
    optimized = TorchModel(path)
    onnx = OnnxModel(path.with_suffix('.onnx'))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    plain = ort.InferenceSession(str(path.with_suffix('.onnx')),opts,providers=['CPUExecutionProvider'])
    result = report[name] = {}
    for backend in ('eager','optimized','onnx','onnx_unoptimized'):
        max_prob = max_tv = max_logit = 0.
        disagree = decisions = 0
        with torch.inference_mode():
            for obs in corpus:
                inputs = tuple(torch.from_numpy(x) for x in obs)
                expected = original(inputs).numpy()
                if backend=='eager': actual = eager(*inputs).numpy()
                elif backend=='optimized': actual = optimized(obs)
                elif backend=='onnx': actual = onnx(obs)
                else: actual = plain.run(None,dict(zip(('main','others','ball','actions','boosts'),obs)))[0]
                p,q = np.exp(log_softmax(actual)),np.exp(log_softmax(expected))
                max_prob = max(max_prob,float(np.abs(p-q).max()))
                max_tv = max(max_tv,float(np.abs(p-q).sum(axis=-1).max()/2))
                max_logit = max(max_logit,float(np.abs(actual-expected).max()))
                disagree += int((actual.argmax(-1)!=expected.argmax(-1)).sum())
                decisions += len(actual)
        result[backend] = {'max_probability_error':max_prob,'max_total_variation':max_tv,
                           'max_logit_error':max_logit,'greedy_disagreements':disagree,'decisions':decisions}
        tolerance = 5e-3 if name==VARIANTS['v1.1'].base else 1e-4
        if disagree or max_prob > tolerance or max_tv > tolerance:
            raise AssertionError(f'{name}/{backend} failed parity limits: {result[backend]}')
output = json.dumps(report,indent=2)
if args.output:
    args.output.write_text(output+'\n',encoding='utf-8')
print(output)
