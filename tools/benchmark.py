"""Compare CPU backends for every version, including the full packet adapter."""
import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from importlib.metadata import version

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agent import Agent,OnnxModel,TorchModel
from variants import VARIANTS,model_paths


def measure(fn,iterations):
    for _ in range(20):
        fn()
    samples = []
    for _ in range(iterations):
        start = perf_counter()
        fn()
        samples.append((perf_counter()-start)*1000)
    return dict(zip(('p50_ms','p95_ms','p99_ms'),np.percentile(samples,[50,95,99])))


def full_pipeline(variant,backend,batch,iterations,bcm_only=False):
    sys.path.insert(0,str(ROOT/'tests'))
    from packets import packet,field_info
    from bot import Ripple
    from rlbot import flat
    bot = Ripple(variant=variant,backend=backend,mode='stochastic',
                 stochastic_kickoffs=False,seed=42,bcm_only=bcm_only)
    bot.indices = list(range(0,batch*2,2))
    bot.team = 0
    bot.field_info = field_info()
    bot.initialize()
    game_packet = packet(100)
    bot.get_outputs(game_packet)

    def advance(force_reset=False):
        nonlocal game_packet
        frame = game_packet.match_info.frame_num+4
        match_info = flat.MatchInfo(frame_num=frame,seconds_elapsed=frame/120.,
            match_phase=flat.MatchPhase.Active,game_time_remaining=300.,world_gravity_z=-650.,game_speed=1.)
        b = bot.obs.ball_predictions[1].copy()
        if force_reset:
            b[0] = (frame%100)*10.
        balls = [flat.BallInfo(physics=flat.Physics(location=flat.Vector3(*map(float,b[:3])),
            rotation=flat.Rotator(),velocity=flat.Vector3(*map(float,b[3:6])),
            angular_velocity=flat.Vector3(*map(float,b[6:]))))]
        game_packet = flat.GamePacket(players=game_packet.players,boost_pads=game_packet.boost_pads,
            balls=balls,match_info=match_info,teams=game_packet.teams,tiles=[])
        bot.get_outputs(game_packet)

    return {
        'cached_prediction':measure(advance,iterations),
        'new_prediction':measure(lambda:advance(True),iterations),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=300)
    parser.add_argument('--variant',choices=['all',*VARIANTS],default='all')
    parser.add_argument('--bcm-only',action='store_true')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    variants = VARIANTS if args.variant=='all' else [args.variant]
    results = {
        'versions':{name:version(name) for name in ('torch','onnxruntime','numpy','rlbot','rocketsim','rlgym-compat')},
        'python':platform.python_version(),'cpu':platform.processor(),
        'iterations':args.iterations,'threads':1,'bcm_only':args.bcm_only,'variants':{},
    }
    for name in variants:
        variant = VARIANTS[name]
        result = results['variants'][name] = {}
        for backend,factory in (('onnx',OnnxModel),('torch',TorchModel)):
            policy,base = model_paths(variant,backend,bcm_only=args.bcm_only)
            model = factory(policy)
            agent = Agent(policy,base,
                          mode='kl' if base else 'stochastic',backend=backend,seed=42)
            timings = result[backend] = {}
            for batch in (1,2,3):
                rng = np.random.default_rng(48)
                obs = [rng.normal(size=shape).astype(np.float32) for shape in
                       ((batch,32),(batch,5,21),(batch,6,9),(batch,6,8),(batch,34,5))]
                obs[0][:,30] = np.arange(batch)%2
                obs[1][:,:,15] = rng.choice([-1,0,1],size=(batch,5))
                timings[str(batch)] = {
                    'network':measure(lambda:model(obs),args.iterations),
                    'kl_selection' if base else 'stochastic_selection':
                        measure(lambda:agent.act(obs),args.iterations),
                }
            for batch in (1,3):
                timings[str(batch)]['full_decision'] = full_pipeline(name,backend,batch,args.iterations,args.bcm_only)
        print(f'Completed {variant.name}',file=sys.stderr,flush=True)
    report = json.dumps(results,indent=2)
    if args.output:
        args.output.write_text(report+'\n',encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
