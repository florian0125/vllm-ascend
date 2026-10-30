"""Run one isolated benchmark and save service-counter deltas.

Example (inside the serving container):
python tools/aisbench_with_stats.py --counter-file /path/benchmark_counters_rank0.json \
    --out /path/bs1 -- ais_bench <usual arguments>
No other client may send requests during the measured interval.
"""
import sys

# This repository has tools/bisect/, which otherwise shadows the stdlib
# bisect module imported transitively by urllib when this file runs directly.
if __name__ == '__main__' and sys.path:
    sys.path.pop(0)

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def calculate(before, after):
    if before['session_id'] != after['session_id']:
        raise ValueError('Service restarted during benchmark; refusing mixed statistics')
    if after['sequence'] <= before['sequence']:
        raise ValueError('No new service snapshot; statistics are stale')
    def delta(a, b):
        result = {key: int(b[key]) - int(a.get(key, 0)) for key in b}
        if any(value < 0 for value in result.values()):
            raise ValueError('Counters decreased; refusing invalid statistics')
        return result
    totals = delta(before['totals'], after['totals'])
    spec = delta(before['speculative'], after['speculative'])
    def ratio(n, d):
        return n / d if d else None
    return dict(totals=totals, speculative=spec, rates=dict(
        pre_route_hit_rate=ratio(totals['pre_hit_routes'], totals['pre_routes']),
        pre_expert_hit_rate=ratio(totals['pre_hit_experts'], totals['pre_experts']),
        post_route_hit_rate=ratio(totals['post_hit_routes'], totals['post_routes']),
        post_expert_hit_rate=ratio(totals['post_hit_experts'], totals['post_experts']),
        npu_execution_share=ratio(totals['npu_routes'], totals['post_routes']),
        draft_token_acceptance_rate=ratio(spec['accepted'], spec['proposed']),
        mean_acceptance_length=(1 + spec['accepted'] / spec['drafts']) if spec['drafts'] else None,
    ), per_layer={layer: delta(before['per_layer'].get(layer, {}), values)
                  for layer, values in after['per_layer'].items()},
        scope=after['scope'], excluded_hash_layers=after['excluded_hash_layers'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counter-file', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8150')
    parser.add_argument('--settle-seconds', type=float, default=2.0)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Supply the normal ais_bench command after --')
    if args.out.exists():
        parser.error('Output directory exists; use a fresh directory for each run')
    # Require health and a current initialized snapshot before launching workload.
    urllib.request.urlopen(args.url.rstrip('/') + '/health', timeout=10).close()
    before = json.loads(args.counter_file.read_text())
    args.out.mkdir(parents=True)
    (args.out / 'before.json').write_text(json.dumps(before, indent=2))
    started = time.time_ns()
    def metrics(name):
        try:
            text = urllib.request.urlopen(args.url.rstrip('/') + '/metrics', timeout=10).read()
            (args.out / name).write_bytes(text)
        except Exception as exc:
            (args.out / (name + '.error')).write_text(str(exc))
    metrics('metrics_before.txt')
    print('Measuring one benchmark; keep other clients idle.', flush=True)
    result = subprocess.run(command, check=False)
    ended = time.time_ns()
    # Writer flushes the final completed step even when the service is idle.
    # The benchmark must return after all of its requests have completed.
    time.sleep(max(0.5, args.settle_seconds))
    after = json.loads(args.counter_file.read_text())
    (args.out / 'after.json').write_text(json.dumps(after, indent=2))
    metrics('metrics_after.txt')
    report = calculate(before, after)
    report.update(command=command, exit_code=result.returncode,
                  started_wall_ns=started, ended_wall_ns=ended,
                  session_id=after['session_id'], configuration=after.get('configuration', {}),
                  limitation='Service-wide interval counters: requires exclusive benchmark traffic. Metrics endpoint snapshots may lag; acceptance in rates uses per-step host bookkeeping counters. Instrumentation overhead is not assumed zero.')
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['rates'], indent=2))
    print('Saved', args.out / 'summary.json')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
