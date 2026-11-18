"""Submit TrainConfig sweeps to the local Slurm cluster (no Modal dependency)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace

SUPPORTED_GPUS = ('H100', 'H200', 'B200')
DEFAULT_PARTITION = 'mit_normal_gpu'


def gpu_types(gpu):
    if gpu is None or (isinstance(gpu, str) and gpu.strip().lower() == 'any'):
        return SUPPORTED_GPUS
    values = gpu.split(',') if isinstance(gpu, str) else list(gpu)
    values = tuple(dict.fromkeys(str(v).strip().upper() for v in values))
    if not values or any(v not in SUPPORTED_GPUS for v in values):
        raise ValueError('gpu must be any, H100, H200, B200, or a sequence of these')
    return values


def node_selection(inventory, allowed, gpus):
    """Exclude nonmatching nodes; never pass an OR pool as --nodelist.

    Slurm interprets --nodelist as required nodes, not alternative nodes.
    Exclude mixed or nonmatching nodes so an untyped allocation uses only allowed GPUs.
    """
    eligible, excluded = set(), set()
    for line in inventory.splitlines():
        if not line.strip():
            continue
        node, gres = line.strip().split('|', 1)
        entries = re.findall(r'(?:^|,)gpu:([^:,()]+):(\d+)', gres)
        kinds = {kind.upper() for kind, _ in entries}
        count = sum(int(n) for _, n in entries)
        if kinds and kinds <= set(allowed) and count >= gpus:
            eligible.add(node)
        else:
            excluded.add(node)
    eligible -= excluded
    if not eligible:
        raise ValueError(f'No nodes have {gpus} GPU(s) of the requested types {allowed}')
    return sorted(eligible), sorted(excluded)


def encode_config(config):
    from metric_logging import importable_metric_loggers
    payload = asdict(replace(config, metric_loggers=importable_metric_loggers(config.metric_loggers)))
    imports = [payload['model_builder'], payload['optimizer_builder']]
    imports.extend(row['fn'] for row in payload['metric_loggers'])
    if any(path and path.startswith('__main__:') for path in imports):
        raise ValueError('Builders/loggers must live in importable modules, not __main__')
    # Round-trip now so unsupported builder kwargs fail before any submission.
    return json.loads(json.dumps(payload, default=_json_path, allow_nan=False))


def _json_path(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Config value is not JSON serializable: {type(value).__name__}')


def decode_config(payload):
    from data import TokenDatasetConfig
    from metric_logging import MetricLogger
    from model_config import LMConfig
    from train import TrainConfig
    values = dict(payload)
    for key in ('train_dataset', 'val_dataset'):
        values[key] = TokenDatasetConfig(**values[key])
    if values['model_config'] is not None:
        values['model_config'] = LMConfig.from_dict(values['model_config'])
    values['metric_loggers'] = tuple(MetricLogger(**row) for row in values['metric_loggers'])
    for key in ('wandb_tags', 'keep_checkpoint_steps'):
        values[key] = tuple(values[key])
    return TrainConfig(**values)


def check_runtime(allowed):
    import torch
    if 'B200' in allowed:
        version = tuple(int(v) for v in torch.__version__.split('+')[0].split('.')[:2])
        cuda = tuple(int(v) for v in (torch.version.cuda or '0.0').split('.')[:2])
        if version < (2, 7) or cuda < (12, 8):
            raise RuntimeError('B200 requires a Blackwell-capable environment (PyTorch >=2.7, '
                               'CUDA >=12.8). Use that environment to launch, or select '
                               'gpu="H100" or gpu="H200" with the course PyTorch 2.6 environment.')


def launch_training_jobs(configs, *, gpu=SUPPORTED_GPUS, max_parallel_runs=None,
                         partition=DEFAULT_PARTITION, account=None, qos=None,
                         gpus=1, cpus=8, mem=64, time_limit='06:00:00',
                         dependency=None, dry_run=False):
    """One array task per config; return job IDs, logs, and staged config paths.

    Uses the calling Python environment and the shared checkout. dry_run stages
    a reviewable manifest/script without submitting. mem is host RAM in GiB.
    """
    from train import checked_train_config, training_run_name
    from utils import REPO_ROOT, USER_CONFIG, DATA_DIR
    from model_io import is_native_model_dir

    if not isinstance(configs, list) or not configs:
        raise ValueError('Supply a nonempty list of TrainConfig objects')
    for name, value in (('gpus', gpus), ('cpus', cpus), ('mem', mem),
                        ('max_parallel_runs', max_parallel_runs)):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            raise ValueError(f'{name} must be a positive integer')
    allowed = gpu_types(gpu)
    jobs, pending, destinations = [], [], set()
    for index, config in enumerate(configs):
        checked_train_config(config)
        name = training_run_name(config)
        destination = (Path(config.model_dir).expanduser().resolve() / name)
        if destination in destinations:
            raise ValueError(f'Duplicate output directory: {destination}; use distinct run_name_suffix values')
        destinations.add(destination)
        skipped = config.save_model and not config.force_run and is_native_model_dir(destination)
        jobs.append(dict(job_index=index, run_name=name, run_name_suffix=config.run_name_suffix,
                         skipped=skipped, slurm_job_id=None, gpu=allowed, dry_run=dry_run))
        if not skipped:
            config = replace(config, model_dir=str(destination.parent))
            paths = {}
            for field in ('train_dataset', 'val_dataset'):
                dataset = getattr(config, field)
                if dataset.path is not None:
                    paths[field] = replace(dataset, path=str(Path(dataset.path).expanduser().resolve()))
            if config.init_checkpoint_path is not None:
                paths['init_checkpoint_path'] = str(Path(config.init_checkpoint_path).expanduser().resolve())
            config = replace(config, **paths)
            pending.append(dict(job_index=index, config=encode_config(config)))
    if not pending:
        print('All requested models already exist; no jobs submitted.')
        return jobs
    if not dry_run:
        check_runtime(allowed)
    inventory = subprocess.run(['sinfo', '-N', '-h', '-p', partition, '-o', '%N|%G'],
                               check=True, capture_output=True, text=True).stdout
    eligible, excluded = node_selection(inventory, allowed, gpus)
    stage_root = Path(USER_CONFIG.temp_script_dir).expanduser().resolve()
    log_dir = Path(USER_CONFIG.slurm_log_dir).expanduser().resolve()
    stage_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='cluster-', dir=stage_root))
    manifest = stage / 'configs.json'
    manifest.write_text(json.dumps(dict(version=1, gpu=allowed, jobs=pending), indent=2) + '\n')
    limit = min(max_parallel_runs or len(pending), len(pending))
    # The local submit plugin rewrites generic --gres/--gpus requests to L40S.
    # --gpus-per-task is a distinct Slurm resource request and retains the OR pool.
    gpu_request = (f'--gres=gpu:{allowed[0].lower()}:{gpus}' if len(allowed) == 1
                   else f'--gpus-per-task={gpus}')
    options = ['--parsable', '--nodes=1', '--ntasks=1', f'--partition={partition}',
               gpu_request, f'--cpus-per-task={cpus}', f'--mem={mem}G',
               f'--time={time_limit}', f'--array=0-{len(pending)-1}%{limit}',
               f'--output={log_dir}/%A_%a.out', '--job-name=dl-alchemy',
               '--signal=B:TERM@120', f'--chdir={REPO_ROOT}']
    for key, value in (('account', account), ('qos', qos), ('dependency', dependency),
                       ('exclude', ','.join(excluded))):
        if value:
            options.append(f'--{key}={value}')
    # Keep the venv path: resolving a venv symlink would select the base Python.
    python = os.path.abspath(sys.executable)
    script = stage / 'job.sh'
    script.write_text('#!/bin/bash\nset -euo pipefail\n'
        'export PYTHONUNBUFFERED=1\n'
        f'export DL_ALCHEMY_DATA_DIR={shlex.quote(str(Path(DATA_DIR).expanduser().resolve()))}\n'
        f'cd {shlex.quote(str(REPO_ROOT))}\n'
        f'{shlex.quote(python)} -m cluster_train --manifest {shlex.quote(str(manifest))} &\n'
        'child_pid=$!\n'
        'forward_signal() {\n'
        '  trap "" TERM INT\n'
        '  kill -TERM "$child_pid" 2>/dev/null || true\n'
        '  set +e\n  wait "$child_pid"\n  exit 143\n}\n'
        'trap forward_signal TERM INT\nwait "$child_pid"\n')
    command = ['sbatch', *options, str(script)]
    (stage / 'submit.json').write_text(json.dumps(command, indent=2) + '\n')
    print(f'{len(pending)} tasks; at most {limit} concurrent; {len(eligible)} eligible nodes.')
    print(shlex.join(command))
    array_id = None
    if not dry_run:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        array_id = result.stdout.strip().split(';')[0]
        if not array_id.isdigit():
            raise RuntimeError(f'Unexpected sbatch response (check squeue before retrying): {result.stdout!r}')
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
    for task, row in enumerate(pending):
        job = jobs[row['job_index']]
        job.update(slurm_array_id=array_id, array_task_id=task,
                   slurm_job_id=f'{array_id}_{task}' if array_id else None,
                   log_path=str(log_dir / f'{array_id or "%A"}_{task}.out'),
                   manifest_path=str(manifest), script_path=str(script),
                   submission_command=command)
        print(f'  {job["run_name"]}: {job["slurm_job_id"] or "dry run"}')
    return jobs


def run_task(manifest_path, task_id):
    import torch
    from train import train
    payload = json.loads(Path(manifest_path).read_text())
    if payload['version'] != 1:
        raise ValueError('Unsupported cluster manifest version')
    if not 0 <= task_id < len(payload['jobs']):
        raise ValueError(f'Invalid array task index {task_id}')
    if not torch.cuda.is_available():
        raise RuntimeError('Slurm worker has no CUDA GPU; refusing to train on CPU')
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if any(not any(kind in name.upper() for kind in payload['gpu']) for name in names):
        raise RuntimeError(f'Slurm allocated unexpected GPU(s): {names}')
    check_runtime(tuple(kind for kind in payload['gpu'] if any(kind in n.upper() for n in names)))
    print(f'GPUs: {names}; PyTorch {torch.__version__}; CUDA {torch.version.cuda}', flush=True)
    train(decode_config(payload['jobs'][task_id]['config']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--task-id', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    args = parser.parse_args()
    run_task(args.manifest, args.task_id)
