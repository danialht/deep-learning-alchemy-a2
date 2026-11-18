# Non-Modal GPU Guide

This repo is still usable without Modal. Modal is only the default course launch
path. The core training entrypoint is the normal Python function
`train(TrainConfig(...))`, and it runs on local CUDA when `torch.cuda.is_available()`
is true.

## Viability

The non-Modal path is viable:

- training does not require Modal;
- training reads raw `.bin` token files through a NumPy memmap;
- checkpoints and final models save to a local `model_dir`;
- W&B can use your normal local `wandb login`;
- one local CUDA GPU uses `torch.compile(mode="reduce-overhead")`;
- multiple visible CUDA GPUs run through `torch.nn.DataParallel`, with compile
  disabled.

The main thing a non-Modal user must do is download the data locally and choose
local storage directories.

## Setup

Install `uv` using the official guide: <https://docs.astral.sh/uv/getting-started/installation/>.

From the repo root:

```bash
uv python install 3.10
uv sync
```

Check CUDA:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If this prints `False`, the default run will fall back to CPU and will be too
slow for normal use.

## Configure Local Storage

Open [../utils.py](../utils.py) and edit the student-facing block near the top.
For a non-Modal GPU user, the important fields are:

```python
CONFIG_WANDB_ENTITY = "YOUR_WANDB_USERNAME_OR_TEAM"
CONFIG_WANDB_PROJECT = "assignments"
CONFIG_SCRATCH_ROOT = "/path/with/space/dl_alchemy"
```

By default, local users use `~/dl_alchemy`. Set `CONFIG_SCRATCH_ROOT` explicitly if your
home directory is small or you want to use an external disk or cluster scratch
filesystem.

Derived defaults:

```text
MODEL_DIR = CONFIG_SCRATCH_ROOT/ckpts
DATA_DIR = CONFIG_SCRATCH_ROOT/data
```

You can override those separately with `CONFIG_MODEL_DIR` and `CONFIG_DATA_DIR`.

## Download Data

Download the raw binary dataset:

```bash
uv run python -m download_data
```

This downloads plain files from `kothasuhas/dl_alchemy_seq9p6m_context1024` into
`DATA_DIR`. The expected layout is:

```text
DATA_DIR/dclm_9p6m_ctx1024/train/tokens.bin
DATA_DIR/dclm_9p6m_ctx1024/train/metadata.json
DATA_DIR/dclm_9p6m_ctx1024/val/tokens.bin
DATA_DIR/dclm_9p6m_ctx1024/val/metadata.json
```

The default `TrainConfig` uses `data_seed=42`. If a preshuffled seed-42 local
cache is not present, training loads the base dataset above and materializes the
globally shuffled prefix once before training starts. This costs extra startup
time and temporary disk roughly proportional to:

```text
num_train_sequences * context_length * token_dtype_size
```

For the default `600,000 x 1024` run, this is around 1.2 GiB when tokens are
`uint16`, or 2.4 GiB if a future dataset uses `uint32`.

For quick debugging only, `TrainConfig(data_seed=None)` skips shuffling and reads
the base data in order. That changes the data order relative to the default
course recipe.

## Run Locally

The example launcher is [local_default_train.py](local_default_train.py):

```python
import torch

from train import TrainConfig, train


CONFIG = TrainConfig(
    num_train_sequences=600_000,
    run_name_suffix="local-gpu",
    wandb_online=False,
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected.")
    train(CONFIG)
```

Run it from the repo root:

```bash
uv run python -m gpu.local_default_train
```

## W&B

Modal secrets are not used outside Modal. For local GPU runs:

```bash
uv run wandb login
```

Then set `wandb_online=True` in your `TrainConfig`. The W&B entity/project come
from `utils.py`, unless you pass `wandb_entity=` or `wandb_project=` directly in
`TrainConfig`.

## Custom Data

For the default dataset, prefer setting `CONFIG_DATA_DIR` in `utils.py` and
running `download_data`.

For a custom preprocessed token dataset, point the config at a split directory:

```python
from data import token_dataset_at
from train import TrainConfig, train

train(
    TrainConfig(
        num_train_sequences=100_000,
        train_dataset=token_dataset_at("/path/to/my_data/train", tag="my-data"),
        val_dataset=token_dataset_at("/path/to/my_data/val", tag="my-data"),
        data_seed=None,
        wandb_online=False,
    )
)
```

Each split directory must contain `tokens.bin` and `metadata.json` with the same
shape/metadata convention as the default data.

## Slurm

On a generic Slurm cluster, the most robust first test is to write a normal
Python launcher like `gpu/local_default_train.py`, then submit it with an
ordinary `sbatch` script:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd /path/to/assignments
uv run python -m gpu.local_default_train
```

After that works, you can use the shared launcher
[../slurm_launch.py](../slurm_launch.py) for experiment grids. For a new cluster,
open that file and edit the `Slurm customization block` near the top.

The most likely fields to change are:

```python
QUEUE_CONFIGS = {
    "gpu": {"account": None, "partition": "gpu", "gpu_request": "gres"},
}

ENV_SETUP_COMMANDS = ""  # Example: "module load cuda/12.1"

LOCAL_SLURM_CONFIG = {
    "repo_dir": "/path/to/assignments",
    "slurm_log_dir": "/path/to/slurm_logs",
    "temp_script_dir": "/path/to/temp_scripts",
    "uv_path": "uv",
    "uv_project_environment": "/path/to/.venv",
    "uv_cache_dir": "/path/to/uv_cache",
    "slurm_exclude": "",
}
```

Use your cluster's real partition/account names. If your cluster does not use
Slurm accounts, leave the queue's `account` as `None` and the launcher will omit
the `#SBATCH --account=...` line. If your cluster supports
`#SBATCH --gpus-per-task=...`, omit `gpu_request` or set it to
`"gpus-per-task"`. If it uses `#SBATCH --gres=gpu:...`, set `gpu_request` to
`"gres"`.
If your cluster requires environment setup before running Python, put those shell
commands in `ENV_SETUP_COMMANDS`; otherwise leave it as the empty string.

You can also leave individual `LOCAL_SLURM_CONFIG` entries as `None`; those
values then come from [../utils.py](../utils.py) and its environment-variable
overrides. This is usually enough if you already configured `CONFIG_SCRATCH_ROOT`
and run from the repo checkout you want Slurm jobs to use.

Then create a small Python launch file that imports the objects used in the
function call string:

```python
from slurm_launch import launch_job
from train import TrainConfig, train

if __name__ == "__main__":
    launch_job(
        "train(TrainConfig(num_train_sequences=600_000, run_name_suffix='slurm-gpu', wandb_online=False))",
        queue="gpu",
        gpus=1,
        mem=64,
        cpus=16,
    )
```

The launcher stages a short Python script and submits it with `sbatch`. The
function call string is evaluated in a script that imports your launch file, so
keep imports like `from train import TrainConfig, train` at module scope and
put submission calls behind the `__main__` guard to avoid resubmitting on import.

## This cluster: H100 / H200 / B200

`cluster_train.py` provides the same `launch_training_jobs(configs, ...)` pattern
as `modal_train.py`, using Slurm job arrays. On this cluster the default partition
is `mit_normal_gpu` (six-hour limit). It discovers GPU types with `sinfo` and
excludes CPU, L40S, and nonmatching GPU nodes. By default each task can run on
any H100, H200, or B200 node; Slurm chooses the type when scheduling. These GPUs
share one partition, so this is a union of GPU types, not separate partitions.
Use `gpu="H200"` for a single type, `gpu=("H100", "H200")` for a subset, or
`gpu="any"` (also `None`) for the default pool. Comma-separated types also work.

The cluster rewrites generic `--gres=gpu:1` requests to L40S. For a pool, the
launcher instead uses `--gpus-per-task=1` with nonmatching nodes excluded; for
a single type it uses a typed GRES. The array concurrency limit applies across
all eligible GPU types, and tasks in the same array may use different types. One array task trains one configuration on one GPU.

```python
from cluster_train import launch_training_jobs
from train import TrainConfig

RUNS = [
    TrainConfig(learning_rate=lr, run_name_suffix='cluster-lr-v1', wandb_online=False)
    for lr in (.0015, .003, .006)
]

if __name__ == '__main__':
    jobs = launch_training_jobs(
        RUNS,
        # Omit gpu to allow H100, H200, and B200.
        max_parallel_runs=2,
        dry_run=True,  # Remove after reviewing to submit the array.
    )
```

For A2, replace the Modal import with:

```python
from experiments.a2.cluster_launcher import config, launch_training_jobs

RUNS = [config(tokens=153_600_000, batch=128, learning_rate=.0015,
               run_name_suffix='a2-cluster-v1')]

if __name__ == '__main__':
    launch_training_jobs(RUNS, gpu='H200', max_parallel_runs=2)
```

The A2 factory preserves diagnostics, accumulation, and optimizer routing. It
uses local data under `DL_ALCHEMY_DATA_DIR` (or the `utils.py` default), reuses a
seed-42 cache when available, and otherwise lets each worker materialize its
shuffled prefix from the base dataset. Prepare reusable prefixes with
`experiments.a2.data.stage_prefix` and pass explicit `train_dataset` overrides
if repeated preparation becomes expensive. No data download or preprocessing
runs on the login node during submission. Download the data before training,
as described above. For diagnostics, retain `wandb_online=True` and configure
local W&B credentials; setting it false disables the shared trainer's probes.

The calling Python interpreter, environment, repository, dataset, and output
paths must be accessible on compute nodes. The launcher uses that interpreter
directly, without `uv sync` on workers. Configurations are frozen in JSON;
custom builder/logger modules must remain importable from the shared checkout.
Builder kwargs must be JSON serializable (tuples become lists). Keep the checkout
unchanged while jobs are queued/running. Use distinct run-name suffixes for
custom settings: completed model directories are skipped unless `force_run=True`,
and training checkpoints resume according to `TrainConfig`.

**B200 environment:** the course's pinned PyTorch 2.6 environment is suitable
for H100/H200, but B200 needs a Blackwell-capable PyTorch/CUDA build. For example,
create a separate shared environment with PyTorch 2.7.1 and CUDA 12.8 wheels:

```bash
uv venv --python 3.11 .venv-cluster
uv pip install --python .venv-cluster/bin/python torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-cluster/bin/python numpy wandb==0.17.3 \
    huggingface-hub matplotlib requests sentencepiece tqdm
.venv-cluster/bin/python -m your_experiment_module
```

Use this environment for all GPUs when comparing results across hardware. Do not
run `uv sync` against it: the root lockfile would restore the course's torch pin.
The launcher rejects pools containing B200 from older builds (including the
default pool); choose `gpu=("H100", "H200")` with the course environment. The worker verifies
CUDA and the actual GPU type before training. See the
[PyTorch Blackwell support announcement](https://pytorch.org/blog/pytorch-2-7/).

Optional submission arguments include `partition`, `account`, `qos`, `gpus`,
`cpus` (default 8), `mem` (64 GiB host RAM), `time_limit` (`06:00:00`), and
`dependency` (e.g. `afterok:12345`). The default account is your Slurm default.
Multiple GPUs use the existing trainer's single-node DataParallel behavior.
`max_parallel_runs` limits concurrently running configurations, not GPUs.

A dry run writes `configs.json`, `job.sh`, and the exact submission argument list
`submit.json` beneath `temp_scripts/cluster-*`, prints the command, and returns
job metadata without submitting. It does not verify GPU runtime compatibility.
An actual submission returns `slurm_job_id` values such as `12345_0` and
`log_path` for each run. Logs default to `slurmjobs/12345_0.out`.

```bash
squeue -u "$USER"
sacct -j 12345 --format=JobID,State,Elapsed,ExitCode
scancel 12345
```

The job forwards TERM/INT to Python and requests TERM two minutes before the
wall-time limit so the existing trainer can checkpoint at an update boundary.
There is no automatic resubmission; resubmit the same config to resume. A forced
kill before checkpoint completion can lose progress since the last checkpoint.
Both staging and logs use the existing `utils.py` path overrides and are ignored
by Git. Keep staged files until the array finishes.
