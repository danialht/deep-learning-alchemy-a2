"""A2 configs for shared cluster storage; submits through cluster_train."""
from dataclasses import replace

from cluster_train import launch_training_jobs
from data import DEFAULT_DATASET_DIR_NAME, DEFAULT_DATA_SEED
from experiments.a2.baseline import config as baseline_config
from pathlib import Path
from utils import DATA_DIR


def config(*, tokens=153_600_000, batch=64, diagnostics=True, **overrides):
    """Same config interface as the A2 Modal launcher, with local data paths.

    The shared loader reuses a seed-42 dataset if present, otherwise materializes
    the globally shuffled prefix on the worker. Explicit datasets may be supplied.
    """
    seed = overrides.pop('data_seed', DEFAULT_DATA_SEED)
    if seed != DEFAULT_DATA_SEED:
        raise ValueError('A2 comparisons require data_seed=42')
    root = Path(DATA_DIR) / DEFAULT_DATASET_DIR_NAME
    seeded = Path(DATA_DIR) / f'{DEFAULT_DATASET_DIR_NAME}_ds42'
    train_path = seeded / 'train' if (seeded / 'train' / 'metadata.json').is_file() else root / 'train'
    c = baseline_config(train_path, root / 'val', tokens=tokens, batch=batch, diagnostics=diagnostics)
    if overrides.get('optimizer_name') in {'adamh', 'hyperball', 'muon'}:
        overrides.setdefault('optimizer_builder', 'experiments.a2.optimizers:build_optimizer')
        if overrides['optimizer_name'] in {'adamh', 'hyperball'}:
            overrides.setdefault('weight_decay', 0.)
    return replace(c, **overrides)
