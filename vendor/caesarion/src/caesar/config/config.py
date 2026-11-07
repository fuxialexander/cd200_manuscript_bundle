from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import MISSING

T = TypeVar("T")


def default_cfg():
    return field(default_factory=T)


@dataclass
class Config:
    data_key: str = MISSING
    data_type: str = MISSING
    integrated: bool = False
    root_path: str = MISSING
    annotation_dir: str = MISSING
    output_dir: str = MISSING
    labels_column: str = MISSING
    barcode_column: str = MISSING
    peak_file: str = MISSING
    fragment_file: str = MISSING
    fragment_zarr_file: str = MISSING
    max_fragment_per_cell: int = 500
    chunk_size: int = 2000000
    n_insertions: int = 10000000
    assembly: str = "hg38"
    gencode_version: int = 44
    n_jobs: int = 64
    tempdir: str = "/tmp"
    seed: int = 42
    specific_peak_names: str = "peaks_q0.01"
    merged_peak_names: str = "peaks_q0.01_tissue"
    assembly_to_liftover_from: str | None = None
    sorted_by_barcode: bool = False
    n_cells: int | None = None
    skip: bool = False
    skip_exist: bool = False
    regenerate_peak_count_sparse: bool = False


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


def load_config(config_name):
    # Initialize Hydra to load the configuration
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="./", version_base="1.3")
    cfg = hydra.compose(config_name=config_name)
    return cfg
