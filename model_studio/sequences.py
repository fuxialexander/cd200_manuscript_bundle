"""Sequence input sources and canonical one-hot encoding."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from numcodecs import get_codec
import zarr

from .domain import SequenceRecord


_BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
_REGION = re.compile(r"^([^:]+):(\d[\d,]*)-(\d[\d,]*)$")


def normalize_sequence(raw: str) -> str:
    lines = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.startswith(">")
    ]
    sequence = re.sub(r"\s+", "", "".join(lines)).upper()
    if not sequence:
        raise ValueError("No DNA sequence was provided")
    invalid = sorted(set(sequence) - set("ACGTN"))
    if invalid:
        raise ValueError(f"Unsupported DNA characters: {', '.join(invalid)}")
    if len(sequence) < 32:
        raise ValueError("Sequence must contain at least 32 bases")
    return sequence


def one_hot_encode(sequence: str) -> np.ndarray:
    encoded = np.zeros((len(sequence), 4), dtype=np.float32)
    for position, base in enumerate(sequence):
        index = _BASE_TO_INDEX.get(base)
        if index is not None:
            encoded[position, index] = 1.0
    return encoded


def one_hot_to_sequence(array: np.ndarray) -> str:
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"Expected sequence array shaped (L, 4), got {array.shape}")
    letters = np.array(list("ACGT"))
    called = array.max(axis=1) > 0
    indices = array.argmax(axis=1)
    return "".join(letters[indices[row]] if called[row] else "N" for row in range(len(array)))


def _read_local_zarr_v2_slice(
    array_path: Path,
    start: int,
    end: int,
) -> tuple[np.ndarray, int]:
    """Read a local 2D zarr-v2 slice without zarr-v3 directory enumeration."""
    metadata = json.loads((array_path / ".zarray").read_text())
    shape = tuple(int(value) for value in metadata["shape"])
    chunks = tuple(int(value) for value in metadata["chunks"])
    if len(shape) != 2 or len(chunks) != 2:
        raise ValueError(f"Expected a 2D sequence zarr, got shape={shape}")
    if chunks[1] != shape[1]:
        raise ValueError(
            "Sequence zarr must store the full one-hot width in each chunk"
        )
    dtype = np.dtype(metadata["dtype"])
    output = np.full(
        (end - start, shape[1]),
        metadata.get("fill_value", 0),
        dtype=dtype,
    )
    compressor = metadata.get("compressor")
    codec = get_codec(compressor) if compressor else None
    filters = metadata.get("filters")
    if filters:
        raise ValueError("Filtered zarr-v2 arrays are not supported by the fast reader")
    separator = metadata.get("dimension_separator", ".")
    chunk_length = chunks[0]
    first_chunk = start // chunk_length
    last_chunk = (end - 1) // chunk_length
    for chunk_index in range(first_chunk, last_chunk + 1):
        chunk_file = array_path / f"{chunk_index}{separator}0"
        if not chunk_file.exists():
            continue
        raw = chunk_file.read_bytes()
        decoded = codec.decode(raw) if codec else raw
        chunk = np.frombuffer(decoded, dtype=dtype)
        rows = chunk.size // shape[1]
        chunk = chunk.reshape((rows, shape[1]), order=metadata.get("order", "C"))
        genomic_start = chunk_index * chunk_length
        source_start = max(start, genomic_start)
        source_end = min(end, genomic_start + rows)
        if source_end <= source_start:
            continue
        output[
            source_start - start : source_end - start
        ] = chunk[
            source_start - genomic_start : source_end - genomic_start
        ]
    return output, shape[0]


def _read_local_zarr_v2_1d_slice(
    array_path: Path,
    start: int,
    end: int,
) -> tuple[np.ndarray, int]:
    """Read a local 1D zarr-v2 slice without importing the full IO stack."""
    metadata = json.loads((array_path / ".zarray").read_text())
    shape = tuple(int(value) for value in metadata["shape"])
    chunks = tuple(int(value) for value in metadata["chunks"])
    if len(shape) != 1 or len(chunks) != 1:
        raise ValueError(f"Expected a 1D signal zarr, got shape={shape}")
    if start < 0 or end <= start or end > shape[0]:
        raise ValueError(
            f"Slice [{start}, {end}) is outside signal length {shape[0]}"
        )
    dtype = np.dtype(metadata["dtype"])
    output = np.full(
        end - start,
        metadata.get("fill_value", 0),
        dtype=dtype,
    )
    compressor = metadata.get("compressor")
    codec = get_codec(compressor) if compressor else None
    filters = metadata.get("filters")
    if filters:
        raise ValueError("Filtered zarr-v2 arrays are not supported by the fast reader")
    separator = metadata.get("dimension_separator", ".")
    chunk_length = chunks[0]
    first_chunk = start // chunk_length
    last_chunk = (end - 1) // chunk_length
    for chunk_index in range(first_chunk, last_chunk + 1):
        chunk_file = array_path / str(chunk_index)
        if separator != ".":
            chunk_file = array_path / str(chunk_index).replace(".", separator)
        if not chunk_file.exists():
            continue
        raw = chunk_file.read_bytes()
        decoded = codec.decode(raw) if codec else raw
        chunk = np.frombuffer(decoded, dtype=dtype)
        genomic_start = chunk_index * chunk_length
        source_start = max(start, genomic_start)
        source_end = min(end, genomic_start + len(chunk))
        if source_end <= source_start:
            continue
        output[source_start - start : source_end - start] = chunk[
            source_start - genomic_start : source_end - genomic_start
        ]
    return output, shape[0]


class SequenceSource(ABC):
    @abstractmethod
    def load(self) -> SequenceRecord:
        """Return a validated sequence record."""


class TextSequenceSource(SequenceSource):
    def __init__(self, text: str, label: str = "Pasted sequence") -> None:
        self.text = text
        self.label = label

    def load(self) -> SequenceRecord:
        return SequenceRecord(normalize_sequence(self.text), self.label)


class GenomeZarrSequenceSource(SequenceSource):
    def __init__(self, region: str, genome_zarr: str | Path) -> None:
        self.region = region
        self.genome_zarr = Path(genome_zarr).expanduser().resolve()

    @staticmethod
    def parse_region(region: str) -> tuple[str, int, int]:
        match = _REGION.match(region.replace(" ", ""))
        if not match:
            raise ValueError("Region must use chromosome:start-end syntax")
        chromosome = match.group(1)
        start = int(match.group(2).replace(",", ""))
        end = int(match.group(3).replace(",", ""))
        if end <= start:
            raise ValueError("Region end must be greater than start")
        return chromosome, start, end

    def load(self) -> SequenceRecord:
        if not self.genome_zarr.exists():
            raise FileNotFoundError(f"Genome zarr not found: {self.genome_zarr}")
        chromosome, start, end = self.parse_region(self.region)
        chromosome_store = self.genome_zarr / "chrs" / chromosome
        if not chromosome_store.exists():
            raise ValueError(f"Chromosome {chromosome!r} is absent from {self.genome_zarr}")
        metadata_path = chromosome_store / ".zarray"
        if metadata_path.exists():
            array, chromosome_length = _read_local_zarr_v2_slice(
                chromosome_store,
                start,
                end,
            )
        else:
            chromosome_array = zarr.open_array(str(chromosome_store), mode="r")
            chromosome_length = int(chromosome_array.shape[0])
            array = np.asarray(chromosome_array[start:end], dtype=np.float32)
        if start < 0 or end > chromosome_length:
            raise ValueError(
                f"{chromosome}:{start}-{end} is outside chromosome length "
                f"{chromosome_length}"
            )
        sequence = one_hot_to_sequence(array)
        return SequenceRecord(
            sequence=sequence,
            label=f"{chromosome}:{start}-{end}",
            chromosome=chromosome,
            start=start,
            end=end,
        )
