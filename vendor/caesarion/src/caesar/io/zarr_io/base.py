from __future__ import annotations

import logging
import re

import numpy as np

if not hasattr(np, "long"):
    np.long = np.longlong


import pandas as pd
import zarr
from tqdm import tqdm
from zarr.codecs import BloscCodec

from ..interface import GenomicIO
from .. import read_peaks  # read_peaks is defined before zarr_io import in io/__init__.py

COMPRESSOR = BloscCodec(cname="zstd", clevel=4, shuffle="bitshuffle")


class ZarrIO(GenomicIO):
    """
    Abstract base class for Zarr-based genomic data I/O, inheriting from GenomicIO.

    Subclasses must implement the abstract methods defined in GenomicIO,
    specifically focusing on Zarr interactions.
    """

    def __init__(
        self,
        path: str,
        mode: str = "r",
        dtype: str = "float32",
        chroms: list[str] | None = None,
    ):
        """
        Initialize base ZarrIO object.

        Args:
            path (str): Path to the Zarr file/directory.
            mode (str): Mode for opening Zarr files ('r': read-only, 'a': read/write).
            dtype (str): Data type for the genomic data.
        """
        super().__init__(path, mode, dtype)

        self._setup_dataset()
        self._determine_assembly()
        if chroms is not None:
            self._active_chroms = set(chroms).intersection(self.data_chroms)
            if len(self._active_chroms) == 0:
                raise ValueError(f"No valid chromosomes found in {chroms}")
        else:
            self._active_chroms = self.data_chroms

        # --- Call abstract methods (to be implemented by subclasses) ---
        # Subclasses MUST call these in their __init__ after super().__init__
        # self._init_data_structure()
        # self._init_chunking() # Often called within _init_data_structure

    @property
    def data_chroms(self) -> list[str]:
        """
        Abstract property: Returns a sorted list of chromosome names
        present in the underlying data source. Must be implemented by subclasses.
        """
        try:
            return list(self.dataset["chrs"].keys())
        except KeyError:
            logging.error("'chrs' group not found in Zarr dataset.")
            return []

    def _setup_dataset(self) -> None:
        try:
            # Use self.load_zarr for consistency
            self.dataset = self.load_zarr(self.path, self.mode)
        except Exception as e:
            # Use FileNotFoundError or specific Zarr errors if possible
            logging.warning(f"Could not load Zarr dataset at {self.path}: {e}")
            self.dataset = None  # Mark as None if loading fails

    def _init_chunking(
        self,
        first_chrom: str | None = None,
        dataset: zarr.Group | None = None,
    ) -> None:
        """
        Initialize chunking-related attributes based on the Zarr structure.
        Should be called by `_init_data_structure` in subclasses.

        Args:
            first_chrom (str | None): A chromosome known to exist in the dataset.
                                       If None, attempts to use the first available chrom.
        """
        if dataset is None:
            dataset = self.dataset
        if not first_chrom:
            # Try to get the first chromosome if not provided
            try:
                first_chrom = next(iter(dataset["chrs"]))
            except (StopIteration, KeyError):
                logging.warning("Cannot determine first chromosome for chunking.")
                self.dir_chunked = None
                self.chunk_size = None
                return

        if first_chrom not in dataset["chrs"]:
            logging.warning(
                f"Provided first_chrom '{first_chrom}' not found in dataset."
            )
            # Attempt fallback? Or fail? Let's fail for now.
            self.dir_chunked = None
            self.chunk_size = None
            return

        # Detect chunking type and size
        chrom_ref = dataset["chrs"][first_chrom]
        if isinstance(chrom_ref, zarr.Group):
            # Directory-based chunking (old format)
            self.dir_chunked = True
            try:
                # Check if chunks exist
                if not chrom_ref:
                    raise KeyError("No chunks found")
                # Assume chunk_0 exists and get its shape
                self.chunk_size = chrom_ref["chunk_0"].shape[0]
            except (KeyError, IndexError, AttributeError) as e:
                logging.warning(
                    f"Could not determine chunk size for directory chunked format: {e}"
                )
                self.chunk_size = None  # Indicate failure
        elif isinstance(chrom_ref, zarr.Array):
            # Internal chunking (new format)
            self.dir_chunked = False
            try:
                self.chunk_size = chrom_ref.chunks[0]
            except (IndexError, AttributeError) as e:
                logging.warning(
                    f"Could not determine chunk size for array chunked format: {e}"
                )
                self.chunk_size = None  # Indicate failure
        else:
            logging.warning(
                f"Unexpected type for chromosome reference: {type(chrom_ref)}"
            )
            self.dir_chunked = None
            self.chunk_size = None

        # Note: The chrom_n_chunks property in GenomicIO calculates based on self.chrom_sizes and self.chunk_size.
        # Subclasses might need to refine self.chrom_sizes in _init_data_structure
        # based on the actual data present in dataset['chrs'].

    @staticmethod
    def load_zarr(path: str, mode="r") -> zarr.Group:
        """
        Load a Zarr group from the specified path.

        Args:
            path: The file system path to the Zarr group/store.
            mode: File opening mode ('r', 'r+', 'a', 'w', 'w-').

        Returns:
            The loaded Zarr group object.

        Raises:
            FileNotFoundError: If the path does not exist and mode is read-based.
            Exception: For other Zarr-related errors during loading.
        """
        try:
            # Use zarr.open for flexibility (handles groups and arrays)
            # Specify return type hint if possible, usually Group for these IO classes
            z = zarr.open(path, mode=mode)
            if not isinstance(z, zarr.Group):
                # If we expect a group but get an array, handle it
                logging.warning(f"Expected Zarr Group at {path} but found Array.")
                # Depending on need, either raise error or adapt
                raise TypeError(f"Expected Zarr Group at {path}, found {type(z)}")
            return z
        except FileNotFoundError:
            logging.error(f"Zarr path not found: {path}")
            raise
        except Exception as e:
            logging.error(f"Error loading Zarr from {path}: {e}")
            raise

    def _load_attr(self, key) -> None:
        """Load attributes from the backend storage."""
        if self.dataset is None:
            logging.error("Cannot load attributes: Zarr dataset not loaded.")
            return
        return self.dataset.attrs.get(key)

    def _save_attr(self, key, value) -> None:
        """Save attributes to the backend storage."""
        if self.dataset is None:
            logging.error("Cannot save attributes: Zarr dataset not loaded.")
            return
        if self.mode == "r":
            logging.warning("Cannot save attributes: Zarr dataset is read-only.")
            return
        self.dataset.attrs[key] = value

    @classmethod
    def detect_store_type(cls, path: str | Path) -> str | None:
        """Detect Zarr store type from path without instantiating.

        First checks .zattrs for 'zarr_type' key, then falls back to
        structural detection using detect_zarr_type.

        Args:
            path: Path to the zarr store

        Returns
        -------
        str | None
            Store type identifier (e.g., "celltype_dense", "single_sparse"),
            or None if the store type cannot be determined or path is not a Zarr store.
        """
        try:
            z = zarr.open_group(str(path), mode="r")
            # First, check .zattrs for explicit zarr_type (most reliable)
            if "zarr_type" in z.attrs:
                return z.attrs["zarr_type"]

            # Fallback to structural detection
            from caesar.io.zarr_io import detect_zarr_type
            return detect_zarr_type(path, mode="r")
        except Exception:
            # Not a Zarr store or cannot be opened
            return None

    def _get_store_type(self) -> str | None:
        """Get store type from Zarr metadata or structural detection.

        First checks .zattrs for 'zarr_type' key, then falls back to
        structural detection using detect_zarr_type.

        Returns
        -------
        str | None
            Store type identifier (e.g., "celltype_dense", "single_sparse"),
            or None if the store type cannot be determined.
        """
        if self.dataset is None:
            return None

        # First, check .zattrs for explicit zarr_type (most reliable)
        if "zarr_type" in self.dataset.attrs:
            return self.dataset.attrs["zarr_type"]

        # Fallback to structural detection
        try:
            from caesar.io.zarr_io import detect_zarr_type
            return detect_zarr_type(self.path, mode="r")
        except Exception as e:
            logging.debug(f"Could not determine zarr store type: {e}")
            return None

    def _write_peaks_internal(
        self,
        group: zarr.Group,
        peaks: str | pd.DataFrame,
        name: str,
        columns_to_include: list[str] | None = None,
        compressors=COMPRESSOR,
        dtype="i4",
        overwrite=False,
    ):
        """Internal helper to write peaks to a specified Zarr group."""
        if self.mode == "r":
            logging.error("Cannot write peaks: Zarr dataset is read-only.")
            return False
        if group is None:
            logging.error("Cannot write peaks: Zarr group is None.")
            return

        peaks_df = read_peaks(peaks)
        peaks_df = peaks_df.sort_values(["Chromosome", "Start"])

        # Determine columns to write
        if columns_to_include is None:
            # Default columns, ensure they exist
            cols_to_write = ["Start", "End"]
            missing_cols = [col for col in cols_to_write if col not in peaks_df.columns]
            if missing_cols:
                raise ValueError(
                    f"Default columns {missing_cols} not found in peaks DataFrame."
                )
        else:
            cols_to_write = columns_to_include
            missing_cols = [col for col in cols_to_write if col not in peaks_df.columns]
            if missing_cols:
                raise ValueError(
                    f"Specified columns {missing_cols} not found in peaks DataFrame."
                )

        regions_group = group.require_group(f"{name}/regions", overwrite=overwrite)

        for chrom in tqdm(
            self.data_chroms, desc=f"Writing {name} chromosomes", leave=False
        ):
            chrom_peaks = peaks_df[peaks_df["Chromosome"] == chrom]
            chrom_peaks_data = chrom_peaks[cols_to_write]

            # Cast data to dtype before creating array (zarr v3 compatibility)
            ds = regions_group.create_array(
                chrom,
                data=chrom_peaks_data.values.astype(dtype),
                chunks=(1000, len(cols_to_write)),
                compressors=compressors,
                overwrite=overwrite,
            )
            ds.attrs["n_regions"] = len(chrom_peaks_data)
            ds.attrs["columns"] = cols_to_write

    def _get_peaks_internal(
        self, group: zarr.Group, name: str
    ) -> pd.DataFrame | None:
        """Internal helper to get peaks from a specified Zarr group."""
        if group is None or f"{name}/regions" not in group:
            logging.info(f"No peaks found for name '{name}' in the provided group.")
            return None

        regions_group = group[f"{name}/regions"]
        peak_dfs = []
        chroms = list(regions_group.keys())

        # intersect with self.chroms
        chroms = [chrom for chrom in chroms if chrom in self.chroms]

        # Sort chromosomes numerically then alphabetically
        chroms = sorted(
            chroms,
            key=lambda x: (
                int(re.search(r"\d+", x).group())
                if re.search(r"\d+", x)
                else float("inf"),
                x,
            ),
        )

        columns = None
        for chrom in chroms:
            chrom_data = regions_group[chrom][:]
            if columns is None:  # Get columns from the first chromosome's attributes
                columns = regions_group[chrom].attrs.get("columns", ["Start", "End"])

            if chrom_data.size > 0:  # Check if data exists before creating DataFrame
                df = pd.DataFrame(chrom_data, columns=columns)
                df["Chromosome"] = chrom
                peak_dfs.append(df)
            else:
                logging.debug(
                    f"No peak data found for chromosome {chrom} under '{name}'."
                )

        if not peak_dfs:
            logging.warning(f"No peak data loaded for any chromosome under '{name}'.")
            return None

        full_peaks_df = pd.concat(peak_dfs).reset_index(drop=True)

        # Note: Peaks are already sorted when written (in _write_peaks_internal),
        # and concatenation preserves order per chromosome. No need to sort again here
        # as it would be expensive for large peak sets. The order matches what's in the matrix.

        # Ensure standard columns exist and have correct dtype if possible
        if "Start" in full_peaks_df:
            full_peaks_df["Start"] = full_peaks_df["Start"].astype(np.int64)
        if "End" in full_peaks_df:
            full_peaks_df["End"] = full_peaks_df["End"].astype(np.int64)

        # Ensure 'Chromosome' is the first column if it exists
        if "Chromosome" in full_peaks_df:
            cols = ["Chromosome"] + [col for col in columns if col != "Chromosome"]
            full_peaks_df = full_peaks_df[cols]

        return full_peaks_df

    def _remove_peaks_internal(self, group: zarr.Group, name: str):
        """Internal helper to remove a peaks group from a specified Zarr group."""
        if self.mode == "r":
            logging.error("Cannot remove peaks: Zarr dataset is read-only.")
            return False
        if group is None:
            logging.error("Cannot remove peaks: Zarr group is None.")
            return False
        peak_group_path = f"{name}"
        if peak_group_path in group:
            try:
                del group[peak_group_path]
                logging.info(f"Successfully removed peak group '{peak_group_path}'.")
                return True
            except Exception as e:
                logging.error(f"Error removing peak group '{peak_group_path}': {e}")
                return False
        else:
            logging.warning(f"Peak group '{peak_group_path}' not found for removal.")
            return False
