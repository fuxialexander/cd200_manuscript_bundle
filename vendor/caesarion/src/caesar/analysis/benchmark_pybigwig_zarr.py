from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyBigWig
import zarr


def generate_test_data(size=1_000_000):
    """Generate random genomic signal data for testing."""
    return np.random.random(size)


def write_bigwig(data, filename):
    """Write data to a bigWig file."""
    bw = pyBigWig.open(filename, "w")
    # Add a dummy header
    bw.addHeader([("chr1", len(data))], maxZooms=0)
    # Write the data
    bw.addEntries("chr1", 0, values=data.tolist(), span=1, step=1)
    bw.close()


def read_bigwig(filename):
    """Read data from a bigWig file."""
    bw = pyBigWig.open(filename)
    data = np.array(bw.values("chr1", 0, len(bw.intervals("chr1"))))
    bw.close()
    return data


def write_zarr_chunk(args):
    """Write a chunk of data to zarr array."""
    data_chunk, filename, start, end = args
    z = zarr.open(filename, mode="r+")
    z[start:end] = data_chunk
    return True


def write_zarr(data, filename, parallel=False):
    """Write data to a zarr array."""
    # Create zarr array first
    z = zarr.open(filename, mode="w", shape=data.shape, dtype=data.dtype)

    if not parallel:
        z[:] = data
    else:
        # Split into 10 chunks
        n_chunks = 10
        chunk_size = len(data) // n_chunks
        chunks = []

        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_chunks - 1 else len(data)
            chunks.append((data[start:end], filename, start, end))

        # Use all available cores
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            list(executor.map(write_zarr_chunk, chunks))


def read_zarr(filename):
    """Read data from a zarr array."""
    z = zarr.open(filename, mode="r")
    return z[:]


def read_zarr_chunk(args):
    """Read a chunk of data from zarr array."""
    filename, start, end = args
    z = zarr.open(filename, mode="r")
    return z[start:end]


def parallel_read_zarr(filename):
    """Read zarr array using multiple processes with 10 chunks."""
    z = zarr.open(filename, mode="r")
    size = len(z)

    # Split into 10 chunks
    n_chunks = 10
    chunk_size = size // n_chunks
    chunks = [
        (filename, i * chunk_size, (i + 1) * chunk_size if i < n_chunks - 1 else size)
        for i in range(n_chunks)
    ]

    # Use all available cores
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(executor.map(read_zarr_chunk, chunks))

    return np.concatenate(results)


def run_benchmark(data_size=10_000_000):
    """Run benchmark comparing zarr and bigWig performance."""
    print(f"Running benchmark with {data_size:,} data points")
    print("-" * 50)

    # Generate test data
    print("Generating test data...")
    data = generate_test_data(data_size)

    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    bw_file = os.path.join(temp_dir, "test.bw")
    zarr_file = os.path.join(temp_dir, "test.zarr")
    zarr_file_parallel = os.path.join(temp_dir, "test_parallel.zarr")

    # Test BigWig write
    start_time = time.time()
    write_bigwig(data, bw_file)
    bw_write_time = time.time() - start_time
    print(f"BigWig write time: {bw_write_time:.2f} seconds")

    # Test BigWig read
    start_time = time.time()
    read_bigwig(bw_file)
    bw_read_time = time.time() - start_time
    print(f"BigWig read time: {bw_read_time:.2f} seconds")

    # Test Zarr sequential write
    start_time = time.time()
    write_zarr(data, zarr_file, parallel=False)
    zarr_write_time = time.time() - start_time
    print(f"Zarr sequential write time: {zarr_write_time:.2f} seconds")

    # Test Zarr parallel write
    start_time = time.time()
    write_zarr(data, zarr_file_parallel, parallel=True)
    zarr_parallel_write_time = time.time() - start_time
    print(f"Zarr parallel write time: {zarr_parallel_write_time:.2f} seconds")

    # Test Zarr sequential read
    start_time = time.time()
    read_zarr(zarr_file)
    zarr_read_time = time.time() - start_time
    print(f"Zarr sequential read time: {zarr_read_time:.2f} seconds")

    # Test Zarr parallel read
    start_time = time.time()
    parallel_read_zarr(zarr_file)
    zarr_parallel_read_time = time.time() - start_time
    print(
        f"Zarr parallel read time ({mp.cpu_count()} cores): {zarr_parallel_read_time:.2f} seconds"
    )

    # Clean up
    os.remove(bw_file)
    import shutil

    shutil.rmtree(zarr_file)
    shutil.rmtree(zarr_file_parallel)
    os.rmdir(temp_dir)

    return {
        "bw_write": bw_write_time,
        "bw_read": bw_read_time,
        "zarr_write": zarr_write_time,
        "zarr_parallel_write": zarr_parallel_write_time,
        "zarr_read": zarr_read_time,
        "zarr_parallel_read": zarr_parallel_read_time,
    }


if __name__ == "__main__":
    # Run benchmark with different data sizes
    for size in [100_000, 1_000_000, 10_000_000]:
        print(f"\nRunning benchmark with {size:,} data points")
        print("=" * 60)
        results = run_benchmark(size)
        print("\nSummary:")
        print(
            f"BigWig write/read ratio: {results['bw_write'] / results['bw_read']:.2f}"
        )
        print(
            f"Zarr sequential write/read ratio: {results['zarr_write'] / results['zarr_read']:.2f}"
        )
        print(
            f"Zarr parallel write speedup: {results['zarr_write'] / results['zarr_parallel_write']:.2f}x"
        )
        print(
            f"Zarr parallel read speedup: {results['zarr_read'] / results['zarr_parallel_read']:.2f}x"
        )
        print("=" * 60)
