from __future__ import annotations

import os
import subprocess
from sys import argv

from gcell._settings import get_setting
from gcell.dna.genome import ChromGap, pandas_to_tabix_region


# Function to get heterochromatin regions for an assembly
def get_heterochromatin_regions(assembly):
    annotation_dir = get_setting("annotation_dir")
    cg = ChromGap(assembly=assembly, annotation_dir=annotation_dir)
    df = cg.get_heterochromatin(return_tabix=False)

    # Convert to tabix format manually since the dataframe uses Chromosome/Start/End
    # instead of chrom/start/end
    if df.empty:
        return ""
    return pandas_to_tabix_region(
        df, chrom_col="Chromosome", start_col="Start", end_col="End"
    )


# Function to index the fragment file using tabix
def index_fragment_file(fragment_file):
    cmd = f"tabix -p bed {fragment_file}"
    subprocess.run(cmd, shell=True, check=True)


# Function to count fragments in heterochromatin regions
def count_fragments_in_heterochromatin(fragment_file, regions):
    cmd = f"tabix {fragment_file} {regions} | wc -l"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
    return int(result.stdout.strip())


# Main function to determine the correct genome assembly
def determine_genome_assembly(fragment_file, root_path, only_modern_assemblies=False):
    """
    Determine genome assembly for fragment file.

    Args:
        fragment_file: Path to fragment file
        root_path: Root path for relative fragment files
        only_modern_assemblies: If True, only check hg38/mm10 (not hg19/mm9)

    Returns:
        tuple: (best_assembly, fragment_count) or (None, None) if not hg38/mm10
    """
    if only_modern_assemblies:
        assemblies = ["hg38", "mm10"]
    else:
        assemblies = ["hg19", "hg38", "mm10"]

    if not os.path.exists(fragment_file):
        fragment_file = os.path.join(root_path, "raw", fragment_file)

    # Index the fragment file
    if not os.path.exists(fragment_file + ".tbi"):
        index_fragment_file(fragment_file)

    # Determine fragment count for each assembly
    fragment_counts = {}
    for assembly in assemblies:
        try:
            regions = get_heterochromatin_regions(assembly)
            if not regions or regions.strip() == "":
                print(f"Warning: No heterochromatin regions found for assembly {assembly}")
                continue
            count = count_fragments_in_heterochromatin(fragment_file, regions)
            fragment_counts[assembly] = count
        except Exception as e:
            print(f"Error processing assembly {assembly}: {e}")
            import traceback
            traceback.print_exc()

    # Check if we have any successful results
    if not fragment_counts:
        raise ValueError(
            "Failed to process any assemblies. Check that annotation_dir is set correctly "
            "and that fragment file is properly formatted."
        )

    # Find the assembly with the smallest fragment count
    best_assembly = min(fragment_counts, key=fragment_counts.get)

    # If only_modern_assemblies is True and best assembly is not hg38/mm10, return None
    if only_modern_assemblies and best_assembly not in ["hg38", "mm10"]:
        print(f"Skipping: Best assembly {best_assembly} is not hg38/mm10")
        print(f"Fragment counts: {fragment_counts}")
        return None, None

    print(
        f"Best assembly: {best_assembly} with fragment count: {fragment_counts[best_assembly]}"
    )
    print(fragment_counts)
    return best_assembly, fragment_counts[best_assembly]


# Example usage
if __name__ == "__main__":
    root_path = "/path/to/root"  # Replace with actual path
    fragment_file = argv[1]
    best_assembly, fragment_count = determine_genome_assembly(fragment_file, root_path)
    print(f"Best assembly: {best_assembly} with fragment count: {fragment_count}")
