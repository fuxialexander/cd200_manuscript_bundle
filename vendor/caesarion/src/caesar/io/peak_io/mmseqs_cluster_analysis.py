#!/usr/bin/env python3
#
# Example usage of the script:
#
# This script can be run in two main modes:
# 1. Direct FASTA input: When you have a FASTA file with sequences to cluster.
# 2. BED file input: When you have genomic regions in a BED file and need to extract sequences from a genome FASTA.
#
# ---
#
# Example 1: Basic usage with a FASTA file
#
# ./mmseqs_cluster_analysis.py your_sequences.fa
#
# Example 2: Using a BED file to extract sequences
#
# ./mmseqs_cluster_analysis.py --bed your_regions.bed --genome your_genome.fa
#
# Example 3: Full-featured example with custom parameters for FASTA input
#
# ./mmseqs_cluster_analysis.py your_sequences.fa \
#   -o my_cluster_results \
#   --temp-dir /path/to/tmp \
#   --min-seq-id 0.5 \
#   --kmer-per-seq-scale 0.4 \
#   --coverage 0.8 \
#   --no-cleanup \
#   --skip-msa \
#   --msa-sequences 200 \
#   --bed-suffix _custom_suffix
#
# ---
#
# Argument explanation:
#
# Positional arugment:
#   input_fasta           : Input FASTA file containing DNA sequences to cluster.
#
# Mutually exclusive input group (one of these is required):
#   --bed BED             : BED file with genomic coordinates to extract sequences from.
#
# Required with --bed:
#   --genome GENOME       : Genome FASTA file (required when using --bed option).
#
# Optional arguments:
#   -o, --output-prefix   : Output prefix for all generated files (default: result_tmp).
#   --temp-dir            : Temporary directory for mmseqs clustering (default: /tmp).
#   --min-seq-id          : Minimum sequence identity for clustering (default: 0.3).
#   --kmer-per-seq-scale  : K-mer per sequence scale parameter (default: 0.3).
#   -c, --coverage        : Minimum coverage for clustering (default: 0.1).
#   --no-cleanup          : Keep temporary files (don't clean up).
#   --skip-msa            : Skip multiple sequence alignment generation for largest cluster.
#   --msa-sequences       : Number of sequences to use for MSA (default: 100).
#   --bed-suffix          : Suffix to add to BED output filename (default: none).
#
"""
Script to run mmseqs clustering on DNA sequences and analyze the results
to count singleton vs clustered sequences. mmseqs handles reverse complements automatically.
"""

import os
import sys
import subprocess
import argparse
from collections import defaultdict, Counter

try:
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq
    from Bio.Align.Applications import ClustalwCommandline
    from Bio import AlignIO
    BIOPYTHON_AVAILABLE = True
except ImportError:
    print("Warning: Biopython not available. MSA functionality will be skipped.")
    BIOPYTHON_AVAILABLE = False


def run_mmseqs_clustering(input_file, output_prefix, temp_dir="/tmp", min_seq_id=0.3,
                         kmer_per_seq_scale=0.3, coverage=0.1):
    """Run mmseqs clustering with specified parameters."""
    print(f"Running mmseqs clustering on {input_file}...")

    cmd = [
        "mmseqs", "easy-linclust", input_file, output_prefix, temp_dir,
        "--dbtype", "2",
        "--min-seq-id", str(min_seq_id),
        "--kmer-per-seq-scale", str(kmer_per_seq_scale),
        "-c", str(coverage)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("mmseqs clustering completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running mmseqs: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def analyze_clustering_results(cluster_file):
    """Analyze clustering results to identify singletons vs clustered sequences."""
    print(f"Analyzing clustering results from {cluster_file}...")

    # Dictionary to store cluster information: cluster_rep -> list of members
    clusters = defaultdict(list)

    # Read clustering results
    with open(cluster_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                representative, member = parts
                clusters[representative].append(member)

    print(f"Total number of clusters: {len(clusters)}")

    # Analyze cluster sizes and find largest cluster
    cluster_sizes = [len(members) for members in clusters.values()]
    cluster_size_counts = Counter(cluster_sizes)
    largest_cluster_size = max(cluster_sizes)

    # Find the largest cluster(s)
    largest_clusters = []
    for rep, members in clusters.items():
        if len(members) == largest_cluster_size:
            largest_clusters.append((rep, members))

    print(f"Largest cluster size: {largest_cluster_size}")
    print(f"Number of clusters with largest size: {len(largest_clusters)}")

    # Get all sequences and identify singletons
    all_sequences = set()
    singleton_sequences = set()

    for rep, members in clusters.items():
        all_sequences.update(members)
        # A sequence is singleton if it's the only member in its cluster
        if len(members) == 1:
            singleton_sequences.add(rep)

    total_sequences = len(all_sequences)
    num_singletons = len(singleton_sequences)
    num_clustered = total_sequences - num_singletons

    print(f"\n" + "="*60)
    print(f"CLUSTERING ANALYSIS RESULTS")
    print(f"="*60)
    print(f"Total sequences: {total_sequences:,}")
    print(f"Singleton sequences: {num_singletons:,}")
    print(f"Sequences in clusters (>1 member): {num_clustered:,}")
    print(f"Percentage singleton: {num_singletons/total_sequences*100:.2f}%")
    print(f"Percentage clustered: {num_clustered/total_sequences*100:.2f}%")

    # Show cluster size distribution summary
    print(f"\nCluster size distribution (top 10):")
    for size in sorted(cluster_size_counts.keys())[:10]:
        count = cluster_size_counts[size]
        print(f"  Size {size}: {count:,} clusters")

    if len(cluster_size_counts) > 10:
        print(f"  ... and {len(cluster_size_counts)-10} more size categories")

    return {
        'total_sequences': total_sequences,
        'singleton': num_singletons,
        'clustered': num_clustered,
        'largest_clusters': largest_clusters,
        'largest_cluster_size': largest_cluster_size,
        'singleton_sequences': singleton_sequences,
        'all_clusters': clusters
    }

def save_largest_cluster_sequences(largest_clusters, input_fasta_file, output_file):
    """Save sequences from the largest cluster to a FASTA file."""
    print(f"Saving largest cluster sequences to {output_file}...")

    if not largest_clusters:
        print("No largest clusters found!")
        return 0

    # Read all sequences from input FASTA
    sequences = {}
    current_header = None
    current_seq = []

    # Try input file first, then mmseqs output files
    source_file = input_fasta_file
    if not os.path.exists(source_file):
        # Use mmseqs all_seqs.fasta file which contains all sequences
        mmseqs_all_seqs = "result_tmp_all_seqs.fasta"
        if os.path.exists(mmseqs_all_seqs):
            source_file = mmseqs_all_seqs
        else:
            print("Error: Cannot find sequence data files.")
            return 0

    with open(source_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header:
                    sequences[current_header[1:]] = ''.join(current_seq)
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)

        # Process last sequence
        if current_header:
            sequences[current_header[1:]] = ''.join(current_seq)

    print(f"Read {len(sequences)} sequences from {source_file}")

    # Write largest cluster sequences to output file
    with open(output_file, 'w') as outf:
        # Use the first largest cluster (in case there are ties)
        rep, members = largest_clusters[0]

        outf.write(f">LARGEST_CLUSTER_INFO\n")
        outf.write(f"# Cluster size: {len(members)}\n")
        outf.write(f"# Representative: {rep}\n")
        outf.write(f"# Members: {len(members)}\n")
        outf.write(f"# Total sequences found in FASTA: {len(sequences)}\n\n")

        found_sequences = 0
        for i, member in enumerate(members, 1):
            if member in sequences:
                outf.write(f">MEMBER_{i:04d}::{member}\n")
                seq = sequences[member]
                # Write sequence with line breaks every 80 characters
                for j in range(0, len(seq), 80):
                    outf.write(seq[j:j+80] + '\n')
                outf.write('\n')
                found_sequences += 1
            else:
                print(f"Warning: Sequence {member} not found in FASTA file")

    print(f"Saved {found_sequences}/{len(largest_clusters[0][1])} sequences from largest cluster to {output_file}")

    return found_sequences

def create_msa_from_largest_cluster(largest_cluster_fasta, num_sequences=100):
    """Create Multiple Sequence Alignment from the first N sequences in largest cluster."""
    if not BIOPYTHON_AVAILABLE:
        print("Biopython not available. Skipping MSA creation.")
        return False, None
    
    print(f"Creating MSA from first {num_sequences} sequences in largest cluster...")
    
    # Read sequences from the largest cluster file
    sequences = []
    seq_count = 0
    
    try:
        for record in SeqIO.parse(largest_cluster_fasta, "fasta"):
            # Skip the info header
            if record.id.startswith("LARGEST_CLUSTER_INFO"):
                continue
            
            sequences.append(record)
            seq_count += 1
            
            if seq_count >= num_sequences:
                break
        
        if len(sequences) == 0:
            print("No sequences found in largest cluster file.")
            return False, None
        
        print(f"Found {len(sequences)} sequences for MSA")
        
        # Create a temporary input file for MSA
        temp_input = "temp_msa_input.fasta"
        temp_output_base = "temp_msa_output"
        
        with open(temp_input, "w") as f:
            SeqIO.write(sequences, f, "fasta")
        
        # Try different MSA methods
        msa_success = False
        
        # Method 1: Try using MUSCLE (if available)
        try:
            print("Trying MUSCLE for MSA...")
            cmd = ["muscle", "-in", temp_input, "-out", f"{temp_output_base}_muscle.fasta"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("MUSCLE MSA completed successfully")
                msa_success = True
                msa_output = f"{temp_output_base}_muscle.fasta"
            else:
                print(f"MUSCLE failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"MUSCLE not available or timed out: {e}")
        
        # Method 2: Try using MAFFT (if available)
        if not msa_success:
            try:
                print("Trying MAFFT for MSA...")
                cmd = ["mafft", "--auto", temp_input]
                with open(f"{temp_output_base}_mafft.fasta", "w") as outf:
                    result = subprocess.run(cmd, stdout=outf, stderr=subprocess.PIPE, text=True, timeout=300)
                if result.returncode == 0:
                    print("MAFFT MSA completed successfully")
                    msa_success = True
                    msa_output = f"{temp_output_base}_mafft.fasta"
                else:
                    print(f"MAFFT failed: {result.stderr}")
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"MAFFT not available or timed out: {e}")
        
        # Method 3: Simple pairwise alignment (fallback)
        if not msa_success:
            print("External MSA tools not available. Creating simple alignment info...")
            msa_output = f"{temp_output_base}_info.txt"
            
            with open(msa_output, "w") as f:
                f.write(f"Multiple Sequence Alignment Info for {len(sequences)} sequences\n")
                f.write("="*60 + "\n\n")
                
                # Calculate basic statistics
                seq_lengths = [len(seq.seq) for seq in sequences]
                f.write(f"Sequence count: {len(sequences)}\n")
                f.write(f"Sequence length range: {min(seq_lengths)} - {max(seq_lengths)}\n")
                f.write(f"Average length: {sum(seq_lengths)/len(seq_lengths):.1f}\n\n")
                
                # Write first few sequences as reference
                f.write("First 5 sequences:\n")
                for i, seq in enumerate(sequences[:5]):
                    f.write(f">{seq.id}\n")
                    f.write(f"{str(seq.seq)[:100]}...\n\n")
        
        # Copy result to final location
        final_msa_file = "largest_cluster_msa.fasta" if msa_success else "largest_cluster_msa_info.txt"
        if os.path.exists(msa_output):
            os.rename(msa_output, final_msa_file)
            print(f"MSA saved to: {final_msa_file}")
        
        # Cleanup temporary files
        for temp_file in [temp_input, f"{temp_output_base}_muscle.fasta", f"{temp_output_base}_mafft.fasta", f"{temp_output_base}_info.txt"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return msa_success, final_msa_file if msa_success else final_msa_file
        
    except Exception as e:
        print(f"Error creating MSA: {e}")
        return False, None

def save_all_regions_bed(singleton_sequences, all_clusters, output_bed_file):
    """Save singleton regions and cluster representatives as a BED file."""
    print(f"Saving singleton regions and cluster representatives to BED file: {output_bed_file}")
    
    bed_entries = []
    
    def parse_header_to_bed(seq_header, region_type, cluster_size=1):
        """Parse FASTA header to BED entry."""
        # Parse the header format: score::chr:start-end
        # Example: "6.322533337315476::chr1:803848-805896"
        try:
            parts = seq_header.split("::")
            if len(parts) >= 2:
                score = parts[0]
                location = parts[1]
                
                # Parse chromosome and coordinates
                if ":" in location and "-" in location:
                    chrom_part, coords_part = location.split(":", 1)
                    if "-" in coords_part:
                        start, end = coords_part.split("-", 1)
                        
                        # Clean up coordinates (remove any extra characters)
                        start = start.strip()
                        end = end.strip()
                        
                        # Validate that we can convert to integers
                        try:
                            start_int = int(start)
                            end_int = int(end)
                            score_float = float(score)
                            
                            return {
                                'chrom': chrom_part,
                                'start': start_int,
                                'end': end_int,
                                'score': score_float,
                                'type': region_type,
                                'cluster_size': cluster_size,
                                'original_header': seq_header
                            }
                        except ValueError:
                            pass
        except Exception:
            pass
        return None
    
    # Add singleton regions
    singleton_count = 0
    for seq_header in singleton_sequences:
        entry = parse_header_to_bed(seq_header, 'singleton', 1)
        if entry:
            bed_entries.append(entry)
            singleton_count += 1
    
    # Add cluster representatives (excluding singletons)
    representative_count = 0
    for representative, members in all_clusters.items():
        # Only add if it's not a singleton cluster (more than one member)
        if len(members) > 1:
            entry = parse_header_to_bed(representative, 'representative', len(members))
            if entry:
                bed_entries.append(entry)
                representative_count += 1
    
    # Sort by chromosome and start position
    bed_entries.sort(key=lambda x: (x['chrom'], x['start']))
    
    # Write BED file
    with open(output_bed_file, 'w') as f:
        # Write header
        for entry in bed_entries:
            f.write(f"{entry['chrom']}\t{entry['start']}\t{entry['end']}\t{entry['score']}\n")
    
    print(f"Saved {len(bed_entries)} total regions to {output_bed_file}")
    print(f"  - {singleton_count} singleton regions")
    print(f"  - {representative_count} cluster representative regions")
    
    # Print some statistics
    if bed_entries:
        chromosomes = set(entry['chrom'] for entry in bed_entries)
        scores = [entry['score'] for entry in bed_entries]
        
        print(f"All regions statistics:")
        print(f"  Chromosomes: {len(chromosomes)} ({', '.join(sorted(chromosomes)[:10])}{'...' if len(chromosomes) > 10 else ''})")
        print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"  Average score: {sum(scores)/len(scores):.3f}")
        
        # Statistics by type
        singleton_scores = [e['score'] for e in bed_entries if e['type'] == 'singleton']
        rep_scores = [e['score'] for e in bed_entries if e['type'] == 'representative']
        
        if singleton_scores:
            print(f"  Singleton scores: {min(singleton_scores):.3f} - {max(singleton_scores):.3f} (avg: {sum(singleton_scores)/len(singleton_scores):.3f})")
        if rep_scores:
            print(f"  Representative scores: {min(rep_scores):.3f} - {max(rep_scores):.3f} (avg: {sum(rep_scores)/len(rep_scores):.3f})")
    
    return len(bed_entries), singleton_count, representative_count

def cleanup_files(*files):
    """Clean up temporary files."""
    print("Cleaning up temporary files...")
    for file in files:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"  Removed {file}")
        except Exception as e:
            print(f"  Warning: Could not remove {file}: {e}")

def extract_sequences_from_bed(bed_file, genome_fasta, output_fasta):
    """Extract sequences from genome FASTA using BED coordinates with bedtools getfasta."""
    print(f"Extracting sequences from {genome_fasta} using {bed_file}...")

    cmd = [
        "bedtools", "getfasta",
        "-fi", genome_fasta,
        "-bed", bed_file,
        "-fo", output_fasta,
        "-name"  # Use BED name field as FASTA header
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Sequence extraction completed. Output: {output_fasta}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running bedtools getfasta: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: bedtools not found. Please install bedtools.")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run mmseqs clustering on DNA sequences and analyze the results "
                   "to count singleton vs clustered sequences. mmseqs handles reverse complements automatically.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.fa                              # Basic usage with FASTA input
  %(prog)s --bed regions.bed --genome genome.fa  # Extract sequences from BED coordinates
  %(prog)s input.fa -o my_results                # Custom output prefix
  %(prog)s input.fa --min-seq-id 0.5             # Higher sequence identity threshold
  %(prog)s input.fa --no-cleanup                 # Keep temporary files
  %(prog)s input.fa --skip-msa                   # Skip MSA generation
        """
    )

    # Input source arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "input_fasta",
        nargs="?",
        help="Input FASTA file containing DNA sequences to cluster"
    )

    input_group.add_argument(
        "--bed",
        help="BED file with genomic coordinates to extract sequences from"
    )

    parser.add_argument(
        "--genome",
        help="Genome FASTA file (required when using --bed option)"
    )

    # Optional arguments
    parser.add_argument(
        "-o", "--output-prefix",
        default="result_tmp",
        help="Output prefix for all generated files (default: result_tmp)"
    )

    parser.add_argument(
        "--temp-dir",
        default="/tmp",
        help="Temporary directory for mmseqs clustering (default: /tmp)"
    )

    parser.add_argument(
        "--min-seq-id",
        type=float,
        default=0.3,
        help="Minimum sequence identity for clustering (default: 0.3)"
    )

    parser.add_argument(
        "--kmer-per-seq-scale",
        type=float,
        default=0.3,
        help="K-mer per sequence scale parameter (default: 0.3)"
    )

    parser.add_argument(
        "-c", "--coverage",
        type=float,
        default=0.1,
        help="Minimum coverage for clustering (default: 0.1)"
    )

    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep temporary files (don't clean up)"
    )

    parser.add_argument(
        "--skip-msa",
        action="store_true",
        default=True,
        help="Skip multiple sequence alignment generation for largest cluster"
    )

    parser.add_argument(
        "--msa-sequences",
        type=int,
        default=100,
        help="Number of sequences to use for MSA (default: 100)"
    )

    parser.add_argument(
        "--bed-suffix",
        default="",
        help="Suffix to add to BED output filename (default: none)"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Determine input mode and prepare FASTA file
    if args.bed:
        # BED + genome mode
        if not args.genome:
            print("Error: --genome is required when using --bed option")
            sys.exit(1)

        # Check if input files exist
        if not os.path.exists(args.bed):
            print(f"Error: BED file {args.bed} not found!")
            sys.exit(1)

        if not os.path.exists(args.genome):
            print(f"Error: Genome file {args.genome} not found!")
            sys.exit(1)

        # Generate FASTA from BED coordinates
        input_fasta = f"{args.output_prefix}_extracted_sequences.fa"
        if not extract_sequences_from_bed(args.bed, args.genome, input_fasta):
            print("Failed to extract sequences from BED file!")
            sys.exit(1)

        # Track that we created this file for cleanup
        created_input_fasta = True
    else:
        # Direct FASTA mode
        input_fasta = args.input_fasta
        if not os.path.exists(input_fasta):
            print(f"Error: Input file {input_fasta} not found!")
            sys.exit(1)
        created_input_fasta = False

    # Derive filenames from arguments
    output_prefix = args.output_prefix
    cluster_file = f"{output_prefix}_cluster.tsv"

    # Validate parameters
    if not (0.0 <= args.min_seq_id <= 1.0):
        print("Error: --min-seq-id must be between 0.0 and 1.0")
        sys.exit(1)

    if not (0.0 <= args.coverage <= 1.0):
        print("Error: --coverage must be between 0.0 and 1.0")
        sys.exit(1)

    try:
        print(f"Input file: {input_fasta}")
        print(f"Output prefix: {output_prefix}")
        print(f"Clustering parameters: min_seq_id={args.min_seq_id}, coverage={args.coverage}")
        print("Note: mmseqs automatically considers reverse complements during clustering")
        print()

        # Count sequences
        with open(input_fasta, 'r') as f:
            sequence_count = sum(1 for line in f if line.startswith('>'))

        print(f"Input sequences: {sequence_count:,}")

        # Step 1: Run mmseqs clustering directly on input file
        if not run_mmseqs_clustering(
            input_fasta,
            output_prefix,
            temp_dir=args.temp_dir,
            min_seq_id=args.min_seq_id,
            kmer_per_seq_scale=args.kmer_per_seq_scale,
            coverage=args.coverage
        ):
            print("Clustering failed!")
            sys.exit(1)

        # Step 2: Analyze results
        if os.path.exists(cluster_file):
            results = analyze_clustering_results(cluster_file)
        else:
            print(f"Error: Cluster file {cluster_file} not found!")
            sys.exit(1)

        # Step 3: Save largest cluster sequences
        largest_cluster_fasta = f"{output_prefix}_largest_cluster.fasta"
        num_saved = save_largest_cluster_sequences(results['largest_clusters'], input_fasta, largest_cluster_fasta)

        # Step 4: Create MSA if requested
        if not args.skip_msa and num_saved > 1:
            msa_success, msa_file = create_msa_from_largest_cluster(largest_cluster_fasta, args.msa_sequences)
            if msa_success:
                print(f"MSA created: {msa_file}")

        # Step 5: Save singleton regions and cluster representatives as BED file
        all_regions_bed_file = f"{output_prefix}.deduplicated.bed"
        bed_total_count, bed_singleton_count, bed_rep_count = save_all_regions_bed(
            results['singleton_sequences'],
            results['all_clusters'],
            all_regions_bed_file
        )

        print(f"\n" + "="*60)
        print(f"FINAL RESULTS")
        print(f"="*60)
        print(f"Singleton sequences: {results['singleton']:,}")
        print(f"Clustered sequences: {results['clustered']:,}")
        print(f"Largest cluster size: {results['largest_cluster_size']:,}")
        print(f"Output files:")
        print(f"  - Cluster results: {cluster_file}")
        print(f"  - Largest cluster: {largest_cluster_fasta}")
        print(f"  - All regions BED: {all_regions_bed_file} ({bed_total_count:,} regions: {bed_singleton_count:,} singletons + {bed_rep_count:,} representatives)")
        if not args.skip_msa and num_saved > 1:
            print(f"  - MSA file: {msa_file if 'msa_file' in locals() else 'N/A'}")

        # Show input source information
        if args.bed:
            print(f"  - Input source: BED file {args.bed} + genome {args.genome}")
            if not args.no_cleanup and created_input_fasta:
                print(f"  - Extracted sequences: {input_fasta} (will be cleaned up)")
        else:
            print(f"  - Input source: FASTA file {input_fasta}")
        print(f"="*60)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        # Cleanup extracted sequences file if created and cleanup is enabled
        if not args.no_cleanup and created_input_fasta and os.path.exists(input_fasta):
            cleanup_files(input_fasta)

if __name__ == "__main__":
    main()