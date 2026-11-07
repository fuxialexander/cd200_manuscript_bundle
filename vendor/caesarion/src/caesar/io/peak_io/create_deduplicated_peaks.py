#!/usr/bin/env python3
"""
Create deduplicated peaks file that retains original columns from scc_all_peaks.bed
by merging mmseqs clustering results with the original data.
"""

import pandas as pd
import argparse
from collections import defaultdict

def parse_clustering_results(cluster_file):
    """Parse mmseqs clustering results."""
    print(f"Parsing clustering results from {cluster_file}...")

    clusters = defaultdict(list)

    with open(cluster_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                representative, member = parts
                # Remove "::" prefix from bedtools output
                representative = representative.lstrip('::')
                member = member.lstrip('::')
                clusters[representative].append(member)

    print(f"Found {len(clusters)} clusters")

    # Get singleton sequences and cluster representatives
    singleton_sequences = set()
    representative_sequences = set()

    for rep, members in clusters.items():
        if len(members) == 1:
            singleton_sequences.add(rep)
        else:
            representative_sequences.add(rep)

    return clusters, singleton_sequences, representative_sequences

def create_sequence_id_mapping(original_bed_file):
    """Create mapping from sequence index to original BED data."""
    print(f"Creating sequence ID mapping from {original_bed_file}...")

    df = pd.read_csv(original_bed_file, sep='\t')
    print(f"Read {len(df)} regions from original BED file")

    # Create sequence ID mapping (bedtools creates IDs as chr:start-end)
    seq_id_to_index = {}
    for idx, row in df.iterrows():
        seq_id = f"{row['Chromosome']}:{row['Start']}-{row['End']}"
        seq_id_to_index[seq_id] = idx

    return df, seq_id_to_index

def create_deduplicated_bed(clusters, singleton_sequences, representative_sequences,
                          original_df, seq_id_to_index, output_file):
    """Create deduplicated BED file with original columns."""
    print(f"Creating deduplicated BED file: {output_file}")

    deduplicated_rows = []
    found_sequences = 0
    missing_sequences = 0

    # Add singleton sequences
    for seq_id in singleton_sequences:
        # Remove any potential prefixes (like "0.0::") to match BED format
        clean_seq_id = seq_id.split("::", 1)[-1] if "::" in seq_id else seq_id
        if clean_seq_id in seq_id_to_index:
            idx = seq_id_to_index[clean_seq_id]
            row = original_df.iloc[idx].copy()
            row['dedup_type'] = 'singleton'
            row['cluster_size'] = 1
            deduplicated_rows.append(row)
            found_sequences += 1
        else:
            missing_sequences += 1

    # Add cluster representatives
    for rep_seq_id in representative_sequences:
        # Remove any potential prefixes (like "0.0::") to match BED format
        clean_rep_seq_id = rep_seq_id.split("::", 1)[-1] if "::" in rep_seq_id else rep_seq_id
        if clean_rep_seq_id in seq_id_to_index:
            idx = seq_id_to_index[clean_rep_seq_id]
            row = original_df.iloc[idx].copy()
            row['dedup_type'] = 'representative'
            row['cluster_size'] = len(clusters[rep_seq_id])
            deduplicated_rows.append(row)
            found_sequences += 1
        else:
            missing_sequences += 1

    # Create deduplicated dataframe
    if deduplicated_rows:
        deduplicated_df = pd.DataFrame(deduplicated_rows)
        # Sort by chromosome and start position
        deduplicated_df = deduplicated_df.sort_values(['Chromosome', 'Start'])
    else:
        # Create empty dataframe with same columns as original
        deduplicated_df = original_df.iloc[0:0].copy()
        deduplicated_df['dedup_type'] = pd.Series(dtype='str')
        deduplicated_df['cluster_size'] = pd.Series(dtype='int')

    # Save to file
    deduplicated_df.to_csv(output_file, sep='\t', index=False)

    print(f"Deduplicated BED file created:")
    print(f"  Total regions: {len(deduplicated_df)}")
    print(f"  Singletons: {len(deduplicated_df[deduplicated_df['dedup_type'] == 'singleton'])}")
    print(f"  Representatives: {len(deduplicated_df[deduplicated_df['dedup_type'] == 'representative'])}")
    print(f"  Found sequences: {found_sequences}")
    print(f"  Missing sequences: {missing_sequences}")

    # Print statistics by peak type
    if 'peak_type' in deduplicated_df.columns:
        peak_type_stats = deduplicated_df.groupby(['peak_type', 'dedup_type']).size().unstack(fill_value=0)
        print(f"\nPeak type distribution:")
        print(peak_type_stats)

    return deduplicated_df

def main():
    parser = argparse.ArgumentParser(description='Create deduplicated peaks with original columns')
    parser.add_argument('--cluster-file', default='scc_all_peaks_cluster.tsv',
                       help='mmseqs cluster results file')
    parser.add_argument('--original-bed', default='atac_prediction/scc_all_peaks.with_header.bed',
                       help='Original BED file with all columns')
    parser.add_argument('--output', default='scc_all_peaks.deduplicated.bed',
                       help='Output deduplicated BED file')

    args = parser.parse_args()

    # Parse clustering results
    clusters, singleton_sequences, representative_sequences = parse_clustering_results(args.cluster_file)

    # Create sequence ID mapping
    original_df, seq_id_to_index = create_sequence_id_mapping(args.original_bed)

    # Create deduplicated BED file
    deduplicated_df = create_deduplicated_bed(
        clusters, singleton_sequences, representative_sequences,
        original_df, seq_id_to_index, args.output
    )

    print(f"\nCompleted! Deduplicated BED file saved to: {args.output}")


if __name__ == "__main__":
    main()