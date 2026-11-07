from __future__ import annotations

import os

import anndata as ad
import imageio
import numpy as np
import pandas as pd
import pyranges as pr
import scanpy as sc
import scvelo as scv
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import cut_tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from gcell.rna.gencode import Gencode

scv.set_figure_params()


class TranscriptionFactorAnalysis:
    def __init__(
        self,
        name,
        input_track_path,
        pred_tracks_path,
        high_quality_tfs_path,
        training_metadata_path,
        peaks_path,
        gencode_path,
        re2g_file_path,
    ):
        """
        Initialize the TranscriptionFactorAnalysis class.

        Parameters:
        name (str): Name of the analysis.
        input_track_path (str): Path to the input track file.
        pred_tracks_path (str): Path to the predicted tracks file.
        high_quality_tfs_path (str): Path to the high-quality transcription factors file.
        training_metadata_path (str): Path to the training metadata file.
        peaks_path (str): Path to the peaks file.
        gencode_path (str): Path to the GENCODE directory.
        re2g_file_path (str): Path to the regulatory elements to gene mapping file.
        """
        self.name = name
        self.input_track = np.load(input_track_path)
        self.pred_tracks = np.load(pred_tracks_path)
        self.high_qual_tfs = np.loadtxt(high_quality_tfs_path, dtype=str)
        self.tf_list = pd.read_csv(training_metadata_path, header=None).values[:, 0]
        self.peaks = pd.read_csv(peaks_path, sep="\t")
        self.gencode = Gencode(assembly="hg38", version=44, gtf_dir=gencode_path)
        self.gene_list = []  # Initialize an empty list for genes
        self.gene_peak_dict = {}  # Initialize an empty dictionary for gene peaks
        self.re2g_data = pd.read_csv(
            re2g_file_path,
            sep="\t",
            header=None,
            names=["Chromosome", "Start", "End", "Genes"],
        )

    def add_gene_for_coexpression_analysis(self, gene_name):
        """
        Add a gene to the list for coexpression analysis.

        Parameters:
        gene_name (str): Name of the gene to be added for analysis.
        """
        self.gene_list.append(gene_name)
        self.gene_peak_dict[gene_name] = self._retrieve_gene_peak_data(
            self.gencode.get_gene(gene_name)
        )

    def _retrieve_gene_peak_data(self, gene, distance_to_tss=200_000):
        """
        Retrieve peak data for a specific gene within a given distance to the transcription start site (TSS).

        Parameters:
        gene (GencodeGene): Gene object from GENCODE.
        distance_to_tss (int): Distance to TSS for peak retrieval. Default is 200,000.

        Returns:
        DataFrame: DataFrame containing peak data for the specified gene.
        """
        chromosome = gene.tss_list.Chromosome.iloc[0]
        gene_peaks = self.peaks[
            (self.peaks.Start > gene.tss_list.Start.min() - distance_to_tss)
            & (self.peaks.End < gene.tss_list.End.max() + distance_to_tss)
            & (self.peaks.Chromosome == chromosome)
        ]
        gene_peaks["gene"] = gene.name
        return gene_peaks

    def execute_regression_analysis(self, indices, residue_mean_threshold=0.05):
        """
        Perform regression analysis on the data.

        Parameters:
        indices (list of int): Indices to perform regression on.
        residue_mean_threshold (float): Threshold for determining the residue mean. Default is 0.05.
        """
        self.indices = indices
        self.per_peak_corr_across_traj = []
        self.residues = []

        for i in tqdm(indices):
            peak_factor_residues = self._compute_regression_residues(
                i, residue_mean_threshold
            )
            self.residues.append(peak_factor_residues)
            self.per_peak_corr_across_traj.append(np.corrcoef(peak_factor_residues.T))

        self.residues = np.stack(self.residues)
        self.per_peak_corr_across_traj = np.stack(self.per_peak_corr_across_traj)

    def _compute_regression_residues(self, index, residue_mean_threshold):
        """
        Compute regression residues for a given index.

        Parameters:
        index (int): Index of the peak observation to compute residues for.
        residue_mean_threshold (float): Threshold for determining the residue mean.

        Returns:
        ndarray: Array of regression residues.
        """
        peak_factor_residues = np.zeros(
            (self.pred_tracks.shape[1], self.pred_tracks.shape[2])
        )
        peak_obs = self.pred_tracks[index]

        for j in range(self.pred_tracks.shape[2]):
            residue_j, mask = self._calculate_linear_regression_residue(
                peak_obs, j, residue_mean_threshold
            )
            peak_factor_residues[:, j] = residue_j * mask

        return peak_factor_residues

    @staticmethod
    def _calculate_linear_regression_residue(peak_obs, column, residue_mean_threshold):
        """
        Calculate residues from linear regression.

        Parameters:
        peak_obs (ndarray): Peak observations.
        column (int): Column index for the observation.
        residue_mean_threshold (float): Threshold for determining the residue mean.

        Returns:
        tuple: A tuple containing the residue array and mask.
        """
        lr = LinearRegression()
        lr.fit(peak_obs.mean(1).reshape(-1, 1), peak_obs[:, column])
        pred = lr.predict(peak_obs.mean(1).reshape(-1, 1))
        residue = peak_obs[:, column] - pred
        mask = peak_obs[:, column].mean() > residue_mean_threshold
        return residue * mask, mask

    def create_tf_coexpression_matrix(self, selected_peak=None):
        """
        Create a transcription factor coexpression matrix.

        Parameters:
        selected_peak (list of int, optional): Indices of selected peaks. If None, all residues are used. Default is None.

        Returns:
        DataFrame: DataFrame representing the transcription factor coexpression matrix.
        """
        if selected_peak is None:
            selected_peak = np.arange(len(self.residues))
        corr_res = np.corrcoef(self.residues[selected_peak].sum(0).T)
        corr_df = pd.DataFrame(corr_res, index=self.tf_list, columns=self.tf_list)
        return corr_df

    def filter_tf_coexpression_matrix(
        self, corr_df, selected_peak=None, level_filter_threshold=0.5
    ):
        """
        Filter the transcription factor coexpression matrix.

        Parameters:
        corr_df (DataFrame): Coexpression matrix DataFrame.
        selected_peak (list of int, optional): Indices of selected peaks. If None, all residues are used. Default is None.
        level_filter_threshold (float): Threshold for level filtering. Default is 0.5.

        Returns:
        DataFrame: Filtered coexpression matrix DataFrame.
        """
        if selected_peak is None:
            selected_peak = np.arange(len(self.residues))
        high_qual_filter = np.array([i in self.high_qual_tfs for i in corr_df.index])
        level_filter = (
            self.pred_tracks[self.indices[selected_peak]].max(0).max(0)
            > level_filter_threshold
        )
        return corr_df.loc[
            high_qual_filter & level_filter, high_qual_filter & level_filter
        ]

    def generate_tf_clustermap(
        self, cmap="RdBu_r", center=0, vmin=-1, vmax=1, figsize=(20, 20), method="ward"
    ):
        """
        Generate a clustermap for transcription factors.

        Parameters:
        corr_df_filtered (DataFrame): Filtered coexpression matrix DataFrame.
        cmap (str): Colormap for the clustermap. Default is 'RdBu_r'.
        center (float): Center value for colormap normalization. Default is 0.
        vmin (float): Minimum value for colormap normalization. Default is -1.
        vmax (float): Maximum value for colormap normalization. Default is 1.
        figsize (tuple): Figure size. Default is (20, 20).
        method (str): Clustering method. Default is 'ward'.

        Returns:
        ClusterGrid: Seaborn ClusterGrid instance.
        """
        g = sns.clustermap(
            self.tf_coexpression_filtered,
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            method=method,
        )
        return g

    def extract_coregulators(self, height=4):
        self.tf_coexpression = self.create_tf_coexpression_matrix()
        self.tf_coexpression_filtered = self.filter_tf_coexpression_matrix(
            self.tf_coexpression
        )
        self.tf_coexpression_heatmap = self.generate_tf_clustermap()
        self.tf_coexpression_clusters, self.tf_coexpression_cut = self.extract_clusters(
            self.tf_coexpression_heatmap, self.tf_coexpression_filtered, height=height
        )

    def get_coregulator_of(self, tf):
        for c in self.tf_coexpression_clusters:
            if tf in c.index.values:
                return c.index.values, np.where(c.columns.isin(c.index.values))[0]

    def find_overlapping_genes(self, cluster_peaks, distance_to_tss=200_000):
        """
        Find genes overlapping with given cluster peaks using re2g data.

        Parameters:
        cluster_peaks (DataFrame): DataFrame containing peak information for a cluster.
        distance_to_tss (int): Distance to the transcription start site for finding overlapping genes. Default is 200,000.

        Returns:
        list: List of unique genes overlapping with the cluster peaks.
        """
        encode_re2g = pr.PyRanges(self.re2g_data)
        peaks_pr = pr.PyRanges(cluster_peaks)
        overlap = encode_re2g.join(peaks_pr)
        return sorted(overlap.df.Genes.unique())

    def save_genes_associated_with_cluster(self, cluster_peaks, output_file):
        """
        Save genes associated with a specific cluster to a file.

        Parameters:
        cluster_peaks (DataFrame): DataFrame containing peak information for a cluster.
        output_file (str): Path to the output file where the gene names will be saved.
        """
        overlapping_genes = self.find_overlapping_genes(cluster_peaks)
        with open(output_file, "w") as file:
            for gene in overlapping_genes:
                file.write(f"{gene}\n")

    def plot_tf_expression_heatmap(
        self, data, title, output_file, cmap="RdBu_r", figsize=(10, 8), vmax=None
    ):
        """
        Plot and save a heatmap of transcription factor expression.

        Parameters:
        data (DataFrame): DataFrame containing transcription factor expression data.
        title (str): Title of the heatmap.
        output_file (str): Path to the output file where the heatmap will be saved.
        cmap (str): Colormap for the heatmap. Default is 'RdBu_r'.
        figsize (tuple): Figure size. Default is (10, 8).
        vmax (float): Maximum value for colormap normalization. If None, it is automatically determined. Default is None.
        """
        plt.figure(figsize=figsize)
        sns.heatmap(data, cmap=cmap, vmax=vmax)
        plt.title(title)
        plt.savefig(output_file)
        plt.close()

    def extract_clusters(self, g, arr, height=10):
        """
        Extract clusters from the provided clustermap.

        Parameters:
        g (ClusterGrid): Seaborn ClusterGrid instance from a clustermap.
        arr (ndarray): Array to extract clusters from.
        height (float): Height at which to cut the dendrogram tree for clustering. Default is 10.

        Returns:
        tuple: A tuple containing the list of clusters and the cut tree.
        """
        linkage_matrix = g.dendrogram_row.linkage
        cut = cut_tree(linkage_matrix, height=height)
        cluster_list = []
        for i in np.unique(cut):
            cluster_list.append(arr[cut.flatten() == i])

        return cluster_list, cut

    def create_heatmap_gif(self, selected_cluster, tf_names, output_file):
        """
        Create and save a GIF of heatmaps for the selected cluster.

        Parameters:
        selected_cluster (ndarray): Indices of the selected cluster.
        output_file (str): Path to the output GIF file.
        """
        frame = []
        for i in range(10):
            selected_peaks_tf_df_i = pd.DataFrame(
                self.pred_tracks[selected_cluster][:, i, :],
                index=self.peaks.iloc[selected_cluster].index,
                columns=self.tf_list,
            )
            selected_peaks_tf_df_i = selected_peaks_tf_df_i.loc[:, tf_names]
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(selected_peaks_tf_df_i, cmap="mako", ax=ax, cbar=False)
            ax.set_title(f"Timepoint {i}")
            fig_path = f"./temp_cluster_{i}.png"
            fig.savefig(fig_path)
            plt.close(fig)
            frame.append(fig_path)

        images = [imageio.imread(filename) for filename in frame]
        imageio.mimsave(output_file, images)

        # Cleanup temporary files
        for filename in frame:
            os.remove(filename)

    def get_coregulator_specific_peaks(
        self, tf="BCL6", trend="increase", fold=2, height=1
    ):
        tf_names, tf_idx = self.get_coregulator_of(tf)
        tf_of_intersest_df = pd.DataFrame(self.pred_tracks[:, :, tf_idx].mean(2))
        pd.DataFrame(self.input_track[:, :, 0])
        g = sns.clustermap(
            tf_of_intersest_df,
            cmap="RdBu_r",
            row_cluster=True,
            col_cluster=False,
            vmax=1,
            figsize=(3, 5),
            method="ward",
        )
        clusters, cut = self.extract_clusters(g, self.peaks, height=height)
        row_color = np.zeros(len(self.peaks))
        for i, c in enumerate(cut):
            row_color[i] = c[0]
        row_color_cluster = pd.Series(row_color, index=self.peaks.index)
        color_map = sns.color_palette("tab20", len(clusters))
        row_color = row_color_cluster.map(
            {i: color_map[i] for i in range(len(clusters))}
        )
        g = sns.clustermap(
            tf_of_intersest_df,
            cmap="RdBu_r",
            row_cluster=True,
            col_cluster=False,
            vmax=1,
            figsize=(3, 5),
            method="ward",
            row_colors=row_color,
        )

        def generate_filter(trend, fold, input):
            if trend == "increase":
                filter = (
                    input.iloc[:, 6:9].sum(0).mean() / input.iloc[:, 0:3].sum(0).mean()
                    > fold
                )
            elif trend == "decrease":
                filter = (
                    input.iloc[:, 0:3].sum(0).mean() / input.iloc[:, 6:9].sum(0).mean()
                    > fold
                )
            return filter

        selected_clusters = []
        selected_peaks = []
        for i, c in enumerate(clusters):
            cluster_matrix = tf_of_intersest_df.iloc[c.index.values]
            if generate_filter(trend, fold, cluster_matrix):
                selected_clusters.append(i)
                selected_peaks.extend(c.index.values)
        cluster_peaks = self.peaks.iloc[selected_peaks]
        return tf_names, selected_peaks, cluster_peaks

    def plot_coregulator_specific_peaks(
        self, output_dir=".", tf="BCL6", trend="increase", fold=2, height=1
    ):
        tf_names, selected_peaks, cluster_peaks = self.get_coregulator_specific_peaks(
            tf=tf, trend=trend, fold=fold, height=height
        )
        self.save_genes_associated_with_cluster(
            cluster_peaks,
            f"{output_dir}/{tf}_coregulator.{trend}_trend.{fold}_fold.cluster_genes.txt",
        )
        self.create_heatmap_gif(
            selected_peaks,
            tf_names,
            f"{output_dir}/{tf}_coregulator.{trend}_trend.{fold}_fold.gif",
        )
        # save tf_names and cluster_peaks to file
        with open(
            f"{output_dir}/{tf}_coregulator.{trend}_trend.{fold}_fold.tf_names.txt", "w"
        ) as file:
            for tf_i in tf_names:
                file.write(f"{tf_i}\n")
        cluster_peaks.to_csv(
            f"{output_dir}/{tf}_coregulator.{trend}_trend.{fold}_fold.cluster_peaks.tsv",
            sep="\t",
            index=False,
        )

        return cluster_peaks, tf_names, selected_peaks

    def create_obs_with_peaks(self):
        obs_list = []
        for i in range(self.pred_tracks.shape[1]):
            # Using peaks information from the analysis instance
            obs_i = pd.DataFrame(self.peaks)
            # Annotating each observation with its stage
            obs_i["stage"] = f"{self.name}{i}"
            # Annotating each observation with its trajectory name
            obs_i["traj"] = self.name
            obs_list.append(obs_i)
        return pd.concat(obs_list, axis=0)

    def get_anndata(self):
        obs = self.create_obs_with_peaks()
        self.adata = ad.AnnData(
            self.pred_tracks.reshape(-1, self.pred_tracks.shape[2]),
            var=pd.DataFrame({"TF": self.tf_list}),
            obs=obs,
        )
        self.adata.var_names = self.tf_list
        return self.adata

    def get_velocity_adata(self, start_stage, end_stage):
        start_idx = np.where(self.adata.obs["stage"] == start_stage)[0]
        end_idx = np.where(self.adata.obs["stage"] == end_stage)[0]
        adata = self.adata[start_idx].copy()
        start_data = self.adata.X[start_idx]
        end_data = self.adata.X[end_idx]
        velocity = end_data - start_data
        adata.layers["velocity"] = velocity
        return adata


# # Execute regression analysis
# indices = np.random.choice(len(analysis_t.pred_tracks), 1000, replace=False)
# analysis_t.execute_regression_analysis(indices)
# analysis_t.extract_coregulators(height=30)
# analysis_t.plot_coregulator_specific_peaks(tf='MED1', output_dir='./tex/')


class VelocityAnndataIntegrator:
    def __init__(self, adata_list):
        """
        Initialize the VelocityAnndataIntegrator class with a list of anndata objects.
        """
        self.adata_list = adata_list
        # merge anndata
        self.adata = ad.concat(adata_list, axis=0)

    def preprocess(
        self, n_neighbors=10, n_pcs=30, use_rep="X", resolution=0.5, min_dist=0.3
    ):
        """
        Preprocess the anndata object for velocity analysis.

        Parameters:
        n_neighbors (int): Number of neighbors for KNN graph construction. Default is 10.
        n_pcs (int): Number of principal components. Default is 30.
        use_rep (str): Representation to use for KNN graph construction. Default is 'X'.
        resolution (float): Resolution for Leiden clustering. Default is 0.5.
        min_dist (float): Minimum distance for UMAP embedding. Default is 0.3.
        """
        scv.pp.pca(self.adata, n_comps=n_pcs)
        scv.pp.neighbors(self.adata, n_neighbors=n_neighbors, use_rep=use_rep)
        sc.tl.leiden(self.adata, resolution=resolution)
        scv.tl.umap(self.adata, min_dist=min_dist)

    def visualize_velocity(self, style="grid", query=None, **kwargs):
        """
        Visualize velocity using scVelo.

        Parameters:
        style (str): Style for visualization ('grid' or 'stream').
        query (str): Query for subsetting the anndata object. Default is None.
        **kwargs: Additional keyword arguments for scv.pl.velocity_embedding_grid().
        """
        adata = self.query_adata(query)
        adata.uns["velocity_graph"] = cosine_similarity(adata.layers["velocity"])
        if style == "grid":
            scv.pl.velocity_embedding_grid(adata, **kwargs)
        elif style == "stream":
            scv.pl.velocity_embedding_stream(adata, **kwargs)

    def query_adata(self, query):
        """
        Query the anndata object using a pandas query string.
        """
        if query is None:
            return self.adata.copy()
        idx = self.adata.obs.eval(query).values
        return self.adata[idx].copy()

    def get_marker_tfs(
        self,
        query=None,
        groupby="leiden",
        method="t-test",
        layer="velocity",
        use_raw=False,
    ):
        """
        Get marker transcription factors using scVelo.

        Parameters:
        query (str): Query for subsetting the anndata object. Default is None.
        groupby (str): Groupby key for marker gene identification. Default is 'leiden'.
        method (str): Method for marker gene identification. Default is 't-test'.
        layer (str): Layer for marker gene identification. Default is 'velocity'.
        use_raw (bool): Whether to use raw data for marker gene identification. Default is False.
        """
        adata = self.query_adata(query)
        # add peak count for each leiden cluster
        leiden_count = adata.obs.value_counts("leiden").to_dict()
        adata.obs["peak_count"] = adata.obs["leiden"].map(leiden_count).astype(int)
        adata = adata[adata.obs["peak_count"] > 5].copy()
        sc.tl.rank_genes_groups(
            adata,
            groupby,
            method=method,
            layer=layer,
            use_raw=use_raw,
            rankby_abs=True,
            n_genes=20,
        )
        return sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)

    def plot_velocity_on_umap(self, query, color=["leiden"], legend_loc="on data"):
        return sc.pl.umap(
            self.query_adata(query),
            color=color,
            legend_loc=legend_loc,
            layer="velocity",
        )

    def compute_cluster_correlation(self, traj1, traj2):
        """
        Compute the correlation between mean velocity of two trajectories, stratified by clusters,
        plot a heatmap of the correlation matrix, and return the correlation matrix.
        """
        adata_traj1 = self.query_adata(f'traj=="{traj1}"')
        adata_traj2 = self.query_adata(f'traj=="{traj2}"')
        cluster_corr = np.zeros(len(adata_traj1.obs["leiden"].unique()))
        for i, c in enumerate(adata_traj1.obs["leiden"].unique()):
            velocity1 = (
                adata_traj1[adata_traj1.obs["leiden"] == c].layers["velocity"].mean(0)
            )
            velocity2 = (
                adata_traj2[adata_traj2.obs["leiden"] == c].layers["velocity"].mean(0)
            )
            cluster_corr[i] = np.corrcoef(velocity1, velocity2)[0, 1]
        return cluster_corr
