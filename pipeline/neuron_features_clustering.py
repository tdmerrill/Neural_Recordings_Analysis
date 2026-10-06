"""
neuron_features_clustering.py

PCA + clustering + cluster-inspection for the general `neuron_features`
SQL table (rate, E/I, tuning, adaptation, temporal coding, PSTH shape).

Usage
-----
    from neuron_features_clustering import (
        load_features, cluster_neurons, write_cluster_labels, plot_cluster_examples
    )

    df = load_features(db_path)                 # all sessions
    df = load_features(db_path, session_id=rec)  # one recording

    df, pca, gmm, used_cols = cluster_neurons(df)
    write_cluster_labels(db_path, df)            # writes cluster_id back to SQL
    plot_cluster_examples(df, plot_cache, n_examples=3)
"""

import sqlite3
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# Every clustering-relevant column from the neuron_features schema.
# Deliberately excludes IDs/metadata (session_id, unit_id, bird_id,
# brain_region, training_status) and cluster_id itself.
FEATURE_COLUMNS = [
    'baseline_fr', 'response_probability', 'mean_pairwise_trial_corr',
    'peak_excitation', 'peak_suppression',
    'excitation_latency', 'suppression_latency',
    'excitation_duration', 'suppression_duration',
    'excitation_inhibition_index',
    'response_centroid', 'response_skewness', 'num_response_peaks',
    'adaptation_index', 'adaptation_slope',
    'steady_state_response', 'recovery_after_silence',
    'vector_strength', 'preferred_phase',
    'stimulus_correlation', 'stimulus_lag',
    'best_frequency', 'bandwidth',
    'natural_sound_preference',
    'psth_pc1', 'psth_pc2', 'psth_pc3',
]

# preferred_phase is circular (radians) -- don't feed it to PCA directly
CIRCULAR_COLUMNS = ['preferred_phase']


# ---------------------------------------------------------------------- #
#  Load / write
# ---------------------------------------------------------------------- #
def load_features(db_path, session_id=None):
    """Reads neuron_features into a DataFrame, optionally filtered to one session_id."""
    conn = sqlite3.connect(db_path)
    if session_id is None:
        df = pd.read_sql_query("SELECT * FROM neuron_features", conn)
    else:
        df = pd.read_sql_query(
            "SELECT * FROM neuron_features WHERE session_id = ?", conn, params=(session_id,)
        )
    conn.close()
    return df


def write_cluster_labels(db_path, df, id_cols=('session_id', 'unit_id'), cluster_col='cluster'):
    """Writes df[cluster_col] back into neuron_features.cluster_id for each (session_id, unit_id)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for _, row in df.iterrows():
        cur.execute(
            "UPDATE neuron_features SET cluster_id = ? WHERE session_id = ? AND unit_id = ?",
            (int(row[cluster_col]), row[id_cols[0]], row[id_cols[1]]),
        )
    conn.commit()
    conn.close()
    print(f'Wrote cluster_id for {len(df)} neurons back to {db_path}')


# ---------------------------------------------------------------------- #
#  PCA + clustering
# ---------------------------------------------------------------------- #
def cluster_neurons(df, feature_columns=None, n_components=10,
                     n_clusters_range=range(2, 7), min_fraction_present=0.7):
    """
    Standardizes features, runs PCA, fits a GMM (best BIC over n_clusters_range).

    min_fraction_present : columns with fewer than this fraction of non-null
        values across neurons are dropped entirely (rather than median-imputed,
        which would be meaningless if e.g. a feature hasn't been implemented
        yet and is all-None).

    Returns (df_with_clusters, pca, gmm, feature_columns_actually_used)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    cols = list(feature_columns or FEATURE_COLUMNS)
    cols = [c for c in cols if c in df.columns]

    X = df[cols].apply(pd.to_numeric, errors='coerce').copy()

    # circular features -> sin/cos components instead of raw radians
    for c in CIRCULAR_COLUMNS:
        if c in X.columns:
            X[f'{c}_sin'] = np.sin(X[c])
            X[f'{c}_cos'] = np.cos(X[c])
            X = X.drop(columns=[c])

    # drop columns that are mostly/entirely missing (e.g. not-yet-implemented
    # features you set to None) -- imputing an all-NaN column is meaningless
    present_frac = X.notna().mean()
    dropped = present_frac[present_frac < min_fraction_present].index.tolist()
    if dropped:
        print(f'Dropping mostly-missing columns from clustering: {dropped}')
    X = X.drop(columns=dropped)

    # drop zero-variance columns (would blow up after z-scoring)
    zero_var = X.columns[X.std(skipna=True).fillna(0) == 0].tolist()
    if zero_var:
        print(f'Dropping zero-variance columns: {zero_var}')
        X = X.drop(columns=zero_var)

    used_cols = X.columns.tolist()
    if len(used_cols) < 2:
        raise ValueError(
            f'Only {len(used_cols)} usable feature column(s) after dropping missing/'
            f'constant columns -- add more features before clustering.'
        )

    X = X.fillna(X.median())

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=0)
    pcs = pca.fit_transform(Xz)

    best_gmm, best_bic = None, np.inf
    for k in n_clusters_range:
        if k >= len(df):
            continue
        gmm = GaussianMixture(n_components=k, random_state=0, n_init=5)
        gmm.fit(pcs)
        bic = gmm.bic(pcs)
        if bic < best_bic:
            best_bic, best_gmm = bic, gmm

    labels = best_gmm.predict(pcs)

    df = df.copy()
    df['cluster'] = labels
    for i in range(pcs.shape[1]):
        df[f'PC{i+1}'] = pcs[:, i]

    print(f'Used {len(used_cols)} features: {used_cols}')
    print(f'Selected {best_gmm.n_components} clusters (BIC={best_bic:.1f})')
    print('PCA explained variance ratio:', np.round(pca.explained_variance_ratio_, 3))

    loadings = pd.DataFrame(pca.components_.T, index=used_cols,
                             columns=[f'PC{i+1}' for i in range(pcs.shape[1])])
    print('\nTop loadings per PC:')
    for pc in loadings.columns[:min(3, loadings.shape[1])]:
        top = loadings[pc].abs().sort_values(ascending=False).head(5)
        print(f'  {pc}:', {k: round(loadings.loc[k, pc], 2) for k in top.index})

    return df, pca, best_gmm, used_cols


# ---------------------------------------------------------------------- #
#  PSTH cache for cluster inspection (picklable, no live DB/Recording objs)
# ---------------------------------------------------------------------- #
def cache_example_psth(unit, recording, recordings_path, db=None, stimulus=None):
    """
    Fetches one PSTH (time, response) for a neuron for later plotting, without
    holding onto the live Neuron/Recording/Database objects (not picklable).
    If stimulus is None, uses the first available stimulus for that neuron.
    """
    import os
    from SiNAPSE.core import Neuron, Recording

    rec_path = os.path.join(recordings_path, recording.split(' ')[0], recording)
    rec = Recording(rec_path, samplerate=30000, db=db)
    N = Neuron(recording, unit, rec=rec, db=db)
    stims = N.load

    stim = stimulus if stimulus in stims else stims[0]
    _, spikes, duration = N.raster(stim, baseline=False, plot=False, ax=None)
    _, [time, response] = N.psth(stim, spikes, plot=False, ax=None)

    return {'stimulus': stim, 'time': np.asarray(time), 'response': np.asarray(response)}


def build_plot_cache(units_and_recordings, recordings_path, db=None):
    """
    units_and_recordings : iterable of (unit, recording) tuples
    Returns {(recording, unit): {'stimulus', 'time', 'response'}}
    """
    cache = {}
    for unit, recording in units_and_recordings:
        try:
            cache[(recording, unit)] = cache_example_psth(unit, recording, recordings_path, db=db)
        except Exception as e:
            print(f'  could not cache PSTH for unit {unit} ({recording}): {e}')
    return cache


def save_plot_cache(plot_cache, path):
    with open(path, 'wb') as f:
        pickle.dump(plot_cache, f)


def load_plot_cache(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------- #
#  Cluster inspection
# ---------------------------------------------------------------------- #
def plot_cluster_examples(df, plot_cache, n_examples=3, cluster_col='cluster',
                           id_cols=('session_id', 'unit_id'), show=True):
    """
    For each cluster, plots example PSTHs so you can eyeball whether clusters
    correspond to real response types (excited, inhibited, tuned, adapting...).
    """
    import matplotlib.pyplot as plt

    for cluster_id, group in df.sort_values(cluster_col).groupby(cluster_col):
        keys = list(zip(group[id_cols[0]], group[id_cols[1]]))
        chosen = keys[:n_examples]
        print(f'\n=== Cluster {cluster_id} (n={len(keys)} neurons) -- showing {chosen} ===')

        fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 3.5), squeeze=False)
        for ax, (recording, unit) in zip(axes[0], chosen):
            cached = plot_cache.get((recording, unit))
            if cached is None:
                ax.set_title(f'unit {unit}\n(no cached PSTH)')
                continue
            ax.plot(cached['time'], cached['response'])
            ax.set_title(f'unit {unit}\n{cached["stimulus"]}', fontsize=8)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Firing rate')

        fig.suptitle(f'Cluster {cluster_id}', y=1.05)
        if show:
            plt.tight_layout()
            plt.show()

def plot_pca_clusters(df, cluster_col='cluster'):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 10))

    ax12 = fig.add_subplot(221)
    ax13 = fig.add_subplot(222)
    ax23 = fig.add_subplot(223)
    ax3d = fig.add_subplot(224, projection='3d')

    for cluster_id, group in df.groupby(cluster_col):

        kwargs = dict(
            s=40,
            alpha=0.8,
            label=f'Cluster {cluster_id}'
        )

        ax12.scatter(group['PC1'], group['PC2'], **kwargs)
        ax13.scatter(group['PC1'], group['PC3'], **kwargs)
        ax23.scatter(group['PC2'], group['PC3'], **kwargs)

        ax3d.scatter(
            group['PC1'],
            group['PC2'],
            group['PC3'],
            **kwargs
        )

    # Labels
    ax12.set_xlabel("PC1")
    ax12.set_ylabel("PC2")
    ax12.set_title("PC1 vs PC2")

    ax13.set_xlabel("PC1")
    ax13.set_ylabel("PC3")
    ax13.set_title("PC1 vs PC3")

    ax23.set_xlabel("PC2")
    ax23.set_ylabel("PC3")
    ax23.set_title("PC2 vs PC3")

    ax3d.set_xlabel("PC1")
    ax3d.set_ylabel("PC2")
    ax3d.set_zlabel("PC3")
    ax3d.set_title("3D PCA")

    # Only one legend
    handles, labels = ax12.get_legend_handles_labels()
    fig.legend(handles, labels,
               title="Cluster",
               loc="center right")

    plt.tight_layout(rect=(0, 0, 0.92, 1))
    plt.show()