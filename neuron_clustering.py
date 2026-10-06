"""
neuron_clustering.py

Feature extraction + PCA/clustering pipeline for beat-omission neurons,
built on top of NeuronOmissionAnalysis / OmissionAnalysis (polar_analysis.py).

Usage
-----
    from polar_analysis import NeuronOmissionAnalysis
    from neuron_clustering import extract_neuron_features, build_population_dataframe, cluster_neurons

    # --- single neuron, for testing ---
    analysis = NeuronOmissionAnalysis(..., unit=231, ...)
    analysis.run()
    feats = extract_neuron_features(analysis)

    # --- whole recording ---
    df = build_population_dataframe(
        unit_ids=[231, 232, ...],   # however you enumerate units for this recording
        db_path=..., stim_lib_path=..., recordings_path=..., raw_sounds_path=...,
        recording=..., stim_subset=[...],
    )
    df, pca, labels = cluster_neurons(df)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------- #
#  Per-stimulus feature extraction (operates on one OmissionAnalysis)
# ---------------------------------------------------------------------- #
def _windows(oa, label):
    """All phase_locking entries for a given label, in time order."""
    return [d for d in oa.phase_locking if d['label'] == label]


def _rate(entry, n_trials, window_s):
    """Firing rate (Hz) for a single window entry."""
    if n_trials == 0 or window_s == 0:
        return np.nan
    return entry['n_spikes'] / (n_trials * window_s)


def _circ_lin_regress(x, angles):
    """
    Simple circular-linear regression: unwrap `angles` (radians) assuming
    slow drift between consecutive points, then fit a straight line vs x.
    Returns (slope, intercept). Used for phase-drift-over-time in the
    decay period, where jumps between consecutive windows should be small.
    """
    if len(angles) < 2:
        return np.nan, np.nan
    unwrapped = np.unwrap(angles)
    slope, intercept = np.polyfit(x, unwrapped, 1)
    return slope, intercept


def extract_stimulus_features(oa, factor=2):
    """
    Extract a flat dict of features for one (neuron, stimulus) OmissionAnalysis
    that has already had .compute_phase_locking() run (this happens inside
    NeuronOmissionAnalysis.run()).
    """
    n_trials = len(oa.trial_spikes) if oa.trial_spikes is not None else 0
    window_s = oa.tempo_s
    feats = {}

    # ---------------- per-label rate + VS summaries ---------------- #
    label_windows = {}
    for label in ('baseline', 'beat', 'omission', 'decay'):
        entries = _windows(oa, label)
        label_windows[label] = entries

        rates = np.array([_rate(e, n_trials, window_s) for e in entries])
        vss = np.array([e['vector_strength'] for e in entries])

        feats[f'{label}_rate_mean'] = np.nanmean(rates) if len(rates) else np.nan
        feats[f'{label}_rate_std'] = np.nanstd(rates) if len(rates) else np.nan
        feats[f'{label}_vs_mean'] = np.nanmean(vss) if len(vss) else np.nan
        feats[f'{label}_n_windows'] = len(entries)

    baseline_rate = feats['baseline_rate_mean']
    baseline_vs = feats['baseline_vs_mean']

    # ---------------- excited vs inhibited (rate-based) ---------------- #
    # z-scored against baseline rate variability across baseline windows
    baseline_rates_raw = np.array([_rate(e, n_trials, window_s) for e in label_windows['baseline']])
    baseline_rate_std = np.nanstd(baseline_rates_raw) if len(baseline_rates_raw) > 1 else np.nan
    eps = 1e-6

    for label in ('beat', 'omission', 'decay'):
        mean_rate = feats[f'{label}_rate_mean']
        if np.isnan(mean_rate) or np.isnan(baseline_rate):
            feats[f'{label}_rate_z'] = np.nan
        elif baseline_rate_std and not np.isnan(baseline_rate_std) and baseline_rate_std > eps:
            feats[f'{label}_rate_z'] = (mean_rate - baseline_rate) / baseline_rate_std
        else:
            # fall back to a simple relative-change score if baseline is too stable to get a std
            feats[f'{label}_rate_z'] = (mean_rate - baseline_rate) / (baseline_rate + eps)

    # first-omission-window excited/inhibited sign (often the cleanest single signal)
    if len(label_windows['omission']) > 0 and not np.isnan(baseline_rate):
        first_omission_rate = _rate(label_windows['omission'][0], n_trials, window_s)
        feats['omission_first_rate_z'] = (
            (first_omission_rate - baseline_rate) / (baseline_rate_std + eps)
            if baseline_rate_std and not np.isnan(baseline_rate_std) and baseline_rate_std > eps
            else (first_omission_rate - baseline_rate) / (baseline_rate + eps)
        )
    else:
        feats['omission_first_rate_z'] = np.nan

    # ---------------- phase-locking (vector components, safe for PCA) ---------------- #
    beat_entries = label_windows['beat']
    if beat_entries:
        beat_phases = np.concatenate([e['phases'] for e in beat_entries])
        beat_phases = beat_phases[~np.isnan(beat_phases)] if len(beat_phases) else beat_phases
    else:
        beat_phases = np.array([])

    if len(beat_phases) > 0:
        beat_z = np.mean(np.exp(1j * beat_phases))
        beat_vs = np.abs(beat_z)
        beat_mean_phase = np.angle(beat_z)
    else:
        beat_vs, beat_mean_phase = np.nan, np.nan

    feats['beat_vs'] = beat_vs
    feats['beat_vs_x'] = beat_vs * np.cos(beat_mean_phase) if not np.isnan(beat_vs) else np.nan
    feats['beat_vs_y'] = beat_vs * np.sin(beat_mean_phase) if not np.isnan(beat_vs) else np.nan

    for label in ('omission', 'decay'):
        entries = label_windows[label]
        if not entries:
            feats[f'{label}_vs_first'] = np.nan
            feats[f'{label}_vs_x'] = np.nan
            feats[f'{label}_vs_y'] = np.nan
            continue
        first = entries[0]
        vs0 = first['vector_strength']
        phase0 = first['mean_phase']
        feats[f'{label}_vs_first'] = vs0
        feats[f'{label}_vs_x'] = vs0 * np.cos(phase0) if not np.isnan(vs0) else np.nan
        feats[f'{label}_vs_y'] = vs0 * np.sin(phase0) if not np.isnan(vs0) else np.nan

    # ---------------- decay dynamics (uses ALL decay windows, not just the first) ---------------- #
    decay_entries = label_windows['decay']
    if len(decay_entries) >= 2:
        idx = np.arange(len(decay_entries))
        decay_vs = np.array([e['vector_strength'] for e in decay_entries])
        decay_phase = np.array([e['mean_phase'] for e in decay_entries])
        decay_rate = np.array([_rate(e, n_trials, window_s) for e in decay_entries])

        valid = ~np.isnan(decay_vs)
        if valid.sum() >= 2:
            # exponential decay fit: vs(t) ~ vs0 * exp(-t/tau)
            # do it in log space, guarding against non-positive VS
            pos = decay_vs > 1e-6
            if pos.sum() >= 2:
                log_vs = np.log(decay_vs[pos])
                slope, intercept = np.polyfit(idx[pos], log_vs, 1)
                tau = -1 / slope if slope < 0 else np.inf  # windows to 1/e; inf = no decay detected
            else:
                tau = np.nan

            # persistence: how many consecutive decay windows stay above threshold
            thresh = factor * baseline_vs if not np.isnan(baseline_vs) else 0.1
            persisting = 0
            for v in decay_vs:
                if np.isnan(v) or v <= thresh:
                    break
                persisting += 1

            # phase drift: slope of unwrapped phase over decay windows (rad/window)
            drift_slope, _ = _circ_lin_regress(idx[valid], decay_phase[valid])
            drift_ms_per_beat = (drift_slope / (2 * np.pi)) * oa.tempo_ms if not np.isnan(drift_slope) else np.nan

            feats['decay_tau_windows'] = tau
            feats['decay_persisting_windows'] = persisting
            feats['decay_phase_drift_ms_per_beat'] = drift_ms_per_beat
            feats['decay_rate_slope'] = np.polyfit(idx, decay_rate, 1)[0] if not np.any(np.isnan(decay_rate)) else np.nan
        else:
            feats['decay_tau_windows'] = np.nan
            feats['decay_persisting_windows'] = np.nan
            feats['decay_phase_drift_ms_per_beat'] = np.nan
            feats['decay_rate_slope'] = np.nan
    else:
        feats['decay_tau_windows'] = np.nan
        feats['decay_persisting_windows'] = np.nan
        feats['decay_phase_drift_ms_per_beat'] = np.nan
        feats['decay_rate_slope'] = np.nan

    # ---------------- reliability ---------------- #
    # fraction of beat windows with at least one spike, as a crude reliability proxy
    if beat_entries:
        feats['beat_reliability'] = np.mean([e['n_spikes'] > 0 for e in beat_entries])
    else:
        feats['beat_reliability'] = np.nan

    # ---------------- candidate flag using existing rule-based logic ---------------- #
    feats['tempo_ms'] = oa.tempo_ms
    feats['baseline_vs'] = baseline_vs
    feats['baseline_rate'] = baseline_rate

    return feats


# ---------------------------------------------------------------------- #
#  Per-neuron aggregation across the stim_subset
# ---------------------------------------------------------------------- #
def extract_neuron_features(analysis, factor=2):
    """
    analysis : a NeuronOmissionAnalysis that has already had .run() called.
    Returns a single flat dict of features, averaged across all stimuli in
    analysis.stim_subset for which the analysis succeeded. Also keeps a
    per-stimulus breakdown under '_by_stimulus' for inspection/debugging.
    """
    per_stim = {}
    for stim, oa in analysis.analyses.items():
        try:
            per_stim[stim] = extract_stimulus_features(oa, factor=factor)
        except Exception as e:
            print(f'  feature extraction failed for stimulus {stim}: {e}')

    if not per_stim:
        return None

    # average numeric features across stimuli (nan-safe)
    all_keys = set()
    for d in per_stim.values():
        all_keys.update(d.keys())

    agg = {}
    for k in all_keys:
        vals = [d[k] for d in per_stim.values() if k in d]
        vals = [v for v in vals if v is not None]
        agg[k] = np.nanmean(vals) if len(vals) else np.nan

    agg['n_stimuli_used'] = len(per_stim)
    agg['recording'] = analysis.recording_name
    agg['unit'] = analysis.unit
    agg['_by_stimulus'] = per_stim  # keep raw per-stim dict for later drill-down

    return agg


# ---------------------------------------------------------------------- #
#  Batch runner across all units in a recording
# ---------------------------------------------------------------------- #
def build_population_dataframe(unit_ids, db_path, stim_lib_path, recordings_path,
                                raw_sounds_path, recording, stim_subset, factor=2):
    """
    Loops NeuronOmissionAnalysis over every unit_id in `unit_ids` for one
    recording, extracts features, and returns a tidy DataFrame (one row per
    neuron) ready for PCA/clustering.
    """
    from polar_analysis import NeuronOmissionAnalysis

    rows = []
    for unit in unit_ids:
        print(f'Processing unit {unit}...')
        try:
            analysis = NeuronOmissionAnalysis(
                db_path=db_path, stim_lib_path=stim_lib_path,
                recordings_path=recordings_path, raw_sounds_path=raw_sounds_path,
                recording=recording, unit=unit, stim_subset=stim_subset,
            )
            analysis.run()
            feats = extract_neuron_features(analysis, factor=factor)
            if feats is not None:
                rows.append(feats)
        except Exception as e:
            print(f'  unit {unit} failed entirely: {e}')

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------- #
#  PCA + clustering
# ---------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    'beat_rate_z', 'omission_rate_z', 'decay_rate_z', 'omission_first_rate_z',
    'beat_vs', 'beat_vs_x', 'beat_vs_y',
    'omission_vs_first', 'omission_vs_x', 'omission_vs_y',
    'decay_vs_first', 'decay_vs_x', 'decay_vs_y',
    'decay_tau_windows', 'decay_persisting_windows',
    'decay_phase_drift_ms_per_beat', 'decay_rate_slope',
    'beat_reliability',
]


def cluster_neurons(df, feature_columns=None, n_components=0.9, n_clusters_range=range(2, 7)):
    """
    Standardizes features, runs PCA, and fits a Gaussian Mixture Model,
    picking the number of clusters by BIC over `n_clusters_range`.

    Returns
    -------
    df : original dataframe with 'cluster' and PC columns appended
    pca : fitted sklearn PCA object
    gmm : fitted sklearn GaussianMixture (best BIC)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    cols = feature_columns or FEATURE_COLUMNS
    cols = [c for c in cols if c in df.columns]

    X = df[cols].copy()

    # cap decay_tau_windows (inf when no decay detected) before scaling
    if 'decay_tau_windows' in X.columns:
        finite_max = X.loc[np.isfinite(X['decay_tau_windows']), 'decay_tau_windows'].max()
        X['decay_tau_windows'] = X['decay_tau_windows'].replace(np.inf, finite_max * 2 if pd.notna(finite_max) else 100)

    # impute remaining NaNs with column median (flag neurons missing omission/decay too)
    df['missing_omission'] = df['omission_vs_first'].isna().astype(int) if 'omission_vs_first' in df else 0
    df['missing_decay'] = df['decay_vs_first'].isna().astype(int) if 'decay_vs_first' in df else 0

    X = X.fillna(X.median())

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=0)
    pcs = pca.fit_transform(Xz)

    best_gmm, best_bic = None, np.inf
    for k in n_clusters_range:
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

    print(f'Selected {best_gmm.n_components} clusters (BIC={best_bic:.1f})')
    print('PCA explained variance ratio:', np.round(pca.explained_variance_ratio_, 3))

    loadings = pd.DataFrame(pca.components_.T, index=cols,
                             columns=[f'PC{i+1}' for i in range(pcs.shape[1])])
    print('\nTop loadings per PC:')
    for pc in loadings.columns[:min(3, loadings.shape[1])]:
        top = loadings[pc].abs().sort_values(ascending=False).head(5)
        print(f'  {pc}:', {k: round(loadings.loc[k, pc], 2) for k in top.index})

    return df, pca, best_gmm