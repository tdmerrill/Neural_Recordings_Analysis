from SiNAPSE.core import Database, Recording, Neuron
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def get_neurons(db_path, stimuli_path, table='neuron_features'):
    db = Database(db_path, stimuli_path)

    cluster = {}
    for i in range(10):
        conditions = {'cluster_id': ('=', f'{i}')}
        select_columns = ['session_id', 'unit_id']
        neurons = db.load_neurons_from_database(select_columns, conditions, table=table)
        if len(neurons) > 0:
            cluster[i] = neurons
    return cluster, db

def find_common_stim(recordings_path, neurons, db):
    common_stim = None
    computed_sessions = set()

    for session_id, unit_id in neurons:
        if session_id in computed_sessions:
            continue

        computed_sessions.add(session_id)
        bird = session_id.split(' ')[0]
        recording_path = os.path.join(recordings_path, bird, session_id)
        rec = Recording(recording_path, samplerate=30000, db=db)
        N = Neuron(session_id, unit_id, rec=rec, db=db)
        stims = N.load
        stims = set(stims)

        if common_stim is None:
            common_stim = stims
        else:
            common_stim &= stims

    return sorted(common_stim) if common_stim else []

def fit_neural_space(recordings_path, neurons, common_stims, cluster_id, db, padding=1, binsize_ms=10):
    # fit a neural space on the response strength of each condition
        # ends with m*n matrix:
        # m = time points (rows)
        # n = neurons (columns)

    save_path = os.path.join(__file__, '..', '..', '.cluster_pca_mats', f'_Cluster{cluster_id}.npz')
    if not os.path.exists(save_path):
        responses = []
        print(f'Generating PCA matrix for Cluster {cluster_id} with {len(neurons)} neurons.')
        for session_id, unit_id in neurons:
            condition_labels=[]
            time_labels=[]

            bird = session_id.split(' ')[0]
            recording_path = os.path.join(recordings_path, bird, session_id)
            rec = Recording(recording_path, samplerate=30000, db=db)
            N = Neuron(session_id, unit_id, rec=rec, db=db)
            N.load

            all_baseline_spikes = []
            all_trial_spikes = {}
            expected_rates = []
            for  stimulus in common_stims:
                _, trial_spikes, tduration = N.raster(stimulus, baseline=False, padding=padding, separate_trials=True, plot=False, ax=None)
                tduration += padding
                if stimulus not in all_trial_spikes.keys():
                    all_trial_spikes[stimulus] = trial_spikes
                    all_trial_spikes[f'{stimulus}_duration'] = tduration
                _, baseline_spikes, bduration = N.raster(stimulus, baseline=True, plot=False, padding=padding,
                                                         separate_trials=False)
                bduration += padding

                all_baseline_spikes.extend(baseline_spikes)

                # calculate expected spikes per time bin
                n_timebins = bduration / (binsize_ms / 1000)
                n_spikes = len(baseline_spikes)
                n_expected_per_trial = n_spikes / n_timebins / len(trial_spikes)
                expected_rates.append(n_expected_per_trial)
            expected_rate = np.array(expected_rates).mean()

            # build trial response matrix
            trial_data = []
            for stimulus in common_stims:
                duration = all_trial_spikes[f'{stimulus}_duration']

                trial_rates = []

                for trial in all_trial_spikes[stimulus]:
                    counts, _ = np.histogram(
                        trial,
                        bins=np.arange(
                            0,
                            duration + binsize_ms / 1000,
                            binsize_ms / 1000
                        )
                    )
                    trial_rates.append(counts - expected_rate)

                # Shape: (n_trials, n_timebins)
                trial_rates = np.asarray(trial_rates)

                # Mean response for this stimulus
                mean_rate = trial_rates.mean(axis=0)
                trial_data.extend(mean_rate)
                condition_labels.extend([stimulus] * len(mean_rate))
                time_labels.extend(np.arange(len(mean_rate)) * binsize_ms / 1000)
            response = np.array(trial_data).T
            responses.append(response)
        X = np.column_stack(responses)
        np.savez_compressed(
            save_path,
            X=X,
            condition_labels=condition_labels,
            time_labels=time_labels
        )
    # load data back in and fit neural space
    data = np.load(save_path)
    X = data['X']
    conditions = data['condition_labels']
    times = data['time_labels']

    print(f'Loaded PCA matrix with these features:')
    print(f'  # neurons: {X.shape[1]}')
    print(f'  Time Series Length: {X.shape[0]}')
    print(f'  Condition labels: {len(conditions)}')
    print(f'  Time labels: {len(times)}')

    print(f'Fitting PCA model...')
    from sklearn.decomposition import PCA
    X_z = (X - X.mean(axis=0)) / X.std(axis=0)
    pca = PCA(n_components=10)  # or 10–20 depending on use
    Z = pca.fit_transform(X_z)
    print(f'  Success! Returning model.')
    return Z, X, pca, conditions, times

def plot_conditions(pca_result,
                    condition='ZF A_20db_180ms_8b_1xomit_8b_silence'):
    Z, X, pca, conditions, times = pca_result

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)

    idx = np.where(conditions == condition)[0]
    idx = idx[np.argsort(times[idx])]

    X_cond = X[idx]
    X_cond_z = (X_cond - mu) / sigma
    Z_cond = pca.transform(X_cond_z)

    sigma = 2  # in time bins

    from scipy.ndimage import gaussian_filter1d
    traj = gaussian_filter1d(
        Z_cond[:, :2],
        sigma=sigma,
        axis=0,
        mode='nearest'
    )

    pts = traj[:, :2]
    t = times[idx]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Colored trajectory
    segments = np.stack([pts[:-1], pts[1:]], axis=1)

    lc = LineCollection(
        segments,
        cmap='viridis',
        array=t[:-1],
        linewidth=3
    )

    ax.add_collection(lc)

    # Start / end
    ax.scatter(
        pts[0,0], pts[0,1],
        color='green',
        s=100,
        label='Start',
        zorder=5
    )

    ax.scatter(
        pts[-1,0], pts[-1,1],
        color='red',
        s=100,
        label='End',
        zorder=5
    )

    ax.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
    ax.set_title(condition)

    ax.axis('equal')
    ax.autoscale()
    ax.legend()

    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Time (s)")

    plt.tight_layout()
    plt.show()

    import scipy
    if condition == 'ZF A_20db_180ms_8b_1xomit_8b_silence':
        tempo = 180
    elif condition == 'ZF A_20db_300ms_8b_1xomit_8b_silence':
        tempo = 300
    else:
        tempo = None

    if tempo is not None:
        sr, sound = scipy.io.wavfile.read(r"C:\Users\tmerri03\Desktop\RhythmStimuli\Awake Recs\Raw Files\ZF A_20db.wav")
        beat_duration = len(sound) / sr
        gap_duration = tempo / 1000 - beat_duration
        omission_duration = gap_duration + tempo / 1000
        beat_times_1 = np.arange(0, 8) * tempo / 1000 + omission_duration
        baseline = beat_times_1[0] - tempo / 1000 - np.arange(0, 1) * tempo / 1000
        omission_times = np.arange(1) * tempo / 1000 + tempo / 1000 + beat_times_1[-1]
        beat_times_2 = np.arange(0, 8) * tempo / 1000 + tempo / 1000 + omission_times[-1]
        offset_times = np.arange(0, 2) * tempo / 1000 + tempo / 1000 + beat_times_2[-1]
        beat_times = np.concatenate([beat_times_1, beat_times_2])
        all_times = np.concatenate([baseline, beat_times_1, omission_times, beat_times_2, offset_times])

    template, covariance, relative_time, beat_segments = make_beat_template(Z_cond, times, all_times)
    comparison_df = compare_beats_to_template(Z_cond, times, all_times, beat_duration, template, covariance)
    fig = plot_individual_beats(template, comparison_df)
    plt.show()

def make_beat_template(Z, time, beat_times,
                       pre=0.05,
                       post=0.18):

        dt = np.median(np.diff(time))

        n_pre = int(round(pre / dt))
        n_post = int(round(post / dt))

        beat_segments = []

        for beat in beat_times:

            center = np.argmin(np.abs(time - beat))

            start = center - n_pre
            stop = center + n_post + 1

            if start < 0 or stop > len(time):
                continue

            beat_segments.append(Z[start:stop])

        beat_segments = np.stack(beat_segments)

        template = beat_segments.mean(axis=0)

        covariance = np.array([
            np.cov(beat_segments[:, i, :].T)
            for i in range(template.shape[0])
        ])

        relative_time = np.arange(
            -n_pre,
            n_post + 1
        ) * dt

        return (
            template,
            covariance,
            relative_time,
            beat_segments
        )

def plot_individual_beats(
                            template,
                            comparison_df,
                            ncols=4,
                            figsize=(14, 8)):

                        import numpy as np
                        import matplotlib.pyplot as plt

                        nbeats = len(comparison_df)

                        nrows = int(np.ceil(nbeats / ncols))

                        fig, axes = plt.subplots(
                            nrows,
                            ncols,
                            figsize=figsize,
                            squeeze=False
                        )

                        axes = axes.ravel()

                        for ax, (_, row) in zip(axes, comparison_df.iterrows()):

                            seg = row["trajectory"]

                            # Beat trajectory
                            ax.plot(
                                seg[:, 0],
                                seg[:, 1],
                                color="steelblue",
                                lw=2,
                                label="Trajectory"
                            )

                            # Template
                            ax.plot(
                                template[:, 0],
                                template[:, 1],
                                color="red",
                                lw=3,
                                label="Template",
                                alpha=0.3
                            )

                            # -------------------------------------------------
                            # Draw physical beat interval
                            # -------------------------------------------------

                            beat_mask = row["beat_mask"]

                            if np.any(beat_mask):

                                beat_seg = seg[beat_mask]

                                ax.plot(
                                    beat_seg[:, 0],
                                    beat_seg[:, 1],
                                    color="black",
                                    lw=1,
                                    solid_capstyle="round",
                                    zorder=10,
                                )

                                if len(beat_seg) > 1:
                                    ax.annotate(
                                        "",
                                        xy=(beat_seg[-1, 0], beat_seg[-1, 1]),
                                        xytext=(beat_seg[-2, 0], beat_seg[-2, 1]),
                                        arrowprops=dict(
                                            arrowstyle="->",
                                            lw=2,
                                            color="black",
                                        ),
                                        zorder=11,
                                    )

                            # -------------------------------------------------

                            # Template start/end
                            ax.scatter(
                                template[0, 0],
                                template[0, 1],
                                color="green",
                                s=35
                            )

                            ax.scatter(
                                template[-1, 0],
                                template[-1, 1],
                                color="red",
                                s=35
                            )

                            # Segment start/end
                            ax.scatter(
                                seg[0, 0],
                                seg[0, 1],
                                color="lime",
                                marker="x",
                                s=35
                            )

                            ax.scatter(
                                seg[-1, 0],
                                seg[-1, 1],
                                color="darkred",
                                marker="x",
                                s=35
                            )

                            ax.set_title(
                                f"{row.beat_time:.2f} s",
                                fontsize=10
                            )

                            txt = (
                                f"r={row.correlation:.2f}\n"
                                f"RMSE={row.rmse:.2f}\n"
                                f"Mah={row.mahalanobis:.2f}\n"
                                f"Lag={row.lag_ms:.0f} ms"
                            )

                            ax.text(
                                0.03,
                                0.97,
                                txt,
                                transform=ax.transAxes,
                                ha="left",
                                va="top",
                                fontsize=8,
                                bbox=dict(
                                    facecolor="white",
                                    alpha=0.8,
                                    edgecolor="none"
                                )
                            )

                            ax.set_xlabel("PC1")
                            ax.set_ylabel("PC2")
                            ax.set_aspect("equal", adjustable="datalim")

                        # Remove unused axes
                        for ax in axes[nbeats:]:
                            ax.remove()

                        handles, labels = axes[0].get_legend_handles_labels()
                        fig.legend(
                            handles,
                            labels,
                            loc="upper right"
                        )

                        fig.tight_layout()

                        return fig

def compare_beats_to_template(
        Z,
        time,
        beat_times,
        beat_duration,
        template,
        covariance,
        pre=0.05,
        post=0.18,
):
    import numpy as np
    import pandas as pd

    dt = np.median(np.diff(time))

    n_pre = int(round(pre / dt))
    n_post = int(round(post / dt))

    segments = []
    labels = []
    beat_masks = []

    for beat in beat_times:

        center = np.argmin(np.abs(time - beat))

        start = center - n_pre
        stop = center + n_post + 1

        if start < 0 or stop > len(time):
            continue

        segment = Z[start:stop]
        window_time = time[start:stop]

        # Samples during the physical beat
        beat_mask = (
                (window_time >= beat) &
                (window_time <= beat + beat_duration)
        )

        segments.append(segment)
        labels.append(beat)
        beat_masks.append(beat_mask)

    segments = np.stack(segments)

    from scipy.spatial.distance import mahalanobis
    from scipy.signal import correlate, correlation_lags

    rows = []

    template_flat = template.reshape(-1)

    template_energy = np.sum(
        np.linalg.norm(template, axis=1)
    )

    for beat_time, segment, beat_mask in zip(labels, segments, beat_masks):

        # ---------- RMSE ----------

        rmse = np.sqrt(
            np.mean((segment - template) ** 2)
        )

        # ---------- Correlation ----------

        corr = np.corrcoef(
            segment.reshape(-1),
            template_flat
        )[0, 1]

        # ---------- Energy ----------

        energy = np.sum(
            np.linalg.norm(segment, axis=1)
        )

        energy_ratio = energy / template_energy

        # ---------- Mahalanobis ----------

        mahal = []

        for x, mu, cov in zip(
                segment,
                template,
                covariance
        ):
            cov = cov + np.eye(cov.shape[0]) * 1e-6

            inv = np.linalg.inv(cov)

            mahal.append(
                mahalanobis(
                    x,
                    mu,
                    inv
                )
            )

        mahal = np.asarray(mahal)

        mahal_mean = mahal.mean()

        # ---------- Lag ----------

        x = segment[:, 0]
        y = template[:, 0]

        x = x - x.mean()
        y = y - y.mean()

        xc = correlate(x, y)

        lag = correlation_lags(
            len(x),
            len(y)
        )[np.argmax(xc)]

        lag_ms = lag * dt * 1000

        rows.append({

            "beat_time": beat_time,

            "rmse": rmse,

            "correlation": corr,

            "energy_ratio": energy_ratio,

            "mahalanobis": mahal_mean,

            "lag_ms": lag_ms,

            "mahal_profile": mahal,

            "trajectory": segment,

            "beat_mask": beat_mask

        })

    return pd.DataFrame(rows)

def main():
    db_path = r"C:\Users\tmerri03\Desktop\Temp Neural Files\awake_recordings.db"
    stimuli_path = r'R:\Data\tyler\Recordings\Stim\Stimuli Library'
    recordings_path = r'N:\Data\RhythmPerception\Neural Recordings'

    cluster, db = get_neurons(db_path, stimuli_path)
    for c in cluster.keys():
        common_stims = find_common_stim(recordings_path, cluster.get(c), db)
        pca_result = fit_neural_space(recordings_path, cluster.get(c), common_stims, c, db=db)
        plot_conditions(pca_result)

if __name__ == '__main__':
    main()