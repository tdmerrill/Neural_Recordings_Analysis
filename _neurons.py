from SiNAPSE.core import Neuron, Recording
import _generate_database, _tones, _plot

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

class EvalQuality:
    def __init__(self, stim_lib_path, recordings_path, recording, db, ntrials=20):
        self.ntrials=ntrials

        bird = recording.split(' ')[0]
        recording_path = os.path.join(recordings_path, bird, recording)
        self.rec = Recording(recording_path, samplerate=30000, db=db)
        self.db = db

    def evaluate_neurons(self, recording):
        select_columns = ['unit_id', 'session_id']
        conditions = {
            'manual_isi_0_7': ('<', 1),
            'session_id': ('=', recording),
        }
        units = self.db.load_neurons_from_database(select_columns, conditions)
        for unit, _ in units:
            N = Neuron(recording, unit, rec=self.rec, db=self.db)
            stims = N.load

            best_window_start = self.evaluate_stability(N=N, stims=stims, ntrials=self.ntrials)
            _generate_database.insert(DB_PATH=self.db.db_path,
                                      unit_id=unit,
                                      session_id=recording,
                                      value=int(best_window_start),
                                      column=f'best_{self.ntrials}trials_window_start')

    @staticmethod
    def evaluate_stability(N, stims, alpha=0.5, beta=0.5, ntrials=20):
        combined_spikes = None
        for stimulus in stims:
            _, baseline_spikes, _ = N.raster(
                stimulus,
                baseline=True,
                plot=False,
                padding=0,
                separate_trials=True
            )

            if combined_spikes is None:
                # first iteration → just copy structure
                combined_spikes = [list(trial) for trial in baseline_spikes]
            else:
                # concatenate trial-wise
                for i, trial in enumerate(baseline_spikes):
                    combined_spikes[i].extend(trial)

        # Now combined_spikes contains all baseline spikes across all stimuli, trial-wise
        # You can now analyze combined_spikes to evaluate stability
        n_trials = len(combined_spikes)
        scores=[]
        for start in range(n_trials - ntrials + 1):
            window = combined_spikes[start:start + ntrials]

            window = np.array([len(trial) for trial in window])

            med = np.median(window)
            mad = np.median(np.abs(window - med)) + 1e-9

            slope = np.polyfit(np.arange(len(window)), window, 1)[0]

            z = (window - med) / mad
            n_outliers = np.sum(np.abs(z) > 3)
            score = mad + alpha * np.abs(slope) + beta * n_outliers
            scores.append(score)

        if len(scores) == 0:
            import warnings
            warnings.warn("Not enough trials to evaluate stability.")
            best_window_start = 0
        else:
            best_window_start = np.argmin(scores)
        return best_window_start

class Response(EvalQuality):
    def __init__(self, stim_lib_path, recordings_path, recording, db, ntrials=20):
        super().__init__(stim_lib_path, recordings_path, recording, db, ntrials=ntrials)

        self.db = db
        self.ntrials=ntrials
        self.recordings_path = recordings_path
        self.recording=recording
        bird = recording.split(' ')[0]
        recording_path = os.path.join(recordings_path, bird, recording)
        self.rec = Recording(recording_path, samplerate=30000, db=db)

    # --- firing properies ---
    def evaluate_neurons(self):
        if not _generate_database.session_is_complete(DB_PATH=self.db.db_path, session_id=self.recording, column=f'best_{self.ntrials}trials_window_start'):
            super().evaluate_neurons(self.recording)
            print(f'Finding stable windows for: {self.recording}')
        else:
            print(f'Skipping: {self.recording}')

    def baseline_fr(self, duration=2, return_raster=False):
        select_columns = ['unit_id', 'session_id', f'best_{self.ntrials}trials_window_start']
        conditions = {
            'manual_isi_0_7': ('<', 1),
            'session_id': ('=', self.recording),
        }
        units = self.db.load_neurons_from_database(select_columns, conditions)

        baselines = {}
        for unit, _, window_start in units:
            baselines[unit] = {}
            N = Neuron(self.recording, unit, rec=self.rec, db=self.db)
            stims = N.load
            baseline_trials, ntrials, nstimuli = N.collect_baseline(duration=duration)
            trial_duration = duration * nstimuli

            trial_frs = [
                len(spikes) / trial_duration
                for spikes in baseline_trials
            ]
            mean_fr = np.mean(trial_frs)
            std_fr = np.std(trial_frs, ddof=1)

            baselines[unit]['mean'] = mean_fr
            baselines[unit]['std'] = std_fr

            if return_raster:
                baselines[unit]['raster'] = baseline_trials
                baselines[unit]['ntrials'] = ntrials
                baselines[unit]['nstimuli'] = nstimuli
                baselines[unit]['duration'] = duration
                baselines[unit]['trial_duration'] = trial_duration

        return baselines

    def compare_rebound(self, comparison_stim):

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from itertools import product

        select_columns = [
            'unit_id',
            'session_id',
            f'best_{self.ntrials}trials_window_start'
        ]

        conditions = {
            'manual_isi_0_7': ('<', 1),
            'session_id': ('=', self.recording),
        }

        units = self.db.load_neurons_from_database(
            select_columns,
            conditions
        )

        baselines = self.baseline_fr()

        rows = []

        # -------------------------------------------------
        # Run comparison for every neuron and every pair
        # -------------------------------------------------
        for unit, recording, window_start in units:

            N = Neuron(
                self.recording,
                unit,
                rec=self.rec,
                db=self.db
            )
            stims=N.load

            # include A vs A, A vs B, etc.
            for stim1, stim2 in product(comparison_stim, repeat=2):

                try:
                    result = self.compare(
                        stim1,
                        stim2,
                        N=N,
                        baseline=baselines[unit]
                    )

                    rows.append({
                        'unit_id': unit,
                        'stim1': stim1,
                        'stim2': stim2,

                        'ratio': result['ratio'],
                        'difference': result['difference'],
                        'correlation': result['correlation'],
                        'xcorr_lag_ms': result['xcorr_lag_ms'],
                        'latency_difference_ms':
                            result['latency_difference_ms']
                    })

                except Exception as e:
                    print(
                        f'Failed: unit {unit}, '
                        f'{stim1} vs {stim2}: {e}'
                    )

        df = pd.DataFrame(rows)

        # -------------------------------------------------
        # Plotting
        # -------------------------------------------------

        metrics = [
            ('ratio', 'Response Ratio', 1),
            ('difference', 'Response Difference', 0),
            ('correlation', 'Waveform Correlation', 0),
            ('xcorr_lag_ms', 'XCorr Lag (ms)', 0),
            ('latency_difference_ms',
             'Peak Latency Difference (ms)', 0)
        ]

        pairs = list(product(
            comparison_stim,
            comparison_stim
        ))

        n_pairs = len(pairs)

        ncols = int(np.ceil(np.sqrt(n_pairs)))
        nrows = int(np.ceil(n_pairs / ncols))

        for metric, ylabel, reference in metrics:

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(4 * ncols, 3 * nrows),
                squeeze=False
            )

            axes = axes.flatten()

            fig.suptitle(
                ylabel,
                fontsize=16
            )

            for ax, (stim1, stim2) in zip(axes, pairs):
                subset = df[
                    (df['stim1'] == stim1) &
                    (df['stim2'] == stim2)
                    ]

                # sort neurons for easier visualization
                subset = subset.sort_values(metric)

                x = np.arange(len(subset))

                ax.scatter(
                    x,
                    subset[metric],
                    s=20
                )

                ax.axhline(
                    reference,
                    linestyle='--',
                    color='black',
                    linewidth=1
                )

                ax.set_title(
                    f'{stim1}\nvs\n{stim2}',
                    fontsize=9
                )

                ax.set_xlabel('Neuron')
                ax.set_ylabel(ylabel)

                ax.set_xticks([])

            # remove unused axes
            for ax in axes[len(pairs):]:
                ax.remove()

            plt.tight_layout()
            plt.show()

        return df

    def compare(self, stim1, stim2, N=None, baseline=None, padding=0):
        if N is None or baseline is None:
            return
        data = {}
        for stimulus in [stim1, stim2]:
            _, trial_spikes, duration = N.raster(
                stimulus, ax=None, plot=False, padding=padding
            )
            _, trial_spikes_separate, _ = N.raster(
                stimulus, ax=None, plot=False, padding=padding, separate_trials=True
            )
            _, [psth_time, psth_data] = N.psth(
                stimulus, trial_spikes, ax=None,
                color='blue', plot=False, padding=padding
            )
            psth_data = np.array(psth_data)
            psth_data = (psth_data - baseline['mean']) / baseline['std']

            #detect beat times
            _, stim_time, stim_data = N.plot_stimulus(stimulus, ax=None, plot=False, padding=padding)
            _, onsets, offsets = N.find_beat_times(stim_data=stim_data, stim_time=stim_time)

            if len(onsets) > 1:
                last_onset = onsets[7]
                last_offset = offsets[7]
            else:
                last_onset = onsets[0]
                last_offset = offsets[0]


            tempo_ms=180
            window_ms=1.5*tempo_ms
            window = window_ms/1000
            mask = (psth_time > last_offset) & (psth_time < last_offset + window)
            window_time = psth_time[mask] - last_offset
            psth_windowed = psth_data[mask]

            # Summary statistics
            avg_psth = np.mean(psth_windowed)
            peak_idx = np.argmax(psth_windowed)
            peak_value = psth_windowed[peak_idx]
            peak_latency = window_time[peak_idx]

            data[stimulus] = {
                'waveform': psth_windowed,
                'time': window_time,
                'mean_z': avg_psth,
                'peak_z': peak_value,
                'peak_latency': peak_latency,
                'last_onset': last_onset,
                'last_offset': last_offset,
                'tempo': tempo_ms / 1000,
            }

        avg1 = data[stim1]['mean_z']
        avg2 = data[stim2]['mean_z']
        ratio = avg2 / avg1
        modulation = (avg2 - avg1) / (abs(avg1) + abs(avg2))

        from scipy.signal import correlate, correlation_lags

        x = data[stim1]['waveform'].copy()
        y = data[stim2]['waveform'].copy()

        # Mean-center for shape comparison
        x0 = x - x.mean()
        y0 = y - y.mean()

        # Pearson correlation (shape similarity)
        corr = np.corrcoef(x0, y0)[0, 1]

        # Cross correlation
        xcorr = correlate(x0, y0, mode='full')
        lags = correlation_lags(len(x0), len(y0), mode='full')

        peak = np.argmax(xcorr)
        lag_bins = lags[peak]

        dt = data[stim1]['time'][1] - data[stim1]['time'][0]
        lag_ms = lag_bins * dt * 1000

        # Response metrics
        mean1 = data[stim1]['mean_z']
        mean2 = data[stim2]['mean_z']

        ratio = np.nan
        if np.abs(mean1) > 1e-6:
            ratio = mean2 / mean1

        difference = mean2 - mean1

        latency_difference = (
                data[stim2]['peak_latency']
                - data[stim1]['peak_latency']
        )

        return {
            'stim1': data[stim1],
            'stim2': data[stim2],

            'ratio': ratio,
            'difference': difference,

            'correlation': corr,

            'xcorr_peak': np.max(xcorr),
            'xcorr_lag_ms': lag_ms,

            'latency_difference_ms': latency_difference * 1000,
        }

    # --- response clustering ---
    def response_metrics(self):
        select_columns = ['unit_id', 'session_id', f'best_{self.ntrials}trials_window_start']
        conditions = {
            'manual_isi_0_7': ('<', 1),
            'session_id': ('=', self.recording),
        }
        units = self.db.load_neurons_from_database(select_columns, conditions)

        baseline=self.baseline_fr(duration=2)
        data = {
            'units': [],
            'recordings': [],
            'mean baseline FR': [],
            'std baseline FR': [],
            'pref tone response': [],
            'pref tone latency': [],
            'best frequency': []
        }
        for unit, recording, _ in units:
            data['units'].append(unit)
            data['recordings'].append(recording)
            data['mean baseline FR'].append(baseline[unit]['mean'])
            data['std baseline FR'].append(baseline[unit]['std'])
            tuning=_tones.analyze_tones(self.db.stim_library,
                                 self.recordings_path,
                                 session_id=self.recording,
                                 unit_id = unit,
                                 db=self.db,
                                 baseline_mean = baseline[unit]['mean'],
                                 baseline_std = baseline[unit]['std'],
                                 ntrials=20,
                                 buffer_ms=10)
            best_freq = max(tuning, key=lambda f: tuning[f]['zscore'])
            data['best frequency'].append(best_freq)
            data['pref tone response'].append(tuning[best_freq]['zscore'])
            data['pref tone latency'].append(tuning[best_freq]['latency'])

        cols = ['unit', 'recording', 'baseline fr', 'pcc', 'stimulus corr', 'latency', 'pref freq']
        metrics=pd.DataFrame(columns=cols)
        metrics['unit'] = data['units']
        metrics['recording'] = data['recordings']
        metrics['baseline fr'] = data['mean baseline FR']
        metrics['latency'] = data['pref tone latency']
        metrics['pref freq'] = data['best frequency']
        metrics['180ms IOI correlation'] = data['stimulus corr']

    # --- plotting ---
    @property
    def summary(self):
        select_columns = ['unit_id', 'session_id', f'best_{self.ntrials}trials_window_start']
        conditions = {
            'manual_isi_0_7': ('<', 1),
            'session_id': ('=', self.recording),
        }
        units = self.db.load_neurons_from_database(select_columns, conditions)

        baselines=self.baseline_fr()

        for unit, recording, window_start in units:
            bird = recording.split(' ')[0]
            save_path = rf'C:\Users\tmerri03\Desktop\Neural Data\Awake Plots\{bird}\Summary'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            N = Neuron(self.recording, unit, rec=self.rec, db=self.db)
            _plot.plot_summary(N=N, save_path=save_path, baseline=baselines[unit], window_start=int(window_start), ntrials=self.ntrials)


    @property
    def offsets(self):
        select_columns = ['unit_id', 'session_id', f'best_{self.ntrials}trials_window_start']
        conditions = {
            'manual_isi_0_7': ('<', 1),
            'session_id': ('=', self.recording),
        }
        units = self.db.load_neurons_from_database(select_columns, conditions)

        baselines = self.baseline_fr()

        for unit, recording, window_start in units:
            bird = recording.split(' ')[0]
            save_path = rf'C:\Users\tmerri03\Desktop\Neural Data\Awake Plots\{bird}\Offsets'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            N = Neuron(self.recording, unit, rec=self.rec, db=self.db)
            stims=N.load
            _plot.plot_offsets(N=N, save_path=save_path, baseline=baselines[unit], window_start=int(window_start),
                               ntrials=self.ntrials)

    @property
    def waveforms(self):
        select_columns = ['unit_id', 'session_id', f'best_{self.ntrials}trials_window_start']
        conditions = {
            'manual_isi_0_7': ('<', 1),
            'session_id': ('=', self.recording),
        }
        units = self.db.load_neurons_from_database(select_columns, conditions)
        bird = self.recording.split(' ')[0]
        save_path = rf'C:\Users\tmerri03\Desktop\Neural Data\Awake Plots\{bird}\Waveforms'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        _plot.plot_waveforms(rec=self.rec, units=units, save_path=save_path)

    @property
    def probe(self):
        select_columns = ['unit_id', 'session_id', 'probe', 'unit_loc_x', 'unit_loc_y']
        conditions = {
            'manual_isi_0_7': ('<', 1),
            'session_id': ('=', self.recording),
        }
        units = self.db.load_neurons_from_database(select_columns, conditions)
        bird = self.recording.split(' ')[0]
        save_path = rf'C:\Users\tmerri03\Desktop\Neural Data\Awake Plots\{bird}\Probe'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        _plot.plot_probe(rec=self.rec, units=units, save_path=save_path)

    # --- pca functions ---
    def neural_space(self, binsize_ms=10):
        # fit a neural space on the response strength of each condition
        # ends with m*n matrix:
            # m = time points (rows)
            # n = neurons (columns)

        #check if mat already exists
        save_path = os.path.join(__file__, '..', '.pca_mats', f'_{self.recording}.npz')
        if not os.path.exists(save_path):
            select_columns = ['unit_id', 'session_id', f'best_{self.ntrials}trials_window_start']
            conditions = {
                'manual_isi_0_7': ('<', 1),
                'session_id': ('=', self.recording),
            }

            responses=[]
            for unit, _, window_start in self.db.load_neurons_from_database(select_columns, conditions):
                condition_labels = []
                trial_labels = []
                time_labels = []

                window_start = int(window_start)

                N = Neuron(self.recording, unit, rec=self.rec, db=self.db)
                stimuli = N.load

                all_baseline_spikes = []
                all_trial_spikes = {}
                expected_rates = []
                for stimulus in stimuli:
                    _, trial_spikes, tduration = N.raster(stimulus, baseline=False, plot=False, padding=0, separate_trials=True)
                    if stimulus not in all_trial_spikes.keys():
                        all_trial_spikes[stimulus] = trial_spikes[window_start:window_start+self.ntrials]
                        all_trial_spikes[f'{stimulus}_duration'] = tduration
                    _, baseline_spikes, bduration = N.raster(stimulus, baseline=True, plot=False, padding=0, separate_trials=False)
                    all_baseline_spikes.extend(baseline_spikes[window_start:window_start+self.ntrials])

                    #calculate expected spikes per time bin
                    n_timebins = bduration / (binsize_ms / 1000)
                    n_spikes = len(baseline_spikes)
                    n_expected_per_trial = n_spikes / n_timebins / self.ntrials
                    expected_rates.append(n_expected_per_trial)
                expected_rate = np.array(expected_rates).mean()

                #build trial response matrix
                trial_data = []
                for stimulus in stimuli:
                    duration = all_trial_spikes[f'{stimulus}_duration']
                    for t, trial in enumerate(all_trial_spikes[stimulus]):
                        trial_spike_counts, _ = np.histogram(trial, bins=np.arange(0, duration + binsize_ms / 1000, binsize_ms / 1000))
                        trial_spike_rate = trial_spike_counts - expected_rate

                        trial_data.extend(trial_spike_rate)

                        trial_labels.extend([t]*len(trial_spike_counts))
                        time_labels.extend(np.arange(len(trial_spike_counts))*binsize_ms/1000)
                        condition_labels.extend([stimulus]*len(trial_spike_counts))
                response = np.array(trial_data).T
                responses.append(response)

            X = np.column_stack(responses)
            np.savez_compressed(
                save_path,
                X=X,
                condition_labels=condition_labels,
                trial_labels=trial_labels,
                time_labels=time_labels
            )

        #load data back in and fit neural space
        data = np.load(save_path)
        X = data['X']
        conditions = data['condition_labels']
        trials = data['trial_labels']
        times = data['time_labels']

        from sklearn.decomposition import PCA
        X_z = (X - X.mean(axis=0)) / X.std(axis=0)
        pca = PCA(n_components=10)  # or 10–20 depending on use
        Z = pca.fit_transform(X_z)

        print(f'Returning PCA results')
        return Z, X, pca, conditions, trials, times

    def project_condition(self, condition, pca_result):
        Z, X, pca, conditions, trials, times = pca_result

        print(f'Projecting {condition} condition into PCA space.')

        idx = np.where(conditions == condition)[0]
        idx = idx[np.argsort(times[idx])]
        X_cond = X[idx]
        X_cond_z = (X_cond - X_cond.mean()) / X_cond.std()
        Z_cond = pca.transform(X_cond_z)

        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        points = Z_cond[:, :2]
        t = np.arange(len(points))

        segments = np.stack([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            cmap="viridis",
            array=t[:-1],
            linewidth=2
        )

        plt.figure()
        plt.gca().add_collection(lc)

        plt.xlim(points[:, 0].min(), points[:, 0].max())
        plt.ylim(points[:, 1].min(), points[:, 1].max())

        mid = len(Z_cond) // 2
        plt.scatter(Z_cond[mid, 0], Z_cond[mid, 1], color='red', s=80, label='midpoint')
        plt.text(Z_cond[mid, 0], Z_cond[mid, 1], "mid", color='red')

        plt.colorbar(lc, label="time (bins)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Time-colored neural trajectory")
        plt.show()

class LatencyCalculator:
    """
    Computes spike-response latency using a bootstrap-derived,
    neuron-specific threshold from baseline spontaneous activity.
    """

    def __init__(self, N, bin_size=0.001, smooth_window=0.005):
        self.N = N
        self.bin_size = bin_size
        self.smooth_window = smooth_window

        win_len = int(smooth_window / bin_size)
        if win_len % 2 == 0:
            win_len += 1
        self.hanning_win = np.hanning(win_len)
        self.hanning_win /= self.hanning_win.sum()

        # populated by fit_baseline()
        self.threshold = None
        self.null_peaks = None
        self.baseline_mean = None
        self.baseline_std = None

    def _smooth_psth(self, spike_times, search_window):
        bins = np.arange(0, search_window + self.bin_size, self.bin_size)
        counts, _ = np.histogram(spike_times, bins=bins)
        fr = counts / self.bin_size
        fr_smooth = np.convolve(fr, self.hanning_win, mode='same')
        bin_centers = bins[:-1] + self.bin_size / 2
        return bin_centers, fr_smooth

    def fit_baseline(self, baseline_trials, search_window=0.100,
                      n_bootstrap=1000, percentile=99):
        """
        Builds the empirical null distribution from baseline spike trains
        and sets self.threshold accordingly.
        """
        n_trials = len(baseline_trials)
        null_peaks = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
            trial_idx = np.random.randint(n_trials)
            spikes = np.asarray(baseline_trials[trial_idx])

            if len(spikes) == 0:
                null_peaks[b] = 0
                continue

            max_start = max(spikes.min(), spikes.max() - search_window)
            window_start = np.random.uniform(spikes.min(), max(max_start, spikes.min()))
            window_end = window_start + search_window

            windowed_spikes = spikes[
                (spikes >= window_start) & (spikes < window_end)
            ] - window_start

            _, fr_smooth = self._smooth_psth(windowed_spikes, search_window)
            null_peaks[b] = fr_smooth.max()

        self.null_peaks = null_peaks
        self.threshold = np.percentile(null_peaks, percentile)

        # also store raw mean/std for z-scoring/plotting purposes
        trial_frs = [len(s) / search_window for s in baseline_trials]
        self.baseline_mean = np.mean(trial_frs)
        self.baseline_std = np.std(trial_frs, ddof=1)

        return self

    def calculate_latency(self, response_spike_times, search_window=0.100,
                           min_consecutive_bins=1):
        """
        Returns latency (s) for a single response spike train (pooled or
        single-trial), using the threshold set by fit_baseline().
        """
        if self.threshold is None:
            raise RuntimeError("Call fit_baseline() before calculate_latency().")

        bin_centers, fr_smooth = self._smooth_psth(response_spike_times, search_window)
        above_threshold = fr_smooth > self.threshold

        for i in range(len(above_threshold) - min_consecutive_bins + 1):
            if np.all(above_threshold[i:i + min_consecutive_bins]):
                return bin_centers[i]

        return None

    def plot(self, response_spike_times, search_window=0.100, latency=None, ax=None):
        bin_centers, fr_smooth = self._smooth_psth(response_spike_times, search_window)

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(bin_centers, fr_smooth, label='smoothed PSTH')
        if self.threshold is not None:
            ax.axhline(self.threshold, color='red', linestyle='--', label='bootstrap threshold')
        if latency is not None:
            ax.axvline(latency, color='green', linestyle='--', label=f'latency = {latency*1000:.1f} ms')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing rate (Hz)')
        ax.legend()
        plt.show()
        return ax
