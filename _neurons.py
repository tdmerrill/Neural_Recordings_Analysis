from SiNAPSE.core import Neuron, Recording
import _generate_database, _tones, _plot
from neuron_clustering import extract_neuron_features, build_population_dataframe, cluster_neurons

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import scipy


from matplotlib.collections import LineCollection

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
    def __init__(self, stim_lib_path, recordings_path, recording, db, ntrials=20, output=None):
        super().__init__(stim_lib_path, recordings_path, recording, db, ntrials=ntrials)

        self.db = db
        self.ntrials=ntrials
        self.recordings_path = recordings_path
        self.recording=recording
        bird = recording.split(' ')[0]
        recording_path = os.path.join(recordings_path, bird, recording)
        self.rec = Recording(recording_path, samplerate=30000, db=db)

        if output is None:
            home_dir = os.path.expanduser('~')
            self.output_path = os.path.join(home_dir, '.Neural_Recordings_Analysis', 'outputs')
        else:
            self.output_path = output
        os.makedirs(self.output_path, exist_ok=True)

    # --- firing properies ---
    def evaluate_neurons(self):
        if not _generate_database.session_is_complete(DB_PATH=self.db.db_path, session_id=self.recording, column=f'best_{self.ntrials}trials_window_start'):
            super().evaluate_neurons(self.recording)
            print(f'Finding stable windows for: {self.recording}')
        else:
            print(f'Skipping: {self.recording}')

    def baseline_fr(self, duration=2, return_raster=False, binsize=1/1000):
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

            bins = np.arange(0, trial_duration + binsize, binsize)

            # histogram each trial's baseline spikes into 1ms bins, convert to Hz
            binned_rates = []
            for spikes in baseline_trials:
                counts, _ = np.histogram(spikes, bins=bins)
                rate = counts / binsize  # Hz, per trial, per bin
                binned_rates.append(rate)

            binned_rates = np.stack(binned_rates)  # shape (ntrials, nbins)

            # trial-averaged rate, matching how `rate` is computed in prepare()
            trial_avg_rate = binned_rates.mean(axis=0)  # shape (nbins,)

            mean_fr = np.mean(trial_avg_rate)
            std_fr = np.std(trial_avg_rate, ddof=1)

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
            save_path = rf'{self.output_path}\{bird}\Summary'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            N = Neuron(self.recording, unit, rec=self.rec, db=self.db)
            if window_start is None:
                window_start = 0
            _plot.plot_summary(N=N, save_path=save_path, baseline=baselines[unit], window_start=0, ntrials=self.ntrials)

    @property
    def stimuli(self):
        select_columns = ['unit_id', 'session_id', f'best_{self.ntrials}trials_window_start']
        conditions = {
            'manual_isi_0_7': ('<', 1),
            'session_id': ('=', self.recording),
        }
        units = self.db.load_neurons_from_database(select_columns, conditions)

        for unit, recording, window_start in units:
            N = Neuron(self.recording, unit, rec=self.rec, db=self.db)
            stims=N.load
            return stims

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
            save_path = rf'{self.output_path}\{bird}\Offsets'
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
        save_path = rf'{self.output_path}\{bird}\Waveforms'
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
        save_path = rf'{self.output_path}\{bird}\Probe'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        _plot.plot_probe(rec=self.rec, units=units, save_path=save_path)

    @property
    def find_omission_candidates(self):
        import omission_analysis

        select_columns = ['unit_id', 'session_id']
        conditions = {
            'manual_isi_0_7': ('<', 1),
            'session_id': ('=', self.recording),
        }
        units = self.db.load_neurons_from_database(select_columns, conditions)

        for unit, recording in units:
            omission_analysis.run(recording, unit, self.db.db_path, self.db.stim_library, self.recordings_path)

    @property
    def decompose(self):
        import omission_analysis

        select_columns = ['unit_id', 'session_id']
        conditions = {
            'manual_isi_0_7': ('<', 1),
            'session_id': ('=', self.recording),
        }
        units = self.db.load_neurons_from_database(select_columns, conditions)

        # for unit, recording in units:
        #     analysis = omission_analysis.run(recording, unit, self.db.db_path, self.db.stim_library, self.recordings_path)
        #     feats = extract_neuron_features(analysis)

        neurons = []
        for unit, recording in units:
            neurons.append(unit)

        raw_sounds_path=r'C:\\Users\\tmerri03\\Desktop\\RhythmStimuli\\Awake Recs\\Raw Files\\regex'
        stim_subset = [
            'ZF A_20db_180ms_8b_1xomit_8b_silence',
            # 'ZF A_20db_300ms_8b_1xomit_8b_silence',
            # '2khz_20db_180ms_8b_1xomit_8b_silence',
            # '2khz_20db_300ms_8b_1xomit_8b_silence',
        ]
        df = build_population_dataframe(neurons, self.db.db_path, self.db.stim_library, self.recordings_path,
                                   raw_sounds_path, recording, stim_subset)
        return df

    # --- PCA functions ---
    def neural_space(self, binsize_ms=10, padding=0):
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
                    _, trial_spikes, tduration = N.raster(stimulus, baseline=False, plot=False, padding=padding, separate_trials=True)
                    tduration += padding

                    if stimulus not in all_trial_spikes.keys():
                        all_trial_spikes[stimulus] = trial_spikes[window_start:window_start+self.ntrials]
                        all_trial_spikes[f'{stimulus}_duration'] = tduration
                    _, baseline_spikes, bduration = N.raster(stimulus, baseline=True, plot=False, padding=padding, separate_trials=False)
                    bduration += padding

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
        bird = self.recording.split(' ')[0]
        save_path = rf'{self.output_path}\{bird}\PCA'

        # if os.path.exists(f'{save_path}\{self.recording}_{condition}_neural_trajectory.png'):
        #     return

        import numpy as np

        Z, X, pca, conditions, trials, times = pca_result
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        print(f'Projecting {condition} condition into PCA space.')

        idx = np.where(conditions == condition)[0]
        idx = idx[np.argsort(times[idx])]
        X_cond = X[idx]
        X_cond_z = (X_cond - mu) / sigma
        Z_cond = pca.transform(X_cond_z)

        # plot trial-by-trial trajectory
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 7))

        unique_trials = np.unique(trials[idx])

        for trial in unique_trials:
            trial_idx = idx[trials[idx] == trial]

            # sort this trial by time
            trial_idx = trial_idx[np.argsort(times[trial_idx])]

            X_trial = X[trial_idx]
            X_trial_z = (X_trial - mu) / sigma
            Z_trial = pca.transform(X_trial_z)

            ax.plot(
                Z_trial[:, 0],
                Z_trial[:, 1],
                lw=2,
                alpha=0.6,
                label=f"Trial {trial}"
            )

            # mark start/end
            ax.scatter(Z_trial[0, 0], Z_trial[0, 1], s=30, c='green')
            ax.scatter(Z_trial[-1, 0], Z_trial[-1, 1], s=30, c='red')

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{condition} - Individual Trial Trajectories")

        # --- plot average trajectory ---
        from scipy.ndimage import gaussian_filter1d
        import matplotlib.pyplot as plt
        import numpy as np

        # Mean PSTH across trials
        unique_times = np.unique(times[idx])

        X_mean = []
        for t in unique_times:
            t_idx = idx[times[idx] == t]
            X_mean.append(X[t_idx].mean(axis=0))

        X_mean = np.vstack(X_mean)

        # Smooth each neuron independently
        sigma_bins = 1.5  # ~15 ms if binsize = 10 ms
        X_mean = gaussian_filter1d(
            X_mean,
            sigma=sigma_bins,
            axis=0,
            mode="nearest"
        )

        # Project
        X_mean_z = (X_mean - mu) / sigma
        Z_mean = pca.transform(X_mean_z)


        import scipy
        if condition == 'ZF A_20db_180ms_8b_1xomit_8b_silence':
            tempo = 180
        elif condition == 'ZF A_20db_300ms_8b_1xomit_8b_silence':
            tempo = 300
        else:
            tempo = None

        if tempo is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            sr, sound = scipy.io.wavfile.read(r"C:\Users\tmerri03\Desktop\RhythmStimuli\Awake Recs\Raw Files\ZF A_20db.wav")
            beat_duration = len(sound)/sr
            gap_duration = tempo/1000 - beat_duration
            omission_duration = gap_duration + tempo/1000
            beat_times_1 = np.arange(0,8) * tempo/1000 + omission_duration
            baseline = beat_times_1[0] - tempo/1000 - np.arange(0,1) * tempo/1000
            omission_times = np.arange(1) * tempo/1000 + tempo/1000 + beat_times_1[-1]
            beat_times_2 = np.arange(0, 8) * tempo/1000 + tempo/1000 + omission_times[-1]
            offset_times = np.arange(0,2) * tempo/1000 + tempo/1000 + beat_times_2[-1]
            beat_times = np.concatenate([beat_times_1, beat_times_2])
            all_times = np.concatenate([baseline, beat_times_1, omission_times, beat_times_2, offset_times])

            fig = self.plot_pca_summary(unique_times, Z_mean, condition)
            plt.savefig(
                os.path.join(save_path,
                f"{self.recording}_{condition}_neural_trajectory.png"),
                dpi=300,
                bbox_inches="tight"
            )

            template, covariance, relative_time, beat_segments = self.make_beat_template(Z_mean, unique_times, beat_times)
            res = self.compare_beats_to_template(Z_mean, unique_times, all_times, beat_duration, template, covariance)
            fig = self.plot_template_comparison(template, res)
            plt.savefig(f'{save_path}\{self.recording}_{condition}_template_comparison.png', dpi=300)
            fig = self.plot_individual_beats(template, res)
            plt.savefig(f'{save_path}\{self.recording}_{condition}_indvididual_beats.png', dpi=300)
            plt.close()

    def plot_conditions(self, conditions_to_plot, pca_result, sigma_bins=1.5):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from scipy.ndimage import gaussian_filter1d

        Z, X, pca, conditions, trials, times = pca_result

        mu = X.mean(axis=0)
        sigma = X.std(axis=0)

        fig, ax = plt.subplots(figsize=(8, 8))

        cmap = plt.get_cmap("tab10")

        for i, condition in enumerate(conditions_to_plot):

            idx = np.where(conditions == condition)[0]

            unique_times = np.unique(times[idx])

            # Average neural activity at each timepoint
            X_mean = []
            for t in unique_times:
                t_idx = idx[times[idx] == t]
                X_mean.append(X[t_idx].mean(axis=0))

            X_mean = np.vstack(X_mean)

            # Smooth each neuron
            X_mean = gaussian_filter1d(
                X_mean,
                sigma=sigma_bins,
                axis=0,
                mode="nearest"
            )

            # Project into PCA space
            X_mean_z = (X_mean - mu) / sigma
            Z_mean = pca.transform(X_mean_z)

            points = Z_mean[:, :2]

            color = cmap(i % 10)

            ax.plot(
                points[:, 0],
                points[:, 1],
                lw=3,
                color=color,
                label=condition
            )

            # start
            ax.scatter(
                points[0, 0],
                points[0, 1],
                color=color,
                edgecolor='k',
                marker='o',
                s=70
            )

            # end
            ax.scatter(
                points[-1, 0],
                points[-1, 1],
                color=color,
                edgecolor='k',
                marker='s',
                s=70
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Mean Neural Trajectories")

        ax.legend()

        plt.tight_layout()
        bird = self.recording.split(' ')[0]
        save_path = rf'{self.output_path}\{bird}\PCA'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f'{save_path}\{self.recording}_summary_neural_trajectory.png', dpi=300)
        plt.close()

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def plot_template_comparison(
            template,
            comparison_df,
            relative_time=None,
            figsize=(14, 8)
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=figsize)

        gs = GridSpec(
            2,
            5,
            figure=fig,
            width_ratios=[2.5, 1, 1, 1, 1],
            height_ratios=[3, 2]
        )

        # --------------------------------------------------------
        # Trajectory panel
        # --------------------------------------------------------

        ax_traj = fig.add_subplot(gs[0, 0])

        for _, row in comparison_df.iterrows():
            seg = row["trajectory"]

            ax_traj.plot(
                seg[:, 0],
                seg[:, 1],
                alpha=.35,
                lw=1.5
            )

        ax_traj.plot(
            template[:, 0],
            template[:, 1],
            color="red",
            lw=4,
            label="Template"
        )

        ax_traj.scatter(
            template[0, 0],
            template[0, 1],
            color="green",
            s=60,
            label="Start"
        )

        ax_traj.scatter(
            template[-1, 0],
            template[-1, 1],
            color="red",
            s=60,
            label="End"
        )

        ax_traj.set_title("Beat trajectories")
        ax_traj.set_xlabel("PC1")
        ax_traj.set_ylabel("PC2")
        ax_traj.legend()

        # --------------------------------------------------------
        # Heatmap
        # --------------------------------------------------------

        ax_heat = fig.add_subplot(gs[0, 1:])

        heat = np.vstack(
            comparison_df.mahal_profile.values
        )

        im = ax_heat.imshow(
            heat,
            aspect="auto",
            origin="lower",
            cmap="viridis"
        )

        ax_heat.set_title("Distance from template")

        ax_heat.set_ylabel("Beat")

        if relative_time is not None:
            ticks = np.linspace(
                0,
                len(relative_time) - 1,
                5,
                dtype=int
            )

            ax_heat.set_xticks(ticks)
            ax_heat.set_xticklabels(
                np.round(relative_time[ticks] * 1000).astype(int)
            )

            ax_heat.set_xlabel("Time relative to beat (ms)")

        plt.colorbar(
            im,
            ax=ax_heat,
            label="Mahalanobis distance"
        )

        # --------------------------------------------------------
        # Summary metrics
        # --------------------------------------------------------

        metrics = [
            ("correlation", "Correlation", 1),
            ("mahalanobis", "Mahalanobis", None),
            ("rmse", "RMSE", None),
            ("energy_ratio", "Energy", 1),
            ("lag_ms", "Lag (ms)", 0),
        ]

        for i, (col, title, ref) in enumerate(metrics):

            ax = fig.add_subplot(gs[1, i])

            y = comparison_df[col].values

            x = np.arange(len(y))

            ax.scatter(
                x,
                y,
                s=40
            )

            ax.plot(
                x,
                y,
                alpha=.3
            )

            if ref is not None:
                ax.axhline(
                    ref,
                    ls="--",
                    c="k",
                    lw=1
                )

            ax.set_title(title)

            ax.set_xticks(x)

            ax.set_xticklabels(
                np.arange(1, len(y) + 1)
            )

            ax.set_xlabel("Beat")

        plt.tight_layout()

        return fig

    @staticmethod
    def plot_individual_beats(
            template,
            comparison_df,
            ncols=4,
            figsize=(14, 8),
    ):

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

    @staticmethod
    def plot_pca_summary(unique_times, Z_mean, condition):
        # ---------- Plot ----------
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax_pc1 = axes[0, 0]
        ax_pc2 = axes[0, 1]
        ax_pc3 = axes[1, 0]
        ax_traj = axes[1, 1]

        # -----------------------
        # PC1, PC2, PC3 vs Time
        # -----------------------
        colors = ["tab:blue", "tab:orange", "tab:green"]

        for i, ax in enumerate([ax_pc1, ax_pc2, ax_pc3]):
            ax.plot(unique_times, Z_mean[:, i],
                    color=colors[i],
                    lw=2)

            ax.scatter(unique_times[0], Z_mean[0, i],
                       color="limegreen", s=60, zorder=3)

            ax.scatter(unique_times[-1], Z_mean[-1, i],
                       color="red", s=60, zorder=3)

            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"PC{i + 1}")
            ax.set_title(f"PC{i + 1} vs Time")

        # -----------------------
        # PC1-PC2 trajectory
        # -----------------------
        points = Z_mean[:, :2]
        segments = np.stack([points[:-1], points[1:]], axis=1)


        lc = LineCollection(
            segments,
            cmap="viridis",
            array=unique_times[:-1],
            linewidth=3
        )

        ax_traj.add_collection(lc)

        ax_traj.scatter(
            points[0, 0],
            points[0, 1],
            color="limegreen",
            s=80,
            label="Start"
        )

        ax_traj.scatter(
            points[-1, 0],
            points[-1, 1],
            color="red",
            s=80,
            label="End"
        )

        ax_traj.autoscale()

        cbar = fig.colorbar(lc, ax=ax_traj)
        cbar.set_label("Time (s)")

        ax_traj.set_xlabel("PC1")
        ax_traj.set_ylabel("PC2")
        ax_traj.set_title("Neural Trajectory")
        ax_traj.legend()

        fig.suptitle(f"{condition} Mean Neural Trajectory", fontsize=16)

        plt.tight_layout()

        return fig

    # --- dPCA functions ---
    def fit_dpca(self, stimulus):
        # fit a neural space on the response strength of each condition
        # ends with m*n matrix:
        # m = time points (rows)
        # n = neurons (columns)
        tempo = None
        if stimulus == 'ZF A_20db_180ms_8b_1xomit_8b_silence':
            tempo = 180
        elif stimulus == 'ZF A_20db_300ms_8b_3xomit_8b_silence':
            tempo = 300

        if tempo is None:
            print('Not a valid stimulus choice for dPCA')
            return

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
        expected_times = np.concatenate([omission_times, offset_times])

        # -- gather baseline frs --
        baselines = self.baseline_fr(duration=2)

        # check if mat already exists
        save_path = os.path.join(__file__, '..', '.dpca_mats', f'_{self.recording}_{stimulus}.npz')
        if not os.path.exists(save_path):

            select_columns = ['unit_id', 'session_id']
            conditions = {
                'manual_isi_0_7': ('<', 1),
                'session_id': ('=', self.recording),
            }

            padding=1 #add padding in case winows go over total stimulus time
            pre=0.1
            post=0.3
            binsize=0.01
            beat_times = beat_times + padding
            expected_times = expected_times + padding
            bins = np.arange(-pre, post + binsize, binsize)

            neurons = self.db.load_neurons_from_database(select_columns, conditions)
            responses = None
            for u, (unit, _) in enumerate(neurons):
                N = Neuron(self.recording, unit, rec=self.rec, db=self.db)
                stimuli = N.load
                _, trial_spikes, tduration = N.raster(stimulus, baseline=False, plot=False, padding=padding, separate_trials=True)

                if responses is None:
                    n_trials = len(trial_spikes)
                    n_bins = len(bins) - 1
                    n_neurons = len(neurons)

                    responses = np.zeros(
                        (n_neurons, 2, n_trials, n_bins),
                        dtype=float
                    )

                for t, trial in enumerate(trial_spikes):
                    beat_hists=[]
                    for beat in beat_times:
                        rel = trial-beat
                        rel = rel[(rel > -pre) & (rel < post)]
                        hist, edges = np.histogram(rel, bins=bins)
                        hist = hist-baselines[unit]['mean']*binsize
                        beat_hists.append(hist)
                    responses[u,0,t] = np.mean(beat_hists, axis=0)

                    expected_hists=[]
                    for expected in expected_times:
                        rel = trial-expected
                        rel = rel[(rel > -pre) & (rel < post)]
                        hist, edges = np.histogram(rel, bins=bins)
                        hist = hist-baselines[unit]['mean']*binsize
                        expected_hists.append(hist)
                    responses[u, 1, t] = np.mean(expected_hists, axis=0)
            np.savez_compressed(
                save_path,
                responses=responses,
                bins=bins,
                stimulus=stimulus,
                tempo=tempo,
                pre=pre,
                post=post,
                binsize=binsize
            )

        #continue with analysis
        data = np.load(save_path)
        responses = data['responses']
        bins = data['bins']
        stimulus = data['stimulus']
        tempo = data['tempo']
        pre = data['pre']
        post = data['post']
        binsize = data['binsize']


        mean_beat = responses.mean(axis=(0, 2))[0]
        mean_omit = responses.mean(axis=(0, 2))[1]

        print(f'Responses Shape: {responses.shape}')

        from dPCA import dPCA
        X = responses.mean(axis=2)
        X -= X.mean(axis=(1, 2), keepdims=True)
        print(f'X Shape: {X.shape}')
        dpca = dPCA.dPCA(
            labels='ct',
            regularizer=None
        )
        Z = dpca.fit_transform(X)

        print(f'Z Keys: {Z.keys()}')

        plt.figure(figsize=(6, 6))

        plt.plot(
            Z['ct'][0, 0],
            Z['ct'][1, 0],
            marker='o',
            label='Beat'
        )

        plt.plot(
            Z['ct'][0, 1],
            Z['ct'][1, 1],
            marker='o',
            label='Omission'
        )

        plt.xlabel('dPCA component 1')
        plt.ylabel('dPCA component 2')
        plt.legend()
        plt.axis('equal')
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
