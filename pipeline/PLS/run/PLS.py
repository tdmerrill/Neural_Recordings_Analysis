from SiNAPSE.core import Database, Recording, Neuron
import os
import numpy as np
import pandas as pd

def find_beat_times(wav_path, plot=False, return_time_only=False):
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy

    fs, stim_data = scipy.io.wavfile.read(wav_path)
    stim_data = stim_data/np.max(stim_data)
    stim_time = np.arange(stim_data.shape[0]) / fs

    if return_time_only:
        return None, None, None, stim_time, None

    # ====== DETECT BEATS =======
    audio_deriv = np.gradient(stim_data)

    from scipy.signal import find_peaks

    peaks_pos, _ = find_peaks(audio_deriv, height=0.05)  # rising edges
    peaks_neg, _ = find_peaks(-audio_deriv, height=0.05)  # falling edges
    all_edges = np.sort(np.concatenate((peaks_pos, peaks_neg)))

    edge_times = stim_time[all_edges]
    edge_times = np.sort(edge_times)

    grouped_edges = []
    current_group = [edge_times[0]]

    for time in edge_times[1:]:
        if time - current_group[-1] <= 0.005:  # 5 ms in seconds
            current_group.append(time)
        else:
            grouped_edges.append(current_group)
            current_group = [time]

    grouped_edges.append(current_group)

    rising_edges_times = []
    falling_edges_times = []
    for group in grouped_edges:
        start = group[0]
        stop = group[-1]

        if stop - start > 0.01:
            rising_edges_times.append(start)
            falling_edges_times.append(stop)

    if plot:
        fig, ax = plt.subplots(1,1,figsize=(8,8))
        ax.plot(stim_time, stim_data)
        for l in rising_edges_times:
            ax.axvline(l, color='g')
        for l in falling_edges_times:
            ax.axvline(l, color='r')
        plt.show()

    tempo = np.mean(np.diff(rising_edges_times))
    return rising_edges_times, falling_edges_times, tempo, stim_time, stim_data
def get_onsets_tempo_omission(filepath):
    """
    Parse beat onset times, tempo, omission location, and beat duration.

    Parameters
    ----------
    filepath : str
        Path to the stimulus metadata/log file.

    Returns
    -------
    onset_times : list of float
        Expected onset times of all beats, including omitted beats.

    tempo : float
        IOI in seconds.

    omission_idx : int
        0-based index of the omitted beat.

    omission_time : float
        Expected onset time of the omitted beat in seconds.

    beat_duration : float
        Duration of the beat sound in seconds.

    sample_rate : int
        Sample rate of the beat WAV file.
    """
    import re
    import numpy as np
    from scipy.io import wavfile
    with open(filepath, "r") as f:
        text = f.read()

    # -------------------------
    # Tempo
    # -------------------------
    tempo_match = re.search(
        r"Tempo \(IOI\):\s*([\d.]+)\s*s",
        text
    )

    if tempo_match is None:
        tempo=None
        # print('Warning: Could not find Tempo (IOI) in file.')
    else:
        tempo = float(tempo_match.group(1))

    # -------------------------
    # Beat sound path
    # -------------------------
    sound_match = re.search(
        r"Beat sound:\s*(.+)",
        text
    )

    if sound_match is None:
        raise ValueError("Could not find Beat sound path.")

    sound_path = sound_match.group(1).strip()

    # -------------------------
    # Read beat WAV
    # -------------------------
    sample_rate, sound = wavfile.read(sound_path)

    # Convert stereo to mono if necessary
    if sound.ndim > 1:
        sound = sound.mean(axis=1)

    # -------------------------
    # Determine sound duration
    # -------------------------
    # Use a threshold relative to the maximum amplitude.
    amplitude = np.abs(sound.astype(float))
    threshold = 0.01 * amplitude.max()

    active = np.where(amplitude > threshold)[0]

    if len(active) == 0:
        raise ValueError("Could not detect sound in beat WAV.")

    first_sample = active[0]
    last_sample = active[-1]

    beat_duration = (last_sample - first_sample + 1) / sample_rate

    # -------------------------
    # Beat onset times
    # -------------------------
    onset_times = []
    omission_idx = None

    for line in text.splitlines():

        match = re.match(
            r"^\s*(pre|omitted|post)\s+\d+\s+\d+\s+([\d.]+)",
            line
        )

        if match:
            segment = match.group(1)
            onset = float(match.group(2))

            if segment == "omitted":
                omission_idx = len(onset_times)

            onset_times.append(onset)

    if not onset_times:
        raise ValueError("No beat onset times found.")

    if omission_idx is None:
        raise ValueError("No omitted beat found.")

    omission_time = onset_times[omission_idx]

    return (
        onset_times,
        tempo,
        omission_idx,
        omission_time,
        beat_duration,
        sample_rate,
    )

def construct_basis(n_bins, n_basis):
    t = np.arange(n_bins)

    # Centers distributed across the history window
    centers = np.linspace(0, n_bins - 1, n_basis)

    # Width chosen so neighboring bases overlap
    width = centers[1] - centers[0]

    B = np.zeros((n_bins, n_basis))

    for i, center in enumerate(centers):
        x = (t - center) / width

        B[:, i] = np.where(
            np.abs(x) <= 1,
            0.5 * (1 + np.cos(np.pi * x)),
            0
        )
    return B

def baseline_fr(N, duration=2, binsize=1/1000):
    baseline_trials, ntrials, nstimuli = N.collect_baseline(duration=duration)
    trial_duration = duration * nstimuli

    bins = np.arange(0, trial_duration + binsize, binsize)

    # histogram each trial's baseline spikes into 1ms bins, convert to Hz
    binned_rates = []
    for spikes in baseline_trials:
        counts, _ = np.histogram(spikes, bins=bins)
        rate = counts / binsize   # Hz, per trial, per bin
        binned_rates.append(rate)

    binned_rates = np.stack(binned_rates)  # shape (ntrials, nbins)

    # trial-averaged rate, matching how `rate` is computed in prepare()
    trial_avg_rate = binned_rates.mean(axis=0)  # shape (nbins,)

    mean_fr = np.mean(trial_avg_rate)
    std_fr = np.std(trial_avg_rate, ddof=1)

    return mean_fr, std_fr

STIM_ALIASES = {
    "exp1_ZF A_20db_180ms_8b_1xomit_8b_silence": [
        "ZF A_20db_180ms_8b_1xomit_8b_silence"
    ]
}


def get_stim_aliases(stim):
    return {stim, *STIM_ALIASES.get(stim, [])}


def has_required_stimuli(stims, required_stims):
    stims = set(stims)

    for required in required_stims:
        aliases = get_stim_aliases(required)

        if not stims.intersection(aliases):
            return False

    return True


def get_actual_stim_names(requested_stims, available_stims):
    actual_stims = []

    for requested in requested_stims:
        aliases = get_stim_aliases(requested)

        match = next(
            (stim for stim in available_stims if stim in aliases),
            None
        )

        if match is not None:
            actual_stims.append(match)

    return actual_stims


class PLS:
    def __init__(self, DB_PATH,
            RECORDINGS_PATH,
            STIM_LIB_PATH,
            save_dir,
            REGION,
            PLS_TRAINING_STIM,
            PLS_TESTING_STIM):
        self.DB_PATH = DB_PATH
        self.RECORDINGS_PATH = RECORDINGS_PATH
        self.STIM_LIB_PATH = STIM_LIB_PATH
        self.REGION = REGION
        self.PLS_TRAINING_STIM = PLS_TRAINING_STIM
        self.PLS_TESTING_STIM = PLS_TESTING_STIM
        self.save_dir = save_dir

    def prepare(self, top_neurons, stimuli='training', jitter=None, pre=0):
        db = Database(self.DB_PATH, self.STIM_LIB_PATH)

        if jitter is not None:
            X_save_path = os.path.join(self.save_dir, f'X_{self.REGION}_{stimuli}_jitter.npy')
            Y_save_path = os.path.join(self.save_dir, f'Y_{self.REGION}_{stimuli}_jitter.npy')
            T_save_path = os.path.join(self.save_dir, f'T_{self.REGION}_{stimuli}_jitter.npy')
            RATE_save_path = os.path.join(self.save_dir, f'RATE_{self.REGION}_{stimuli}_jitter.npy')
            print(f'Adding some jitter (+- {jitter}s) to the spiking data.')
        else:
            X_save_path = os.path.join(self.save_dir, f'X_{self.REGION}_{stimuli}.npy')
            Y_save_path = os.path.join(self.save_dir, f'Y_{self.REGION}_{stimuli}.npy')
            T_save_path = os.path.join(self.save_dir, f'T_{self.REGION}_{stimuli}.npy')
            RATE_save_path = os.path.join(self.save_dir, f'RATE_{self.REGION}_{stimuli}.npy')

        if (not os.path.exists(X_save_path)) and (not os.path.exists(Y_save_path)) and (not os.path.exists(T_save_path)):
            print(f'--- Generating {stimuli} PLS Response Matrices for {self.REGION} ---')
            print(f'')

            X = None
            Y = None
            T = None
            RATE = None
            calc_Y = True
            n_neurons = 0
            all_stims = []
            for r, recording in enumerate(pd.unique(top_neurons['recording'])):
                region = self.REGION.replace(' ', '').replace('_', '').lower()
                recording_name = recording.replace(' ', '').replace('_', '').lower()

                if region not in recording_name:
                    continue

                bird = recording.split(' ')[0]
                rec_fp = os.path.join(self.RECORDINGS_PATH, bird, recording)
                rec = Recording(rec_fp, samplerate=30000, db=db)

                _valid_recording = True
                recording_subset = top_neurons[top_neurons['recording'] == recording]

                sw = [] #hold spike widths
                for ridx, row in recording_subset.iterrows():
                    session = row['recording']
                    unit = int(row['unit'])

                    if _valid_recording:
                        N = Neuron(session, unit, rec=rec, db=db)
                        stims=N.load

                        if not has_required_stimuli(stims, self.PLS_TRAINING_STIM):
                            _valid_recording = False
                            continue

                        if not has_required_stimuli(stims, self.PLS_TESTING_STIM):
                            _valid_recording = False
                            continue

                        #compute baseline after checking stim to save time
                        baseline_mean, baseline_std = baseline_fr(N, duration=2)
                        if baseline_std == 0 or np.isnan(baseline_std):
                            _valid_recording = False
                            print(
                                f'Warning: Baseline std is 0 or NaN for {session} unit {unit}. Skipping this recording.')
                            continue

                        X_n = None
                        Y_n = None
                        T_n = None
                        RATE_n = None
                        if stimuli == 'training':
                            stim_set = self.PLS_TRAINING_STIM
                        else:
                            stim_set = [stimuli]

                        stim_set = get_actual_stim_names(stim_set, stims)

                        # save the p2p spike width for later
                        sw.append(N.get_neuron_data['spike_width_pp'])

                        for training_stimulus in stim_set:
                            # 1. Create per-neuron response vector (len(time), 1)
                            padding = 1
                            _, spikes, duration = N.raster(training_stimulus, baseline=False, plot=False, ax=None, padding=padding)
                            _, trials, _ = N.raster(training_stimulus, baseline=True, plot=False, ax=None, separate_trials=True, padding=0)
                            n_trials = len(trials)

                            if jitter is not None:
                                spikes = spikes + np.random.uniform(-jitter, jitter, size=len(spikes))

                            wav_fp = os.path.join(self.STIM_LIB_PATH, f'{training_stimulus}.wav')
                            _, _, _, stimulus_time, _ = find_beat_times(wav_path=wav_fp, plot=False)

                            dat_fp_base = r'C:\Users\tmerri03\Desktop\RhythmStimuli\Tempi Range Stimuli 8-25-26\Stim Files\File Data'
                            dat_fp = os.path.join(dat_fp_base, f'{training_stimulus}_onsets.txt')
                            try:
                                if not training_stimulus == 'exp1_ZF A_20db_180ms_8b_1xomit_8b_silence':
                                    dat_fp = os.path.join(dat_fp_base, 'exp1_ZF A_20db_180ms_8b_1xomit_8b_silence_onsets.txt')
                                onset_times, tempo, omission_location, omission_time, beat_duration, sr = get_onsets_tempo_omission(dat_fp)
                            except:
                                #need to hard code some things because these files dont have beat time description text files.
                                if '180' in training_stimulus:
                                    onset_times = np.array([
                                        0.279,
                                        0.459,
                                        0.639,
                                        0.819,
                                        0.999,
                                        1.179,
                                        1.359,
                                        1.539,
                                        # omission: expected next beat at 1.719
                                        1.818,
                                        1.998,
                                        2.178,
                                        2.358,
                                        2.538,
                                        2.718,
                                        2.898,
                                        3.078,
                                    ])
                                    tempo = 0.180
                                elif '300' in training_stimulus:
                                    onset_times = np.array([
                                        0.519,
                                        0.819,
                                        1.119,
                                        1.419,
                                        1.719,
                                        2.019,
                                        2.319,
                                        2.619,
                                        # omission: expected next beat at 2.919
                                        3.138,
                                        3.438,
                                        3.738,
                                        4.038,
                                        4.338,
                                        4.638,
                                        4.938,
                                        5.238,
                                    ])
                                    tempo = 0.300
                                beat_duration = 0.081
                                omission_location = 7

                            onset_times = np.asarray(onset_times)
                            offset_times = onset_times + beat_duration
                            first_onset = onset_times[0]
                            onset_times = onset_times - first_onset
                            offset_times = offset_times - first_onset
                            stimulus_time = stimulus_time - first_onset
                            spikes = np.asarray(spikes) - first_onset
                            binsize=1/1000 #1ms binsize (same as Bao Le, et al.)

                            if len(onset_times) > 1:
                                if stimuli == 'training':
                                    mask = ((stimulus_time >= -0.3) & (stimulus_time < onset_times[omission_location-1]))
                                    stimulus_time_masked = stimulus_time[mask]
                                    stop = np.max(stimulus_time_masked)
                                else:
                                    stop = np.ceil(duration + padding)
                            else:
                                stop = np.ceil(duration + padding)

                            bins = np.arange(-0.3-pre, stop + binsize, binsize)
                            rate, bin_times = np.histogram(spikes, bins=bins)
                            rate = rate / binsize / n_trials #normalize to hz per single trial
                            rate_z = (rate-baseline_mean)/baseline_std

                            # 2. Create Hankel matrix (len(time), window_size)
                            from numpy.lib.stride_tricks import sliding_window_view
                            window_size = 90/1000  # ~90 ms window (similar to Bao Le, et al. --> 100ms)
                            window_samples = int(window_size / binsize)
                            H = sliding_window_view(rate_z, window_samples)
                            RATE_n_s_full = rate_z[window_samples - 1:]

                            # 3. Project onto 15 raised-cosine basis (len(time), n_basis * n_neurons)
                            B = construct_basis(np.shape(H)[1], n_basis=15) #15 raised cosine basis (Bao Le, et al.)
                            X_n_s_full = H @ B

                            bin_centers = (bin_times[:-1] + bin_times[1:]) / 2
                            response_times_full = bin_centers[window_samples - 1:]

                            # Trim to onset-relative time >= 0 — baseline only existed to give the
                            # window real context, not to be analyzed itself
                            valid = response_times_full >= -pre
                            X_n_s = X_n_s_full[valid]
                            response_times = response_times_full[valid]
                            RATE_n_s = RATE_n_s_full[valid]

                            if X_n is None:
                                X_n = X_n_s
                            else:
                                X_n = np.concatenate((X_n, X_n_s), axis=0)

                            if RATE_n is None:
                                RATE_n = RATE_n_s
                            else:
                                RATE_n = np.concatenate((RATE_n, RATE_n_s), axis=0)

                            # 4. Generate response matrix per neuron (len(time), N_targets) --> (len(time), [sin(), cos())
                            if calc_Y:
                                #not best practice, but give tempo a dummy value so it runs --> we don't save Y anyway unless training
                                #where tempo will always have a value
                                if tempo is None:
                                    tempo = 0.180
                                beat_idx = np.searchsorted(onset_times, response_times, side='right') - 1
                                phase = (2 * np.pi  * (response_times - onset_times[beat_idx]) / tempo
                                )
                                Y_n_s = np.column_stack([np.cos(phase), np.sin(phase)])

                                if Y_n is None:
                                    Y_n = Y_n_s
                                    T_n = response_times
                                else:
                                    Y_n = np.concatenate((Y_n, Y_n_s), axis=0)
                                    T_n = np.concatenate((T_n, response_times), axis=0)

                        if Y is None:
                            Y = Y_n
                            T = T_n
                            calc_Y = False

                        if X is None:
                            X = X_n
                        else:
                            X = np.concatenate((X, X_n), axis=1)

                        if RATE is None:
                            RATE = RATE_n[np.newaxis, :]  # shape (1, time)
                        else:
                            RATE = np.vstack((RATE, RATE_n))  # shape (n_neurons, time)

                        all_stims.append(stims)
                        n_neurons += 1

            if X is not None:
                os.makedirs(self.save_dir, exist_ok=True)
                np.save(X_save_path, X)
                if stimuli == 'training':
                    np.save(Y_save_path, Y)
                np.save(T_save_path, T)
                np.save(RATE_save_path, RATE)

                print(f'Saved output matrices to {self.save_dir}')
                # print(f'You may use these stim: {set(all_stims[0]).intersection(*all_stims[1:])}')
                print(f'')

                import matplotlib.pyplot as plt
                if stimuli == 'training':
                    plt.hist(sw, bins=25)
                    plt.xlabel("Value")
                    plt.ylabel("Count")
                    plt.show()

    def fit(self):
        X = np.load(os.path.join(self.save_dir, f'X_{self.REGION}_training.npy'))
        Y = np.load(os.path.join(self.save_dir, f'Y_{self.REGION}_training.npy'))

        print(' --- Training Inputs: --- ')
        print(f'    | Neurons: {int(np.shape(X)[1] / 15)}')
        print(f'    | X Shape: {np.shape(X)}')
        print(f'    | Y Shape: {np.shape(Y)}')
        print(f'    \u2514\u2192 Training Partial Least Squares Regression Model...')
        print(f'')

        from sklearn.cross_decomposition import PLSRegression
        self.pls = PLSRegression(n_components=10, scale=False)
        self.pls.fit(X, Y)

    # def trajectory(self, top_neurons, stimulus, plot_phase=False, jitter=None, pre=0):
    #     self.prepare(top_neurons, stimuli=stimulus, jitter=jitter, pre=pre)
    #
    #     if jitter is None:
    #         X = np.load(os.path.join(self.save_dir, f'X_{self.REGION}_{stimulus}.npy'))
    #         # Y = np.load(os.path.join(self.save_dir, f'Y_{self.REGION}_{stimulus}.npy'))
    #         Y=None #don't need Y for testing
    #         T = np.load(os.path.join(self.save_dir, f'T_{self.REGION}_{stimulus}.npy'))
    #     else:
    #         X = np.load(os.path.join(self.save_dir, f'X_{self.REGION}_{stimulus}_jitter.npy'))
    #         # Y = np.load(os.path.join(self.save_dir, f'Y_{self.REGION}_{stimulus}_jitter.npy'))
    #         Y=None #see above
    #         T = np.load(os.path.join(self.save_dir, f'T_{self.REGION}_{stimulus}_jitter.npy'))
    #     return self.pls.transform(X), Y, T
    def trajectory(self, top_neurons, stimulus, plot_phase=False, jitter=None, pre=0):

        self.prepare(top_neurons, stimuli=stimulus, jitter=jitter, pre=pre)

        if jitter is None:
            X = np.load(
                os.path.join(
                    self.save_dir,
                    f'X_{self.REGION}_{stimulus}.npy'
                )
            )
            Y = None
            T = np.load(
                os.path.join(
                    self.save_dir,
                    f'T_{self.REGION}_{stimulus}.npy'
                )
            )

        else:
            X = np.load(
                os.path.join(
                    self.save_dir,
                    f'X_{self.REGION}_{stimulus}_jitter.npy'
                )
            )
            Y = None
            T = np.load(
                os.path.join(
                    self.save_dir,
                    f'T_{self.REGION}_{stimulus}_jitter.npy'
                )
            )

        Z = self.pls.transform(X)

        return Z, Y, T

    def rate(self, top_neurons, stimulus, pre=0):
        import numpy as np
        self.prepare(top_neurons, stimuli=stimulus, pre=pre)

        RATE = np.load(
            os.path.join(self.save_dir, f'RATE_{self.REGION}_{stimulus}.npy')
        )
        T = np.load(
            os.path.join(self.save_dir, f'T_{self.REGION}_{stimulus}.npy')
        )

        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 6))

        # Use percentile limits so a few very high FR values don't dominate
        vmin = np.percentile(RATE, 1)
        vmax = np.percentile(RATE, 99)

        im = ax.imshow(
            RATE,
            aspect='auto',
            origin='upper',
            extent=[T[0], T[-1], RATE.shape[0], 0],
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Firing rate (Hz)')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron')
        ax.set_title(f'{self.REGION} — {stimulus}')

        plt.tight_layout()
        plt.show()

    def contribution(self, component=1):
        contrib = self.pls.x_rotations_[:, component-1]  # component 1
        order = np.argsort(-np.abs(contrib))
        print(f'Neuron contributions to component {component}:')
        print(order)