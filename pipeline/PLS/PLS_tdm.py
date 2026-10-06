from SiNAPSE.core import Recording, Neuron, Database
import os
import numpy as np

def find_beat_times(wav_path, plot=False):
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy

    fs, stim_data = scipy.io.wavfile.read(wav_path)
    stim_data = stim_data/np.max(stim_data)
    stim_time = np.arange(stim_data.shape[0]) / fs

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

def prediction_response_matrix(stimulus, plot_stimulus=False, plotted=False, last_beat=0):
    RECORDINGS_PATH = r'R:\Data\RhythmPerception\Neural Recordings\Recordings'
    DB_PATH = r'C:\Users\tmerri03\Desktop\Temp Neural Files\awake_recordings.db'
    STIM_LIB_PATH = r'R:\Data\tyler\Recordings\Stim\Stimuli Library'

    db = Database(DB_PATH, STIM_LIB_PATH)
    recordings = sorted(db.recordings)

    X = None
    Y = None
    calc_Y = True
    n_neurons = 0
    for r, recording in enumerate(recordings):
        bird = recording.split(' ')[0]
        rec_fp = os.path.join(RECORDINGS_PATH, bird, recording)
        rec = Recording(rec_fp, samplerate=30000, db=db)

        select_columns = ['session_id', 'unit_id']
        conditions = {
            'manual_isi_0_7': ('<', 1),
            'session_id': ('=', recording),
        }
        units = db.load_neurons_from_database(select_columns, conditions)
        _valid_recording = True
        for session, unit in units:
            if _valid_recording:
                N = Neuron(session, unit, rec=rec, db=db)
                stims = N.load

                if not len(set(PLS_TRAINING_STIM) & set(stims)) == len(PLS_TRAINING_STIM):
                    _valid_recording = False
                    continue

                if not len(set(PLS_TESTING_STIM) & set(stims)) == len(PLS_TESTING_STIM):
                    _valid_recording = False
                    continue

                X_n = None
                Y_n = None

                # 1. Create per-neuron response vector (len(time), 1)
                _, spikes, duration = N.raster(stimulus, baseline=False, plot=False, ax=None,
                                               padding=0)
                _, trials, _ = N.raster(stimulus, baseline=True, plot=False, ax=None,
                                        separate_trials=True, padding=0)
                n_trials = len(trials)

                import PLS_helper
                tempo=0.18
                wav_fp = os.path.join(STIM_LIB_PATH, f'{stimulus}.wav')
                onset_times, offset_times, _, stimulus_time, stimulus_data = find_beat_times(wav_path=wav_fp, plot=False)
                onset_times = np.asarray(onset_times)
                offset_times = np.asarray(offset_times)
                first_onset = onset_times[0]
                onset_times = onset_times - first_onset
                offset_times = offset_times - first_onset
                stimulus_time = stimulus_time - first_onset
                # mask = ((stimulus_time >= 0) & (stimulus_time < onset_times[-1] + tempo / 2))
                mask = (stimulus_time >= 0) & (stimulus_time <= offset_times[-1-last_beat])
                stimulus_time_short = stimulus_time[mask]

                binsize = 1 / 1000  # 1ms binsize (same as Bao Le, et al.)
                bins = np.arange(0, np.max(stimulus_time_short) + binsize, binsize)
                rate, bin_times = np.histogram(spikes, bins=bins)

                if n_trials == 0:
                    print('error! no trials for this stimulus...')
                rate = rate / binsize / n_trials  # normalize to hz per single trial

                if plot_stimulus and not plotted:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(2,1,figsize=(10,10),sharex=True)
                    axes[0].plot(stimulus_time, stimulus_data)
                    axes[1].plot(stimulus_time_short, stimulus_data[mask])
                    plt.show()
                    plotted=True

                # 2. Create Hankel matrix (len(time), window_size)
                from numpy.lib.stride_tricks import sliding_window_view
                window_size = tempo / 2  # ~90 ms window (similar to Bao Le, et al. --> 100ms)
                window_samples = int(window_size / binsize)
                H = sliding_window_view(rate, window_samples)

                # 3. Project onto 15 raised-cosine basis (len(time), n_basis * n_neurons)
                B = construct_basis(np.shape(H)[1], n_basis=15)  # 15 raised cosine basis (Bao Le, et al.)
                X_n_s = H @ B

                if X_n is None:
                    X_n = X_n_s
                else:
                    X_n = np.concatenate((X_n, X_n_s), axis=0)

                # 4. Generate response matrix per neuron (len(time), N_targets) --> (len(time), [sin(), cos())
                if calc_Y:
                    bin_centers = (bin_times[:-1] + bin_times[1:]) / 2
                    response_times = bin_centers[window_samples - 1:]
                    beat_idx = np.searchsorted(
                        onset_times,
                        response_times,
                        side='right'
                    ) - 1
                    phase = (2 * np.pi * (response_times - onset_times[beat_idx]) / tempo
                             )
                    Y_n_s = np.column_stack([np.cos(phase), np.sin(phase)])

                    if Y_n is None:
                        Y_n = Y_n_s
                    else:
                        Y_n = np.concatenate((Y_n, Y_n_s), axis=0)

                if Y is None:
                    Y = Y_n
                    calc_Y = False

                if X is None:
                    X = X_n
                else:
                    X = np.concatenate((X, X_n), axis=1)

                n_neurons += 1

    print(f'--- Stimulus Response Matrix: {stimulus} ---')
    print(f'    Analyzed: {n_neurons} neurons')
    print(f'    X: {np.shape(X)}')
    print(f'    Y: {np.shape(Y)}')
    print('')
    return X, Y

PLS_TRAINING_STIM = ['ZF A_20db_180ms_18b_full_silence',
                     '2khz_20db_180ms_18b_full_silence',
                    ]
PLS_TESTING_STIM = ['ZF A_20db_180ms_18b_full_silence',
                    'ZF A_20db_180ms_8b_1xomit_8b_silence',
                    'ZF A irregular control']
REGION = 'Field L'

RECORDINGS_PATH = r'R:\Data\RhythmPerception\Neural Recordings\Recordings'
DB_PATH = r'C:\Users\tmerri03\Desktop\Temp Neural Files\awake_recordings.db'
STIM_LIB_PATH = r'R:\Data\tyler\Recordings\Stim\Stimuli Library'
save_dir = r'C:\Users\tmerri03\Desktop\Temp Neural Files\PLS'

db = Database(DB_PATH, STIM_LIB_PATH)
recordings = sorted(db.recordings)

if not os.path.exists(os.path.join(save_dir, f'X_{REGION}.npy')) and not os.path.exists(os.path.join(save_dir, f'Y_{REGION}.npy')):
    print(f'--- Generating PLS Response Matrices for {REGION} ---')
    print(f'')

    X = None
    Y = None
    calc_Y = True
    n_neurons = 0
    for r, recording in enumerate(recordings):
        region = REGION.replace(' ', '').replace('_', '').lower()
        recording_name = recording.replace(' ', '').replace('_', '').lower()

        if region not in recording_name:
            continue

        bird = recording.split(' ')[0]
        rec_fp = os.path.join(RECORDINGS_PATH, bird, recording)
        rec = Recording(rec_fp, samplerate=30000, db=db)

        select_columns = ['session_id', 'unit_id']
        conditions = {
            'manual_isi_0_7': ('<', 1),
            'session_id': ('=', recording),
        }
        units = db.load_neurons_from_database(select_columns, conditions)
        _valid_recording = True
        for session, unit  in units:
            if _valid_recording:
                N = Neuron(session, unit, rec=rec, db=db)
                stims=N.load

                if not len(set(PLS_TRAINING_STIM) & set(stims)) == len(PLS_TRAINING_STIM):
                    _valid_recording=False
                    continue

                if not len(set(PLS_TESTING_STIM) & set(stims)) == len(PLS_TESTING_STIM):
                    _valid_recording=False
                    continue

                X_n = None
                Y_n = None
                for training_stimulus in PLS_TRAINING_STIM:
                    # 1. Create per-neuron response vector (len(time), 1)
                    _, spikes, duration = N.raster(training_stimulus, baseline=False, plot=False, ax=None, padding=0)
                    _, trials, _ = N.raster(training_stimulus, baseline=True, plot=False, ax=None, separate_trials=True, padding=0)
                    n_trials = len(trials)

                    import PLS_helper
                    wav_fp = os.path.join(STIM_LIB_PATH, f'{training_stimulus}.wav')
                    onset_times, offset_times, tempo, stimulus_time, _ = find_beat_times(wav_path=wav_fp, plot=False)
                    onset_times = np.asarray(onset_times)
                    offset_times = np.asarray(offset_times)
                    first_onset = onset_times[0]
                    onset_times = onset_times - first_onset
                    offset_times = offset_times - first_onset
                    stimulus_time = stimulus_time - first_onset
                    mask = ((stimulus_time >= 0) & (stimulus_time < onset_times[-1]))
                    stimulus_time = stimulus_time[mask]
                    binsize=1/1000 #1ms binsize (same as Bao Le, et al.)
                    bins = np.arange(0, np.max(stimulus_time)+binsize, binsize)
                    rate, bin_times = np.histogram(spikes, bins=bins)
                    rate = rate / binsize / n_trials #normalize to hz per single trial

                    # 2. Create Hankel matrix (len(time), window_size)
                    from numpy.lib.stride_tricks import sliding_window_view
                    window_size = tempo/2  # ~90 ms window (similar to Bao Le, et al. --> 100ms)
                    window_samples = int(window_size / binsize)
                    H = sliding_window_view(rate, window_samples)

                    # 3. Project onto 15 raised-cosine basis (len(time), n_basis * n_neurons)
                    B = construct_basis(np.shape(H)[1], n_basis=15) #15 raised cosine basis (Bao Le, et al.)
                    X_n_s = H @ B

                    if X_n is None:
                        X_n = X_n_s
                    else:
                        X_n = np.concatenate((X_n, X_n_s), axis=0)

                    # 4. Generate response matrix per neuron (len(time), N_targets) --> (len(time), [sin(), cos())
                    if calc_Y:
                        bin_centers = (bin_times[:-1] + bin_times[1:]) / 2
                        response_times = bin_centers[window_samples - 1:]
                        beat_idx = np.searchsorted(
                            onset_times,
                            response_times,
                            side='right'
                        ) - 1
                        phase = (2 * np.pi  * (response_times - onset_times[beat_idx])/ tempo
                        )
                        Y_n_s = np.column_stack([np.cos(phase), np.sin(phase)])

                        if Y_n is None:
                            Y_n = Y_n_s
                        else:
                            Y_n = np.concatenate((Y_n, Y_n_s), axis=0)


                if Y is None:
                    Y = Y_n
                    calc_Y = False

                if X is None:
                    X = X_n
                else:
                    X = np.concatenate((X, X_n), axis=1)

                n_neurons += 1

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'X_{REGION}.npy'), X)
    np.save(os.path.join(save_dir, f'Y_{REGION}.npy'), Y)

    print(f'Saved output matrices to {save_dir}')
    print(f'')

#load back in matrices and fit model -- not very intensive compared to reloading data from source
X = np.load(os.path.join(save_dir, f'X_{REGION}.npy'))
Y = np.load(os.path.join(save_dir, f'Y_{REGION}.npy'))

print(' --- Training Inputs: --- ')
print(f'    | Neurons: {int(np.shape(X)[1]/15)}')
print(f'    | X Shape: {np.shape(X)}')
print(f'    | Y Shape: {np.shape(Y)}')
print(f'    \u2514\u2192 Training Partial Least Squares Regression Model...')
print(f'')

def phase_from_Y(Y):
    return np.arctan2(Y[:, 1], Y[:, 0])

def circular_error(pred, true):
    return np.angle(np.exp(1j * (pred - true)))

def plot_pls_prediction(true_phase, pred_phase, title):
    import numpy as np
    import matplotlib.pyplot as plt

    if true_phase is not None:
        # Make sure the arrays have matching lengths
        n = min(len(true_phase), len(pred_phase))
        true_phase = np.asarray(true_phase[:n])
        pred_phase = np.asarray(pred_phase[:n])

        error = circular_error(pred_phase, true_phase)
    else:
        error = None

    fig = plt.figure(figsize=(12, 12))

    # ============================================================
    # 1. True vs predicted phase
    # ============================================================
    ax1 = fig.add_subplot(3, 1, 1)

    if true_phase is not None:
        ax1.plot(
            true_phase,
            label='True phase'
        )

    ax1.plot(
        pred_phase,
        label='PLS predicted phase',
        alpha=0.7
    )

    ax1.set_ylabel('Phase (rad)')
    ax1.set_title(title)
    ax1.legend()

    # ============================================================
    # 2. Phase error
    # ============================================================
    ax2 = fig.add_subplot(3, 1, 2)

    if error is not None:
        ax2.plot(np.degrees(error))

        ax2.axhline(
            0,
            linestyle='--'
        )

        ax2.set_ylabel('Phase error (degrees)')
        ax2.set_xlabel('Time bin')

    else:
        ax2.text(
            0.5,
            0.5,
            'No true phase available',
            ha='center',
            va='center',
            transform=ax2.transAxes
        )

        ax2.set_axis_off()

    # ============================================================
    # 3. Polar phase trajectory
    # ============================================================
    ax3 = fig.add_subplot(
        3, 1, 3,
        projection='polar'
    )

    # Plot true phase
    if true_phase is not None:
        ax3.plot(
            true_phase,
            np.ones_like(true_phase),
            label='True phase',
            linewidth=2
        )

    # Plot predicted phase
    ax3.plot(
        pred_phase,
        np.ones_like(pred_phase),
        label='PLS predicted phase',
        alpha=0.7,
        linewidth=2
    )

    ax3.set_title('Phase trajectory')

    # Put 0 rad at the top and make phase increase clockwise
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)

    ax3.set_yticks([])

    ax3.legend(
        loc='upper right',
        bbox_to_anchor=(1.25, 1.15)
    )

    plt.tight_layout()
    plt.show()

    # ============================================================
    # Print error
    # ============================================================
    if error is not None:
        print(
            "Mean phase error:",
            np.degrees(np.mean(np.abs(error))),
            "degrees"
        )

        print(
            "Median phase error:",
            np.degrees(np.median(np.abs(error))),
            "degrees")


from sklearn.cross_decomposition import PLSRegression
pls = PLSRegression(n_components=10, scale=True)
pls.fit(X, Y)


# analyze omission & irregular stimuli
X_omit, _ = prediction_response_matrix('ZF A_20db_180ms_8b_1xomit_8b_silence', plot_stimulus=False)
X_regular, _ = prediction_response_matrix('ZF A_20db_180ms_18b_full_silence', last_beat=1, plot_stimulus=False)
X_jittered, _ = prediction_response_matrix('ZF A irregular control')

Z_omit = pls.transform(X_omit)
Z_regular = pls.transform(X_regular)
Z_jittered = pls.transform(X_jittered)

#find beat times for each stim
import scipy
import numpy as np
import matplotlib.pyplot as plt

# 1. Regular
stimulus = 'ZF A_20db_180ms_18b_full_silence'
stim_path = os.path.join(STIM_LIB_PATH, f'{stimulus}.wav')
fs, regular_data = scipy.io.wavfile.read(stim_path)
rising_edges_times, falling_edges_times, _, _, _ = find_beat_times(wav_path=stim_path)
first_beat_regular = rising_edges_times[0]
rising_edges_times -= rising_edges_times[0]
falling_edges_times -= falling_edges_times[0]
beat_times_regular = rising_edges_times
omission_beat_regular = beat_times_regular[8]

#2. Omission
stimulus = 'ZF A_20db_180ms_8b_1xomit_8b_silence'
stim_path = os.path.join(STIM_LIB_PATH, f'{stimulus}.wav')
fs, omission_data = scipy.io.wavfile.read(stim_path)
rising_edges_times, falling_edges_times, _, _, _ = find_beat_times(wav_path=stim_path)
first_beat_omission = rising_edges_times[0]
rising_edges_times -= rising_edges_times[0]
falling_edges_times -= falling_edges_times[0]
diffs = np.diff(rising_edges_times)
omission_loc = np.where(diffs>1.5*np.mean(diffs))[0][0]
omission_beat_time = rising_edges_times[omission_loc] + np.mean(diffs[:5])
rising_edges_times = sorted(np.append(rising_edges_times, omission_beat_time))
beat_times_omission = rising_edges_times
omission_beat_omit = beat_times_omission[8]

# 3. Irregular
stimulus = 'ZF A irregular control'
stim_path = os.path.join(STIM_LIB_PATH, f'{stimulus}.wav')
fs, irregular_data = scipy.io.wavfile.read(stim_path)
rising_edges_times, falling_edges_times, _, _, _ = find_beat_times(wav_path=stim_path)
first_beat_jittered = rising_edges_times[0]
rising_edges_times -= rising_edges_times[0]
falling_edges_times -= falling_edges_times[0]
omission_beat_time = rising_edges_times[omission_loc] + np.mean(diffs[:5])
rising_edges_times = sorted(np.append(rising_edges_times, omission_beat_time))
beat_times_jittered  = rising_edges_times
omission_beat_jittered = rising_edges_times[8]

# 4. Plot to make sure it's correct
fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(np.arange(len(regular_data))/fs-first_beat_regular, regular_data)
for t in beat_times_regular:
    axes[0].axvline(t)
axes[0].axvline(omission_beat_regular, color='g', linestyle='--')

axes[1].plot(np.arange(len(omission_data))/fs-first_beat_omission, omission_data)
for t in beat_times_omission:
    axes[1].axvline(t)
axes[1].axvline(omission_beat_omit, color='g', linestyle='--')

axes[2].plot(np.arange(len(irregular_data))/fs-first_beat_jittered, irregular_data)
for t in beat_times_jittered:
    axes[2].axvline(t)
axes[2].axvline(omission_beat_jittered, color='g', linestyle='--')
plt.show()

# 5. Plot trajectories
def plot_3d_trajectory_variable_width(
    ax,
    trajectory,
    color,
    min_width=0.5,
    max_width=4,
    label=None
):
    """
    Plot a 3D trajectory with line width increasing over time.
    """

    trajectory = np.asarray(trajectory)

    if trajectory.shape[0] < 2:
        return

    # Normalize time from 0 -> 1
    t = np.linspace(0, 1, trajectory.shape[0])

    widths = min_width + (max_width - min_width) * t

    for i in range(len(trajectory) - 1):

        ax.plot(
            trajectory[i:i+2, 0],
            trajectory[i:i+2, 1],
            trajectory[i:i+2, 2],
            color=color,
            linewidth=widths[i],
            label=label if i == 0 else None
        )

def plot_beat_by_beat_trajectories(
    Z_regular,
    Z_omit,
    Z_jittered,
    beat_times_regular,
    beat_times_omission,
    beat_times_jittered,
    binsize=0.001,
    n_cols=4
):

    # ---------------------------------------------------------
    # Time axes
    # ---------------------------------------------------------

    time_regular = np.arange(Z_regular.shape[0]) * binsize
    time_omit = np.arange(Z_omit.shape[0]) * binsize
    time_jittered = np.arange(Z_jittered.shape[0]) * binsize

    # ---------------------------------------------------------
    # Number of complete beat windows
    #
    # We need a following beat to define the end of the window.
    # ---------------------------------------------------------

    n_beats = min(
        len(beat_times_regular) - 1,
        len(beat_times_omission) - 1,
        len(beat_times_jittered) - 1
    )

    n_rows = int(np.ceil(n_beats / n_cols))

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    fig = plt.figure(
        figsize=(4.5 * n_cols, 4.5 * n_rows)
    )

    # ---------------------------------------------------------
    # First determine global axis limits
    # so trajectories are comparable across panels
    # ---------------------------------------------------------

    all_data = np.vstack([
        Z_regular[:, :3],
        Z_omit[:, :3],
        Z_jittered[:, :3]
    ])

    finite = np.all(np.isfinite(all_data), axis=1)
    all_data = all_data[finite]

    mins = np.min(all_data, axis=0)
    maxs = np.max(all_data, axis=0)

    # Make the limits slightly larger
    padding = 0.05 * (maxs - mins)

    mins -= padding
    maxs += padding

    # ---------------------------------------------------------
    # Plot each beat
    # ---------------------------------------------------------

    for i in range(n_beats):

        ax = fig.add_subplot(
            n_rows,
            n_cols,
            i + 1,
            projection='3d'
        )

        # =====================================================
        # REGULAR
        # =====================================================

        start = beat_times_regular[i]
        end = beat_times_regular[i + 1]

        mask = (
            (time_regular >= start) &
            (time_regular < end)
        )

        regular = Z_regular[mask, :3]

        if len(regular) > 1:
            plot_3d_trajectory_variable_width(
                ax,
                regular,
                color='blue',
                min_width=0.5,
                max_width=4,
                label='Regular'
            )

        # =====================================================
        # OMISSION
        # =====================================================

        start = beat_times_omission[i]
        end = beat_times_omission[i + 1]

        mask = (
            (time_omit >= start) &
            (time_omit < end)
        )

        omission = Z_omit[mask, :3]

        if len(omission) > 1:
            plot_3d_trajectory_variable_width(
                ax,
                omission,
                color='red',
                min_width=0.5,
                max_width=4,
                label='Omission'
            )

        # =====================================================
        # JITTERED
        # =====================================================

        start = beat_times_jittered[i]
        end = beat_times_jittered[i + 1]

        mask = (
            (time_jittered >= start) &
            (time_jittered < end)
        )

        jittered = Z_jittered[mask, :3]

        if len(jittered) > 1:
            plot_3d_trajectory_variable_width(
                ax,
                jittered,
                color='green',
                min_width=0.5,
                max_width=4,
                label='Jittered'
            )

        # =====================================================
        # Mark beat onset
        # =====================================================

        # Get starting points
        if len(regular) > 0:
            ax.scatter(
                regular[0, 0],
                regular[0, 1],
                regular[0, 2],
                color='blue',
                s=30
            )

        if len(omission) > 0:
            ax.scatter(
                omission[0, 0],
                omission[0, 1],
                omission[0, 2],
                color='red',
                s=30
            )

        if len(jittered) > 0:
            ax.scatter(
                jittered[0, 0],
                jittered[0, 1],
                jittered[0, 2],
                color='green',
                s=30
            )

        # =====================================================
        # Formatting
        # =====================================================

        ax.set_title(f'Beat {i + 1}')

        ax.set_xlabel('PLS 1')
        ax.set_ylabel('PLS 2')
        ax.set_zlabel('PLS 3')

        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])

        ax.legend()

    plt.tight_layout()

    return fig

fig = plot_beat_by_beat_trajectories(
    Z_regular,
    Z_omit,
    Z_jittered,
    beat_times_regular,
    beat_times_omission,
    beat_times_jittered,
    binsize=0.001,
    n_cols=4
)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
plot_3d_trajectory_variable_width(
    ax,
    Z_regular,
    color='blue',
)
plot_3d_trajectory_variable_width(
    ax,
    Z_jittered,
    color='green'
)
plot_3d_trajectory_variable_width(
    ax,
    Z_omit,
    color='red'
)
plt.show()