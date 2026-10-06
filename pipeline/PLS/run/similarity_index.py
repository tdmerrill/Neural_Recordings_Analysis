RECORDINGS_PATH = r'R:\Data\RhythmPerception\Neural Recordings\Recordings'
STIM_LIB_PATH = r'R:\Data\tyler\Recordings\Stim\Stimuli Library'
save_dir = r'C:\Users\tmerri03\Desktop\Temp Neural Files\PLS'

FILTER_STIM = {'regular': 'ZF A_20db_180ms_18b_full_silence',
               'omission': 'ZF A_20db_180ms_8b_1xomit_8b_silence'}
REGION = 'Field L'

# --- develop similarity index ---
# 1. take neurons with the full and omission stim
# 2. count the spikes in the omission period
# 3. calculate the expected baseline spikes in the same time frame
# 4. calculate the similarity index SI = (R_baseline - R_beat) / (R_baseline + R_beat - 2*R_omit)

from SiNAPSE.core import Database, Recording, Neuron
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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
def baseline_fr(recording, db, rec, duration=2, binsize=1/1000):
    select_columns = ['unit_id', 'session_id']
    conditions = {
        'manual_isi_0_7': ('<', 1),
        'session_id': ('=', recording),
    }
    units = db.load_neurons_from_database(select_columns, conditions)

    baselines = {}
    for unit, _ in units:
        baselines[unit] = {}
        N = Neuron(recording, unit, rec=rec, db=db)
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

    return baselines
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
        raise ValueError("Could not find Tempo (IOI) in file.")

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

def subset(
        PLS_TRAINING_STIM,
        PLS_TESTING_STIM,
        cutoff=25,
        REGION='Field L',
        plot=False,
        birds=None,
        save_dir=save_dir,
        DB_PATH=r'C:\Users\tmerri03\Desktop\Temp Neural Files\awake_recordings.db',
        depth_lower=None,
        depth_upper=None,
        pcc_lower=None,
        pcc_upper=None,
        spikewidth=None,
        overwrite_save_path=None,
):
    import numpy as np

    print(
        f'Trying to find a subset of neurons that have these stim: '
        f'{PLS_TRAINING_STIM} and {PLS_TESTING_STIM}.'
    )
    print(f'--> using: {DB_PATH}')

    if overwrite_save_path is None:
        save_path = os.path.join(
            save_dir,
            f'{REGION}_top_neurons_{cutoff}.csv'
        )
    else:
        save_path = os.path.join(save_dir, overwrite_save_path)

    if not os.path.exists(save_path):

        db = Database(DB_PATH, STIM_LIB_PATH)
        recordings = sorted(db.recordings)

        rows = []

        for r, recording in enumerate(recordings):

            region = REGION.replace(' ', '').replace('_', '').lower()
            recording_name = recording.replace(' ', '').replace('_', '').lower()

            if region not in recording_name:
                continue

            bird = recording.split(' ')[0]
            rec_fp = os.path.join(RECORDINGS_PATH, bird, recording)
            rec = Recording(rec_fp, samplerate=30000, db=db)

            select_columns = [
                'session_id',
                'unit_id',
                'depth',
                'pcc',
                'spike_width_pp',
            ]

            conditions = {
                'manual_isi_0_7': ('<', 1),
                'session_id': ('=', recording),
            }

            units = db.load_neurons_from_database(
                select_columns,
                conditions
            )

            # -----------------------------------------------------------------
            # Baseline is only needed for SI calculation.
            # Therefore, don't calculate it when cutoff == 100.
            # -----------------------------------------------------------------
            _valid_recording = True

            if cutoff != 100:
                try:
                    baselines = baseline_fr(
                        recording=recording,
                        db=db,
                        rec=rec,
                        duration=2
                    )
                except Exception as e:
                    _valid_recording = False
                    print(
                        f'Could not calculate baseline firing rates '
                        f'for recording: {recording}'
                    )
                    print(f'    Error: {e}')

            for session, unit, depth, pcc, p2p in units:

                # -------------------------------------------------------------
                # Apply depth/PCC filters
                # -------------------------------------------------------------
                if depth_lower is not None:
                    if depth < depth_lower:
                        continue

                if depth_upper is not None:
                    if depth > depth_upper:
                        continue

                if pcc_upper is not None:
                    if pcc > pcc_upper:
                        continue

                if pcc_lower is not None:
                    if pcc < pcc_lower:
                        continue

                if spikewidth is not None:
                    if spikewidth == 'narrow':
                        if  p2p >= 0.5:
                            continue
                    elif spikewidth == 'broad':
                        if p2p <= 0.5:
                            continue

                # -------------------------------------------------------------
                # Bird filter
                # -------------------------------------------------------------
                if birds is not None:
                    if bird not in birds:
                        continue

                # -------------------------------------------------------------
                # Create neuron and check stimuli
                # -------------------------------------------------------------
                N = Neuron(
                    session,
                    unit,
                    rec=rec,
                    db=db
                )

                stims = N.load
                if not len(set(PLS_TRAINING_STIM) & set(stims)) == len(PLS_TRAINING_STIM):
                    print(f'Incorrect stimulus for recording: {recording}')
                    continue

                if not len(set(PLS_TESTING_STIM) & set(stims)) == len(PLS_TESTING_STIM):
                    print(f'Incorrect stimulus for recording: {recording}')
                    continue

                # =============================================================
                # CUTOFF == 100
                #
                # Use the neuron without calculating SI.
                # =============================================================
                if cutoff == 100:

                    row = {
                        'recording': recording,
                        'unit': unit,
                        'depth': depth,
                        'pcc': pcc,
                    }

                    rows.append(row)

                    continue

                # =============================================================
                # SI CALCULATION
                # =============================================================

                if not _valid_recording:
                    continue

                dat_fp = (
                    r'C:\Users\tmerri03\Desktop\RhythmStimuli'
                    r'\Tempi Range Stimuli 8-25-26'
                    r'\Stim Files\File Data'
                )

                dat_fp = os.path.join(
                    dat_fp,
                    f'{PLS_TRAINING_STIM[0]}_onsets.txt'
                )

                onset_times, tempo, omission_location, omission_time, \
                    beat_duration, sr = get_onsets_tempo_omission(dat_fp)

                critical_duration = beat_duration + tempo / 2

                _, spikes, duration = N.raster(
                    PLS_TRAINING_STIM[0],
                    baseline=False,
                    ax=None,
                    plot=False
                )

                spikes = np.array(spikes)

                omission_window = (
                    (spikes > onset_times[omission_location] + 0.18) &
                    (spikes <
                     onset_times[omission_location] +
                     0.18 +
                     critical_duration)
                )

                omission_spikes = len(spikes[omission_window])

                _, sep_spikes, duration = N.raster(
                    PLS_TRAINING_STIM[0],
                    baseline=False,
                    separate_trials=True,
                    ax=None,
                    plot=False
                )

                n_omission_trials = len(sep_spikes)

                omission_spikes = (
                    omission_spikes / n_omission_trials
                )

                beat_spikes = []

                for onset in onset_times[:6]:

                    mask = (
                        (spikes > onset) &
                        (spikes < onset + critical_duration)
                    )

                    beat_spikes.append(
                        len(spikes[mask]) / n_omission_trials
                    )

                avg_beat_spikes = np.mean(beat_spikes)

                if n_omission_trials > 0:
                    baseline_spikes = (
                        baselines[unit][('mean')] *
                        critical_duration
                    )
                else:
                    baseline_spikes = None

                if baseline_spikes is not None:

                    SI = (
                        2 *
                        (omission_spikes - baseline_spikes) /
                        (avg_beat_spikes - baseline_spikes)
                    ) - 1

                    row = {
                        'recording': recording,
                        'unit': unit,
                        'depth': depth,
                        'pcc': pcc,
                        'SI': SI,
                    }

                    rows.append(row)

        # =====================================================================
        # CUTOFF == 100
        # =====================================================================
        if cutoff == 100:

            top_neurons = pd.DataFrame(rows)

            print(
                f' -- Using all {len(top_neurons)} valid neurons '
                f'(no SI calculation) --'
            )

        # =====================================================================
        # CUTOFF < 100
        # =====================================================================
        else:

            SIs = pd.DataFrame(rows)

            print(len(SIs))

            # Remove invalid SI values
            valid_SIs = SIs[
                SIs['SI'].notna() &
                np.isfinite(SIs['SI']) &
                (SIs['SI'] > -1)
            ].copy()

            if plot:

                fig, ax = plt.subplots(figsize=(8, 5))

                ax.hist(
                    valid_SIs['SI'],
                    bins=25,
                    density=False,
                    alpha=0.7,
                    edgecolor='black'
                )

                ax.axvline(
                    0,
                    linestyle='--',
                    linewidth=2,
                    label='SI = 0'
                )

                ax.set_xlabel('Similarity Index (SI)')
                ax.set_ylabel('Number of neurons')
                ax.set_title('Distribution of Similarity Index')
                ax.legend()

                plt.tight_layout()
                plt.show()

            # Sort by SI
            valid_SIs = valid_SIs.sort_values(
                'SI',
                ascending=False
            )

            # Select top X%
            n_top = int(
                np.ceil(
                    len(valid_SIs) * cutoff / 100
                )
            )

            top_neurons = valid_SIs.head(n_top).copy()

            print(
                f' -- Saving top {cutoff}% of neurons --'
            )

            print(
                f' -- Selected {len(top_neurons)} / '
                f'{len(valid_SIs)} neurons --'
            )

        # Save
        top_neurons.to_csv(
            save_path,
            index=False
        )

    # Load
    top_neurons = pd.read_csv(save_path)

    return top_neurons