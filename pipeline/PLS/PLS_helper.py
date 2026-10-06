pattern = r'(?P<tempo>\d+(?:\.\d+)?)ms_(?P<pre_beats>\d+)b_(?P<n_omissions>\d+)xomit_\d+b'
valid_ = ['2khz_20db_180ms_8b_1xomit_8b_silence',
          '2khz_20db_300ms_8b_1xomit_8b_silence',
          'ZF A_20db_180ms_8b_1xomit_8b_silence',
          'ZF A_20db_300ms_8b_1xomit_8b_silence',
          ]

import re, os
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def decompose_stimulus(stimulus, stim_lib):
    if stimulus in valid_:
        match = re.search(pattern, stimulus)

        if match:
            tempo_ms = float(match.group("tempo"))
            omission_idx = int(match.group("pre_beats")) + 1

            print(stimulus)
            print(tempo_ms)  # 180.0
            print(omission_idx)  # 1

            wav_path = os.path.join(os.path.join(stim_lib), f'{stimulus}.wav')
            beat_info = find_beat_times(wav_path, plot=False)

            beat_times = sorted(beat_info['expected_beat_times'])
            tempo_s = tempo_ms / 1000
            is_irregular = False

            stimulus_data = {
                "beat_times": beat_times,
                "omission_idx": omission_idx,
                "tempo": tempo_s,
                "is_irregular": is_irregular,
            }
            return stimulus_data
        return None

def find_beat_times(
    wav_path,
    ax=None,
    plot=False,
    derivative_threshold=0.05,
    edge_group_ms=5,
    min_sound_duration_ms=10,
):
    import os
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    from scipy.signal import find_peaks

    # =========================================================
    # Parse filename
    # =========================================================
    filename = os.path.basename(wav_path)

    pattern = (
        r'(?P<tempo>\d+(?:\.\d+)?)ms_'
        r'(?P<pre_beats>\d+)b_'
        r'(?P<n_omissions>\d+)xomit_'
        r'(?P<post_beats>\d+)b'
    )

    match = re.search(pattern, filename)

    if match is None:
        raise ValueError(
            f"Could not parse filename: {filename}\n"
            "Expected format such as:\n"
            "180ms_8b_1xomit_8b.wav"
        )

    tempo_ms = float(match.group("tempo"))
    pre_beats = int(match.group("pre_beats"))
    n_omissions = int(match.group("n_omissions"))
    post_beats = int(match.group("post_beats"))

    tempo_s = tempo_ms / 1000

    # 0-based indices of omitted beats
    omission_indices = np.arange(
        pre_beats,
        pre_beats + n_omissions
    )

    # =========================================================
    # Load WAV
    # =========================================================
    fs, stim_data = wavfile.read(wav_path)

    # Stereo -> mono
    if stim_data.ndim > 1:
        stim_data = stim_data.mean(axis=1)

    stim_data = stim_data.astype(float)

    # Normalize to roughly [-1, 1]
    max_abs = np.max(np.abs(stim_data))

    if max_abs > 0:
        stim_data /= max_abs

    stim_time = np.arange(len(stim_data)) / fs

    # =========================================================
    # Plot setup
    # =========================================================
    if plot and ax is None:
        fig, ax = plt.subplots(
            1, 1,
            figsize=(15, 5)
        )

    if plot:
        ax.plot(
            stim_time,
            stim_data,
            linewidth=0.7,
            label="Stimulus"
        )

    # =========================================================
    # DETECT BEATS
    #
    # Same basic algorithm as your original function:
    #
    # waveform
    #     ↓
    # derivative
    #     ↓
    # rising/falling edges
    #     ↓
    # group nearby edges
    #     ↓
    # identify sound onsets
    # =========================================================

    audio_deriv = np.gradient(stim_data)

    peaks_pos, _ = find_peaks(
        audio_deriv,
        height=derivative_threshold
    )

    peaks_neg, _ = find_peaks(
        -audio_deriv,
        height=derivative_threshold
    )

    all_edges = np.sort(
        np.concatenate(
            (peaks_pos, peaks_neg)
        )
    )

    if len(all_edges) == 0:
        raise RuntimeError(
            f"No waveform edges detected in {filename}"
        )

    edge_times = stim_time[all_edges]

    # =========================================================
    # GROUP EDGES THAT BELONG TO SAME SOUND
    # =========================================================

    edge_group_s = edge_group_ms / 1000

    grouped_edges = []

    current_group = [edge_times[0]]

    for time in edge_times[1:]:

        if time - current_group[-1] <= edge_group_s:
            current_group.append(time)

        else:
            grouped_edges.append(current_group)
            current_group = [time]

    grouped_edges.append(current_group)

    # =========================================================
    # Extract rising/falling edges
    # =========================================================

    rising_edges_times = []
    falling_edges_times = []

    min_sound_duration_s = min_sound_duration_ms / 1000

    for group in grouped_edges:

        start = group[0]
        stop = group[-1]

        # Ignore tiny noise transients
        if stop - start > min_sound_duration_s:

            rising_edges_times.append(start)
            falling_edges_times.append(stop)

            if plot:

                ax.axvline(
                    x=start,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    label="Detected onset"
                    if len(rising_edges_times) == 1
                    else None
                )

                ax.axvline(
                    x=stop,
                    linestyle="--",
                    linewidth=1,
                    alpha=0.5,
                    label="Detected offset"
                    if len(falling_edges_times) == 1
                    else None
                )

    rising_edges_times = np.asarray(
        rising_edges_times
    )

    falling_edges_times = np.asarray(
        falling_edges_times
    )

    if len(rising_edges_times) == 0:
        raise RuntimeError(
            f"No stimulus onsets detected in {filename}"
        )

    # =========================================================
    # CONSTRUCT EXPECTED BEAT GRID
    # =========================================================

    total_beats = (
        pre_beats
        + n_omissions
        + post_beats
    )

    # First detected sound is beat 1
    first_beat_time = rising_edges_times[0]

    expected_beat_times = (
        first_beat_time
        + np.arange(total_beats) * tempo_s
    )

    # =========================================================
    # Plot expected beat times
    # =========================================================

    if plot:

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        for beat_idx, beat_time in enumerate(
            expected_beat_times
        ):

            if beat_idx in omission_indices:

                # ---------------------------------------------
                # OMITTED BEAT
                # ---------------------------------------------

                ax.axvline(
                    beat_time,
                    linestyle="--",
                    linewidth=2,
                    alpha=0.9
                )

                ax.text(
                    beat_time,
                    y_max - 0.05 * y_range,
                    f"Beat {beat_idx + 1}\nOMIT",
                    rotation=90,
                    ha="right",
                    va="top",
                    fontsize=9
                )

            else:

                # ---------------------------------------------
                # NORMAL BEAT
                # ---------------------------------------------

                ax.axvline(
                    beat_time,
                    linestyle=":",
                    linewidth=1,
                    alpha=0.7
                )

                ax.text(
                    beat_time,
                    y_max - 0.05 * y_range,
                    str(beat_idx + 1),
                    rotation=90,
                    ha="right",
                    va="top",
                    fontsize=9
                )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

        ax.set_title(
            f"{filename}\n"
            f"Tempo = {tempo_ms:g} ms | "
            f"Omission = {omission_indices + 1}"
        )

        ax.legend()

        plt.tight_layout()
        plt.show()

    # =========================================================
    # RETURN
    # =========================================================

    return {
        "rising_edges": rising_edges_times,
        "falling_edges": falling_edges_times,

        "expected_beat_times": expected_beat_times,

        "omission_indices": omission_indices,
        "omission_indices_1based": omission_indices + 1,

        "tempo_ms": tempo_ms,

        "pre_beats": pre_beats,
        "n_omissions": n_omissions,
        "post_beats": post_beats,

        "ax": ax,
    }

# 2khz irregular control
# ZF A irregular control

# 2khz_20db_180ms_8b_1xomit_8b_silence
# 2khz_20db_180ms_8b_2xomit_8b_silence

# 2khz_20db_300ms_8b_1xomit_8b_silence
# 2khz_20db_300ms_8b_2xomit_8b_silence


# ZF A_20db_180ms_8b_1xomit_8b_silence
# ZF A_20db_180ms_8b_2xomit_8b_silence

# ZF A_20db_300ms_8b_1xomit_8b_silence
# ZF A_20db_300ms_8b_2xomit_8b_silence

# ZF A 180ms 4b_2xomit_12b_silence
# ZF A 180ms 12b_2xomit_4b_silence

import os
import pickle


def save_processed_recording(
    recording_name,
    trials,
    unit_ids,
    X_train,
    Y_train,
    basis,
    save_dir="pls_cache"
):
    """
    Save all processed data needed to fit PLS later.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Give each recording a simple filename.
    filepath = os.path.join(
        save_dir,
        f"{recording_name}.pkl"
    )

    data = {
        "recording_name": recording_name,
        "trials": trials,
        "unit_ids": unit_ids,
        "X_train": X_train,
        "Y_train": Y_train,
        "basis": basis,
    }

    with open(filepath, "wb") as f:
        pickle.dump(
            data,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print(f"Saved processed recording:")
    print(f"  {filepath}")

    return filepath


def load_processed_recording(filepath):
    """
    Load a previously processed recording.
    """

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded processed recording:")
    print(f"  {data['recording_name']}")
    print(f"  Neurons: {len(data['unit_ids'])}")
    print(f"  X shape: {data['X_train'].shape}")
    print(f"  Y shape: {data['Y_train'].shape}")

    return data
