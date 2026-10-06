import os
import numpy as np
import json
import matplotlib.pyplot as plt
import json

EXPERIMENT = 'tempo_range_analysis_FieldL'
CUTOFF = 100
TRAINING = 'trained'
prepend = None
recording = None
# --- 1. Load settings ---
def load_pls_settings(settings_path, analysis_name):
    """
    Load PLS settings and return the selected analysis configuration.

    Parameters
    ----------
    settings_path : str
        Path to the JSON settings file.

    analysis_name : str
        Name of the analysis to run.

    Returns
    -------
    dict
        Configuration dictionary containing paths, stimuli,
        birds, and analysis-specific settings.
    """
    with open(settings_path, "r") as f:
        settings = json.load(f)

    if analysis_name not in settings["analyses"]:
        available = list(settings["analyses"].keys())
        raise ValueError(
            f"Unknown analysis '{analysis_name}'. "
            f"Available analyses: {available}"
        )

    analysis = settings["analyses"][analysis_name]

    return {
        "db_path": settings["paths"]["db_path"],
        "save_dir": settings["paths"]["save_dir"],
        "recordings_path": settings["paths"]["recordings_path"],
        "stim_lib_path": settings["paths"]["stim_lib_path"],

        "training_stim": analysis["training_stim"],
        "testing_stim": analysis["testing_stim"],
        "region": analysis["region"],

        "trained_birds": settings["birds"]["trained"],
        "yoked_birds": settings["birds"]["yoked"],
        "naive_birds": settings["birds"]["naive"]
    }
settings = load_pls_settings('settings.json', EXPERIMENT)

DB_PATH = settings["db_path"]

base_save_dir = settings["save_dir"]

save_dir = os.path.join(base_save_dir, EXPERIMENT)
os.makedirs(save_dir, exist_ok=True)

PLS_TRAINING_STIM = settings["training_stim"]
PLS_TESTING_STIM = settings["testing_stim"]

trained_birds = settings["trained_birds"]
yoked_birds = settings["yoked_birds"]
naive_birds = settings["naive_birds"]

REGION = settings["region"]

RECORDINGS_PATH = settings["recordings_path"]
STIM_LIB_PATH = settings["stim_lib_path"]

# --- 2. Filter neurons by most beat-like firing ---
# To use all neurons, set cutoff=100
import similarity_index
training=TRAINING
if training is None:
    save_dir = os.path.join(save_dir, 'all')
    training_status = None
elif training == 'trained':
    save_dir = os.path.join(save_dir, 'trained')
    training_status = trained_birds
elif training == 'yoked':
    save_dir = os.path.join(save_dir, 'yoked')
    training_status = yoked_birds
elif training == 'naive':
    save_dir = os.path.join(save_dir, 'naive')
    training_status = naive_birds
elif training == 'manual':
    save_dir = os.path.join(save_dir, 'manual')

if prepend is not None:
    save_dir = os.path.join(save_dir,prepend)
training_status = None

save_dir = r'C:\Users\tmerri03\Desktop\PLS Results\template'
os.makedirs(save_dir, exist_ok=True)
top_neurons = similarity_index.subset(PLS_TRAINING_STIM, PLS_TESTING_STIM,
                                      birds=training_status, cutoff=CUTOFF,
                                      REGION=REGION,plot=True, save_dir=save_dir,
                                      DB_PATH=DB_PATH,
                                      depth_lower = None,
                                      depth_upper = None,
                                      pcc_lower = None,
                                      pcc_upper = None,
                                      spikewidth=None,
                                      overwrite_save_path=r'top_neurons_10.csv')
top_neurons = top_neurons.dropna(subset=['recording', 'unit'])
if recording is not None:
    top_neurons = top_neurons[top_neurons['recording'] == recording]
print(f'Found {len(top_neurons)} neurons in the top {CUTOFF}% of the similarity score distribution.')

# --- 3. Fit Partial Least Squares Model ---
from PLS import PLS
pls = PLS(DB_PATH,
        RECORDINGS_PATH,
        STIM_LIB_PATH,
        save_dir,
        REGION,
        PLS_TRAINING_STIM,
        PLS_TESTING_STIM,)
pls.prepare(top_neurons)
pls.fit()

# --- 4. Plot Results  ---
stim = {
    '120 Regular': 'exp1_ZF A_20db_120ms_8b_1xomit_8b_silence',
    # '120 Irregular': 'exp1_exp1_ZF A_20db_120ms_8b_1xomit_8b_silence_ir',
    '144 Regular': 'exp1_ZF A_20db_144ms_8b_1xomit_8b_silence',
    # '144 Irregular': 'exp1_exp1_ZF A_20db_144ms_8b_1xomit_8b_silence_ir',
    '180 Regular': 'exp1_ZF A_20db_180ms_8b_1xomit_8b_silence',
    # '180 Irregular': 'exp1_exp1_ZF A_20db_180ms_8b_1xomit_8b_silence_ir',
    '220 Regular': 'exp1_ZF A_20db_220ms_8b_1xomit_8b_silence',
    # '220 Irregular': 'exp1_exp1_ZF A_20db_220ms_8b_1xomit_8b_silence_ir',
    '300 Regular': 'exp1_ZF A_20db_300ms_8b_1xomit_8b_silence',
    # '300 Irregular': 'exp1_exp1_ZF A_20db_300ms_8b_1xomit_8b_silence_ir',
    # '180 Late': 'exp1_ZF A_20db_180ms_12b_1xomit_4b_silence',
    # 'Single Beat': 'exp1_ZF A_20db_300_onebeatcontrol_silence'
    }

Z_omit_A_120_reg, Y_omit_A_120_reg, T_omit_A_120_reg = pls.trajectory(top_neurons, stim['120 Regular'])
# Z_omit_A_120_ir, Y_omit_A_120_ir, T_omit_A_120_ir = pls.trajectory(top_neurons, stim['120 Irregular'])
Z_omit_A_144_reg, Y_omit_A_144_reg, T_omit_A_144_reg = pls.trajectory(top_neurons, stim['144 Regular'])
# Z_omit_A_144_ir, Y_omit_A_144_ir, T_omit_A_144_ir = pls.trajectory(top_neurons, stim['144 Irregular'])
Z_omit_A_180_reg, Y_omit_A_180_reg, T_omit_A_180_reg = pls.trajectory(top_neurons, stim['180 Regular'])
# Z_omit_A_180_ir, Y_omit_A_180_ir, T_omit_A_180_ir = pls.trajectory(top_neurons, stim['180 Irregular'])
Z_omit_A_220_reg, Y_omit_A_220_reg, T_omit_A_220_reg = pls.trajectory(top_neurons, stim['220 Regular'])
# Z_omit_A_220_ir, Y_omit_A_220_ir, T_omit_A_220_ir = pls.trajectory(top_neurons, stim['220 Irregular'])
Z_omit_A_300_reg, Y_omit_A_300_reg, T_omit_A_300_reg = pls.trajectory(top_neurons, stim['300 Regular'])
# Z_omit_A_300_ir, Y_omit_A_300_ir, T_omit_A_300_ir = pls.trajectory(top_neurons, stim['300 Irregular'])
# Z_omit_A_180_late, Y_omit_A_180_late, T_omit_A_180_late = pls.trajectory(top_neurons, stim['180 Late'])
# Z_single_A, Y_single_A, T_single_A = pls.trajectory(top_neurons, stim['Single Beat'])
trajectories = {
    '120 Regular': Z_omit_A_120_reg,
    # '120 Irregular': Z_omit_A_120_ir,
    '144 Regular': Z_omit_A_144_reg,
    # '144 Irregular': Z_omit_A_144_ir,
    '180 Regular': Z_omit_A_180_reg,
    # '180 Irregular': Z_omit_A_180_ir,
    '220 Regular': Z_omit_A_220_reg,
    # '220 Irregular': Z_omit_A_220_ir,
    '300 Regular': Z_omit_A_300_reg,
    # '300 Irregular': Z_omit_A_300_ir,
    # '180 Late': Z_omit_A_180_late,
    # 'Single Beat': Z_single_A,
    }
T = {
    '120 Regular': T_omit_A_120_reg,
    # '120 Irregular': T_omit_A_120_ir,
    '144 Regular': T_omit_A_144_reg,
    # '144 Irregular': T_omit_A_144_ir,
    '180 Regular': T_omit_A_180_reg,
    # '180 Irregular': T_omit_A_180_ir,
    '220 Regular': T_omit_A_220_reg,
    # '220 Irregular': T_omit_A_220_ir,
    '300 Regular': T_omit_A_300_reg,
    # '300 Irregular': T_omit_A_300_ir,
    # '180 Late': T_omit_A_180_late,
    # 'Single Beat': T_single_A,
    }
colors = {
    '120 Regular': '#9ecae1',
    # '120 Irregular': '#2171b5',
    '144 Regular': '#99d8c9', #teal
    # '144 Irregular': '#238b74',
    '180 Regular': '#a1d99b', #green
    # '180 Irregular': '#31a354',
    '220 Regular': '#fdae6b', #orange
    # '220 Irregular': '#e6550d',
    '300 Regular': '#fcae91',
    # '300 Irregular': '#cb181d',
    # '180 Late': '#000000',
    # 'Single Beat': '#A47DAB',
}


try:
    beat_times = {}
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
            r"(?:Target\s+)?Tempo \(IOI\):\s*([\d.]+)\s*s",
            text,
            re.IGNORECASE
        )

        if tempo_match is None:
            tempo=None
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
    base_fp = r'C:\Users\tmerri03\Desktop\RhythmStimuli\Tempi Range Stimuli 8-25-26\Stim Files\File Data'
    for key in stim.keys():
        if key == 'Single Beat':
            onset_times = [0.518594]
            tempo = 0.180
        else:
            onset_times, tempo, omission_location, omission_time, beat_duration, sr = get_onsets_tempo_omission(os.path.join(base_fp, f'{stim[key]}_onsets.txt'))
            beats = np.array(onset_times) - np.array(onset_times)[0]

        for i in range(0,2):
            beats = np.append(beats, beats[-1]+0.100)
        beat_times[key] = beats


except Exception as e:
    print(e)
    beat_times = {
            'Omission 180 ms A': np.arange(0,20,1)*0.180,
            'Omission 300 ms A': np.arange(0,20,1)*0.300,
        }

import plotting
beat_traj, beat_time = plotting.plot_beatwise_share(
    trajectories=trajectories,
    response_times=T,
    Y=None,
    beat_times=beat_times,
    colors=colors,
    n_cols=4,
    N=len(top_neurons),
    plot=True
)

plotting.plot_color_legend(colors)
# plotting.plot_beat_by_beat_correlation_heatmap(
#     trajectories=trajectories,
#     response_times=T,
#     beat_times=beat_times,
# )

# plotting.plot_trajectory_correlation_heatmap(
#     trajectories=trajectories,
#     response_times=T,
#     beat_times=beat_times,
# )
# plt.show()

def average_trajectory_beats(output_traj, stimulus, beat_indices):
    """
    Average the multidimensional trajectory across selected beats.

    Parameters
    ----------
    output_traj : dict
        Nested dictionary:
            output_traj[beat_idx][stimulus]
            = (n_phase_points, n_dimensions)

    stimulus : str
        Stimulus condition to use.

    beat_indices : list or array
        Beat indices to average.

    Returns
    -------
    average_traj : np.ndarray
        Shape:
            (n_phase_points, n_dimensions)
    """

    trajectories = []

    for beat_idx in beat_indices:

        if beat_idx not in output_traj:
            raise ValueError(
                f"Beat {beat_idx} is not present in output_traj."
            )

        if stimulus not in output_traj[beat_idx]:
            raise ValueError(
                f"Stimulus '{stimulus}' is not present for beat {beat_idx}."
            )

        trajectories.append(
            np.asarray(output_traj[beat_idx][stimulus])
        )

    trajectories = np.stack(trajectories, axis=0)

    # Average across beats, preserving phase × dimensions
    average_traj = np.mean(trajectories, axis=0)

    return average_traj
avg_traj = average_trajectory_beats(
    beat_traj,
    stimulus='180 Regular',
    beat_indices=[1,2,3,4,5,6,7],
)

def template_correlations(output_traj, avg_traj, stimulus):
    corrs = []
    for idx, key in enumerate(output_traj):
        if stimulus in output_traj[key].keys():
            traj = output_traj[key][stimulus]
            r = np.corrcoef(traj.ravel(), avg_traj.ravel())[0,1]
            print(f'Beat {idx+1}: {r}')
            corrs.append(r)
    return corrs

# single = template_correlations(beat_traj, avg_traj, stimulus='Single Beat')
# reg = template_correlations(beat_traj, avg_traj, stimulus='180 Regular')
# late = template_correlations(beat_traj, avg_traj, stimulus='180 Late')
#
# plt.plot(single, label='single')
# plt.plot(reg, label='8')
# plt.plot(late, label='12')
# plt.xticks(np.arange(0, 18, 1))
# plt.legend()
# plt.show()

# traj_120 = average_trajectory_beats(beat_traj,
#                                     stimulus='120 Regular',
#                                     beat_indices=[0])
# traj_144 = average_trajectory_beats(beat_traj,
#                                     stimulus='144 Regular',
#                                     beat_indices=[0])
# traj_180 = average_trajectory_beats(beat_traj,
#                                     stimulus='180 Regular',
#                                     beat_indices=[0])
# traj_220 = average_trajectory_beats(beat_traj,
#                                     stimulus='220 Regular',
#                                     beat_indices=[0])
# traj_300 = average_trajectory_beats(beat_traj,
#                                     stimulus='300 Regular',
#                                     beat_indices=[0])
# one20 = template_correlations(beat_traj, traj_120, stimulus='120 Regular')
# one44 = template_correlations(beat_traj, traj_144, stimulus='144 Regular')
# one80 = template_correlations(beat_traj, traj_180, stimulus='180 Regular')
# two20 = template_correlations(beat_traj, traj_220, stimulus='220 Regular')
# three30 = template_correlations(beat_traj, traj_300, stimulus='300 Regular')
# plt.plot(one20, label='120')
# plt.plot(one44, label='144')
# plt.plot(one80, label='180')
# plt.plot(two20, label='220')
# plt.plot(three30, label='300')
# plt.legend()
# plt.title(f'{TRAINING} - compared to first beat')
# plt.show()

from scipy.signal import savgol_filter
stimulus = '180 Regular'
omission_idx = 8

omission_traj = beat_traj[omission_idx][stimulus]

# Smooth each PLS dimension independently
window = 11   # increase for more smoothing; must be odd
polyorder = 2

avg_smooth = savgol_filter(
    avg_traj,
    window_length=window,
    polyorder=polyorder,
    axis=0
)

omission_smooth = savgol_filter(
    omission_traj,
    window_length=window,
    polyorder=polyorder,
    axis=0
)

plt.figure(figsize=(7, 7))

# Average beat
plt.plot(
    avg_smooth[:, 0],
    avg_smooth[:, 1],
    linewidth=2,
    label='Average beat'
)

# Omission beat
plt.plot(
    omission_smooth[:, 0],
    omission_smooth[:, 1],
    linewidth=2,
    label='Omission beat'
)

# Starting points
plt.scatter(
    avg_smooth[0, 0],
    avg_smooth[0, 1],
    s=50
)

plt.scatter(
    omission_smooth[0, 0],
    omission_smooth[0, 1],
    s=50
)

plt.xlabel('PLS 1')
plt.ylabel('PLS 2')
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.savefig('GRFP_trajectories.svg', format='svg')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D

from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# =============================================================================
# SETTINGS
# =============================================================================

# Beats to average for the "average beat" trajectory
beat_indices = [1, 2, 3, 4, 5, 6, 7]

window = 11
polyorder = 2

# =============================================================================
# PLOT
# =============================================================================

fig, ax = plt.subplots(figsize=(7, 7))

for stimulus in trajectories.keys():

    # -------------------------------------------------------------------------
    # Average beat trajectory for this tempo
    # -------------------------------------------------------------------------

    avg_traj = average_trajectory_beats(
        beat_traj,
        stimulus=stimulus,
        beat_indices=beat_indices
    )

    # -------------------------------------------------------------------------
    # Omission trajectory
    # -------------------------------------------------------------------------

    # In your beat_traj structure, beat index 8 is the omission
    omission_idx = 8
    omission_traj = beat_traj[omission_idx][stimulus]

    # -------------------------------------------------------------------------
    # Smooth
    # -------------------------------------------------------------------------

    avg_smooth = savgol_filter(
        avg_traj,
        window_length=window,
        polyorder=polyorder,
        axis=0
    )

    omission_smooth = savgol_filter(
        omission_traj,
        window_length=window,
        polyorder=polyorder,
        axis=0
    )

    # -------------------------------------------------------------------------
    # Color
    # -------------------------------------------------------------------------

    color = colors[stimulus]

    # -------------------------------------------------------------------------
    # Plot average beat
    # -------------------------------------------------------------------------

    ax.plot(
        avg_smooth[:, 0],
        avg_smooth[:, 1],
        color=color,
        linewidth=2.5,
        label=stimulus.replace(' Regular', '')
    )

    # -------------------------------------------------------------------------
    # Plot omission
    # -------------------------------------------------------------------------

    ax.plot(
        omission_smooth[:, 0],
        omission_smooth[:, 1],
        color=color,
        linewidth=2.5,
        linestyle='--'
    )

    # -------------------------------------------------------------------------
    # Starting points
    # -------------------------------------------------------------------------

    ax.scatter(
        avg_smooth[0, 0],
        avg_smooth[0, 1],
        color=color,
        s=40
    )

    ax.scatter(
        omission_smooth[0, 0],
        omission_smooth[0, 1],
        color=color,
        s=40,
        marker='o'
    )


# =============================================================================
# LABELS / LEGEND
# =============================================================================

ax.set_xlabel('PLS 1')
ax.set_ylabel('PLS 2')

ax.legend(
    title='Tempo (ms)',
    frameon=False
)

ax.axis('equal')
plt.tight_layout()

plt.savefig(
    'GRFP_trajectories_all_tempi.svg',
    format='svg',
    bbox_inches='tight'
)

plt.show()

pls.contribution()
