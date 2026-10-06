OUT_DIR = r'C:\Users\tmerri03\Desktop\PLS Results\simoultaneous'
LOCATION = 'HVC'

PLS_TESTING_STIM = [
                    "exp1_ZF A_20db_120ms_8b_1xomit_8b_silence",
                    "exp1_ZF A_20db_144ms_8b_1xomit_8b_silence",
                    "exp1_ZF A_20db_180ms_8b_1xomit_8b_silence",
                    "exp1_ZF A_20db_220ms_8b_1xomit_8b_silence",
                    "exp1_ZF A_20db_300ms_8b_1xomit_8b_silence",
                    "exp1_2khz_20db_180ms_8b_1xomit_8b_silence"
                    ]
PLS_TRAINING_STIM = [
                    "exp1_ZF A_20db_120ms_8b_1xomit_8b_silence",
                    "exp1_ZF A_20db_144ms_8b_1xomit_8b_silence",
                    "exp1_ZF A_20db_180ms_8b_1xomit_8b_silence",
                    "exp1_ZF A_20db_220ms_8b_1xomit_8b_silence",
                    "exp1_ZF A_20db_300ms_8b_1xomit_8b_silence",
                    "exp1_2khz_20db_180ms_8b_1xomit_8b_silence"
                    ]
DB_PATH = r"C:\\Users\\tmerri03\\Desktop\\Temp Neural Files\\awake_recordings.db"
STIM_LIB_PATH = r"R:\\Data\\tyler\\Recordings\\Stim\\Stimuli Library"
RECORDINGS_PATH = r"R:\\Data\\RhythmPerception\\Neural Recordings\\Recordings"

import similarity_index
from PLS import PLS
import matplotlib.pyplot as plt
import os
import numpy as np
from SiNAPSE.core import Database
import plotting
import pandas as pd

all_neurons = similarity_index.subset(PLS_TESTING_STIM,
                                      PLS_TRAINING_STIM,
                                      birds=None,
                                      plot=False,
                                      save_dir = OUT_DIR,
                                      DB_PATH = DB_PATH,
                                      cutoff=100,
                                      REGION = LOCATION,
                                      )

db = Database(DB_PATH, STIM_LIB_PATH)
for recording in db.recordings:
    if recording not in pd.unique(all_neurons['recording']):
        continue

    if os.path.exists(os.path.join(OUT_DIR, recording, 'beatwise_3d.svg')):
        print(f'Recording {recording} is already processed!')
        continue

    print(f'Starting work on recording: {recording}')
    top_neurons = all_neurons.dropna(subset=['recording', 'unit'])
    top_neurons = top_neurons[top_neurons['recording'] == recording].reset_index(drop=True)

    pls = PLS(DB_PATH,
              RECORDINGS_PATH,
              STIM_LIB_PATH,
              os.path.join(OUT_DIR, recording),
              LOCATION,
              PLS_TRAINING_STIM,
              PLS_TESTING_STIM,
              )
    pls.prepare(top_neurons)
    pls.fit()

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
        '144 Regular': '#99d8c9',  # teal
        # '144 Irregular': '#238b74',
        '180 Regular': '#a1d99b',  # green
        # '180 Irregular': '#31a354',
        '220 Regular': '#fdae6b',  # orange
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
                tempo = None
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
                onset_times, tempo, omission_location, omission_time, beat_duration, sr = get_onsets_tempo_omission(
                    os.path.join(base_fp, f'{stim[key]}_onsets.txt'))
                beats = np.array(onset_times) - np.array(onset_times)[0]

            for i in range(0, 2):
                beats = np.append(beats, beats[-1] + 0.100)
            beat_times[key] = beats
    except Exception as e:
        print(e)
        beat_times = {
            'Omission 180 ms A': np.arange(0, 20, 1) * 0.180,
            'Omission 300 ms A': np.arange(0, 20, 1) * 0.300,
        }

    beat_traj, beat_time = plotting.plot_beatwise_share(
        trajectories=trajectories,
        response_times=T,
        Y=None,
        beat_times=beat_times,
        colors=colors,
        n_cols=4,
        N=len(top_neurons),
        plot=True,
        save_path = os.path.join(OUT_DIR, recording)
    )

