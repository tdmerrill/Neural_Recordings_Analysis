import os

urethane = False

#urethane stim
if urethane:
    print('need to set train & test stim')

    DB_PATH = r'C:\Users\tmerri03\Desktop\Temp Neural Files\neurons.db'
    save_dir = r'C:\Users\tmerri03\Desktop\Temp Neural Files\Urethane PLS'
else:
    DB_PATH = r'C:\Users\tmerri03\Desktop\Temp Neural Files\awake_recordings.db'
    # PLS_TRAINING_STIM = ['ZF A_20db_180ms_8b_1xomit_8b_silence',
    #                      'ZF A_20db_300ms_8b_1xomit_8b_silence'
    #                      ]


    PLS_TRAINING_STIM = ['exp1_ZF A_20db_120ms_8b_1xomit_8b_silence',
                         'exp1_ZF A_20db_144ms_8b_1xomit_8b_silence',
                         'exp1_ZF A_20db_180ms_8b_1xomit_8b_silence',
                         'exp1_ZF A_20db_220ms_8b_1xomit_8b_silence',
                         'exp1_ZF A_20db_300ms_8b_1xomit_8b_silence',

                         'exp1_2khz_20db_180ms_8b_1xomit_8b_silence',
                         ]
    PLS_TESTING_STIM = ['exp1_ZF A_20db_120ms_8b_1xomit_8b_silence',
                        'exp1_ZF A_20db_144ms_8b_1xomit_8b_silence',
                        'exp1_ZF A_20db_180ms_8b_1xomit_8b_silence',
                        'exp1_ZF A_20db_220ms_8b_1xomit_8b_silence',
                        'exp1_ZF A_20db_300ms_8b_1xomit_8b_silence',
                        'exp1_exp1_ZF A_20db_120ms_8b_1xomit_8b_silence_ir',
                        'exp1_exp1_ZF A_20db_144ms_8b_1xomit_8b_silence_ir',
                        'exp1_exp1_ZF A_20db_180ms_8b_1xomit_8b_silence_ir',
                        'exp1_exp1_ZF A_20db_220ms_8b_1xomit_8b_silence_ir',
                        'exp1_exp1_ZF A_20db_300ms_8b_1xomit_8b_silence_ir',

                        'exp1_2khz_20db_180ms_8b_1xomit_8b_silence',
                        'exp1_exp1_2khz_20db_180ms_8b_1xomit_8b_silence_ir',
                        ]





training = None
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
    training_status = None
os.makedirs(save_dir, exist_ok=True)

# 1. Filter neurons by most beat-like firing
from pipeline.PLS.run import similarity_index

cutoff=100
top_neurons = similarity_index.subset(PLS_TRAINING_STIM, PLS_TESTING_STIM,
                                      birds=training_status, cutoff=cutoff,
                                      REGION=REGION, plot=True, save_dir=save_dir,
                                      DB_PATH=DB_PATH,
                                      depth_lower = None,
                                      depth_upper = None,
                                      pcc_lower = None,
                                      pcc_upper = None)
top_neurons = top_neurons.dropna(subset=['recording', 'unit'])
print(f'Found {len(top_neurons)} neurons in the top {cutoff}% of the similarity score distribution.')
print(top_neurons.head())
print('')

# 1.5. Conditionally subsample top_neurons to some N
N = None
if N is not None:
    if len(top_neurons) > N:
        top_neurons = top_neurons.sample(n=N, random_state=42)
    print(f"Randomly sampled {len(top_neurons)} neurons.")
    save_dir = os.path.join(save_dir, 'subsampled')
    os.makedirs(save_dir, exist_ok=True)

# 2. Fit PLS model using regular stimuli for the subset of responsive neurons
from pipeline.PLS.run.PLS import PLS
pls = PLS(DB_PATH,
        RECORDINGS_PATH,
        STIM_LIB_PATH,
        save_dir,
        REGION,
        PLS_TRAINING_STIM,
        PLS_TESTING_STIM,)
pls.prepare(top_neurons)
pls.fit()

stim = {
    '120 Regular': 'exp1_ZF A_20db_120ms_8b_1xomit_8b_silence',
    '120 Irregular': 'exp1_exp1_ZF A_20db_120ms_8b_1xomit_8b_silence_ir',
    '144 Regular': 'exp1_ZF A_20db_144ms_8b_1xomit_8b_silence',
    '144 Irregular': 'exp1_exp1_ZF A_20db_144ms_8b_1xomit_8b_silence_ir',
    '180 Regular': 'exp1_ZF A_20db_180ms_8b_1xomit_8b_silence',
    '180 Irregular': 'exp1_exp1_ZF A_20db_180ms_8b_1xomit_8b_silence_ir',
    '220 Regular': 'exp1_ZF A_20db_220ms_8b_1xomit_8b_silence',
    '220 Irregular': 'exp1_exp1_ZF A_20db_220ms_8b_1xomit_8b_silence_ir',
    '300 Regular': 'exp1_ZF A_20db_300ms_8b_1xomit_8b_silence',
    '300 Irregular': 'exp1_exp1_ZF A_20db_300ms_8b_1xomit_8b_silence_ir',
    }

Z_omit_A_120_reg, Y_omit_A_120_reg, T_omit_A_120_reg = pls.trajectory(top_neurons, stim['120 Regular'])
Z_omit_A_120_ir, Y_omit_A_120_ir, T_omit_A_120_ir = pls.trajectory(top_neurons, stim['120 Irregular'])
Z_omit_A_144_reg, Y_omit_A_144_reg, T_omit_A_144_reg = pls.trajectory(top_neurons, stim['144 Regular'])
Z_omit_A_144_ir, Y_omit_A_144_ir, T_omit_A_144_ir = pls.trajectory(top_neurons, stim['144 Irregular'])
Z_omit_A_180_reg, Y_omit_A_180_reg, T_omit_A_180_reg = pls.trajectory(top_neurons, stim['180 Regular'])
Z_omit_A_180_ir, Y_omit_A_180_ir, T_omit_A_180_ir = pls.trajectory(top_neurons, stim['180 Irregular'])
Z_omit_A_220_reg, Y_omit_A_220_reg, T_omit_A_220_reg = pls.trajectory(top_neurons, stim['220 Regular'])
Z_omit_A_220_ir, Y_omit_A_220_ir, T_omit_A_220_ir = pls.trajectory(top_neurons, stim['220 Irregular'])
Z_omit_A_300_reg, Y_omit_A_300_reg, T_omit_A_300_reg = pls.trajectory(top_neurons, stim['300 Regular'])
Z_omit_A_300_ir, Y_omit_A_300_ir, T_omit_A_300_ir = pls.trajectory(top_neurons, stim['300 Irregular'])

trajectories = {
    '120 Regular': Z_omit_A_120_reg,
    '120 Irregular': Z_omit_A_120_ir,
    '144 Regular': Z_omit_A_144_reg,
    '144 Irregular': Z_omit_A_144_ir,
    '180 Regular': Z_omit_A_180_reg,
    '180 Irregular': Z_omit_A_180_ir,
    '220 Regular': Z_omit_A_220_reg,
    '220 Irregular': Z_omit_A_220_ir,
    '300 Regular': Z_omit_A_300_reg,
    '300 Irregular': Z_omit_A_300_ir,
    }
T = {
    '120 Regular': T_omit_A_120_reg,
    '120 Irregular': T_omit_A_120_ir,
    '144 Regular': T_omit_A_144_reg,
    '144 Irregular': T_omit_A_144_ir,
    '180 Regular': T_omit_A_180_reg,
    '180 Irregular': T_omit_A_180_ir,
    '220 Regular': T_omit_A_220_reg,
    '220 Irregular': T_omit_A_220_ir,
    '300 Regular': T_omit_A_300_reg,
    '300 Irregular': T_omit_A_300_ir,
    }

beat_times = {}
import numpy as np






# import numpy as np
# #awake
# if not urethane:
#     Z_omit_180_A, Y_omit_180_A, T_omit_180_A = pls.trajectory(top_neurons, 'ZF A_20db_180ms_8b_1xomit_8b_silence', pre=5)
#     Z_omit_300_A, Y_omit_300_A, T_omit_300_A = pls.trajectory(top_neurons, 'ZF A_20db_300ms_8b_1xomit_8b_silence', pre=5)
#     # Z_omit_180_2khz, Y_omit_180_2khz, T_omit_180_2khz = pls.trajectory(top_neurons, '2khz_20db_180ms_8b_1xomit_8b_silence')
#     # Z_omit_300_2khz, Y_omit_300_2khz, T_omit_300_2khz = pls.trajectory(top_neurons, '2khz_20db_300ms_8b_1xomit_8b_silence')
#     Z_single_A, Y_single_A, T_single_A = pls.trajectory(top_neurons, 'ZF A_20db_180_onebeatcontrol_silence')
#     # Z_omit_180_A_jitter, Y_omit_180_A_jitter, T_omit_180_A_jitter = pls.trajectory(top_neurons, 'ZF A_20db_180ms_8b_1xomit_8b_silence', jitter=.1)
#
#     trajectories = {
#         'Omission 180 ms A': Z_omit_180_A,
#         'Omission 300 ms A': Z_omit_300_A,
#         # 'Omission 180 ms 2khz': Z_omit_180_2khz,
#         # 'Omission 300 ms 2khz': Z_omit_300_2khz,
#         'Single Beat A 180': Z_single_A,
#         'Single Beat A 300': Z_single_A,
#         # 'jitter': Z_omit_180_A_jitter
#     }
#     T = {
#         'Omission 180 ms A': T_omit_180_A,
#         'Omission 300 ms A': T_omit_300_A,
#         # 'Omission 180 ms 2khz': T_omit_180_2khz,
#         # 'Omission 300 ms 2khz': T_omit_300_2khz,
#         'Single Beat A 180': T_single_A,
#         'Single Beat A 300': T_single_A,
#         # 'jitter': T_omit_180_A_jitter
#     }

#     colors = {
#         'Omission 180 ms A': 'blue',
#         'Omission 300 ms A': 'green',
#         # 'Omission 180 ms 2khz': 'red',
#         # 'Omission 300 ms 2khz': 'yellow',
#         'Single Beat A 180': 'purple',
#         'Single Beat A 300': 'black',
#         # 'jitter': 'red',
#     }
#
# #urethane
# else:
#     Z_omit_A_120, Y_omit_A_120, T_omit_A_120 = pls.trajectory(top_neurons, 'ZF A 120ms 8b_omit_8b_silence')
#     Z_omit_A_180, Y_omit_A_180, T_omit_A_180 = pls.trajectory(top_neurons, 'ZF A 180ms 8b_omit_8b_silence')
#     Z_omit_A_220, Y_omit_A_220, T_omit_A_220 = pls.trajectory(top_neurons, 'ZF A 220ms 8b_omit_8b_silence')
#
#     trajectories={
#         'Omission 120 ms': Z_omit_A_120,
#         'Omission 180 ms': Z_omit_A_180,
#         'Omission 220 ms': Z_omit_A_220,
#     }
#     T = {
#         'Omission 120 ms': T_omit_A_120,
#         'Omission 180 ms': T_omit_A_180,
#         'Omission 220 ms': T_omit_A_220,
#     }
#     beat_times = {
#         'Omission 120 ms': np.arange(0, 20, 1)*0.120,
#         'Omission 180 ms': np.arange(0, 20, 1)*0.180,
#         'Omission 220 ms': np.arange(0, 20, 1)*0.220,
#     }
#     colors = {
#         'Omission 120 ms': 'red',
#         'Omission 180 ms': 'blue',
#         'Omission 220 ms': 'green',
#     }
#



#stim detection code - 8/20/26 TDM

# ##
# import scipy
# import numpy as np
#
# def find_beat_times(wav_path, plot=False):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import scipy
#
#     fs, stim_data = scipy.io.wavfile.read(wav_path)
#     stim_data = stim_data/np.max(stim_data)
#     stim_time = np.arange(stim_data.shape[0]) / fs
#
#     # ====== DETECT BEATS =======
#     audio_deriv = np.gradient(stim_data)
#
#     from scipy.signal import find_peaks
#
#     peaks_pos, _ = find_peaks(audio_deriv, height=0.05)  # rising edges
#     peaks_neg, _ = find_peaks(-audio_deriv, height=0.05)  # falling edges
#     all_edges = np.sort(np.concatenate((peaks_pos, peaks_neg)))
#
#     edge_times = stim_time[all_edges]
#     edge_times = np.sort(edge_times)
#
#     grouped_edges = []
#     current_group = [edge_times[0]]
#
#     for time in edge_times[1:]:
#         if time - current_group[-1] <= 0.005:  # 5 ms in seconds
#             current_group.append(time)
#         else:
#             grouped_edges.append(current_group)
#             current_group = [time]
#
#     grouped_edges.append(current_group)
#
#     rising_edges_times = []
#     falling_edges_times = []
#     for group in grouped_edges:
#         start = group[0]
#         stop = group[-1]
#
#         if stop - start > 0.01:
#             rising_edges_times.append(start)
#             falling_edges_times.append(stop)
#
#     if plot:
#         fig, ax = plt.subplots(1,1,figsize=(8,8))
#         ax.plot(stim_time, stim_data)
#         for l in rising_edges_times:
#             ax.axvline(l, color='g')
#         for l in falling_edges_times:
#             ax.axvline(l, color='r')
#         plt.show()
#
#     tempo = np.mean(np.diff(rising_edges_times))
#     return rising_edges_times, falling_edges_times, tempo, stim_time, stim_data
#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(4,1, sharex=True, figsize=(15,10))
# # 1. Regular
# stimulus = 'ZF A_20db_180ms_18b_full_silence'
# stim_path = os.path.join(STIM_LIB_PATH, f'{stimulus}.wav')
# fs, regular_data = scipy.io.wavfile.read(stim_path)
# rising_edges_times, falling_edges_times, _, _, _ = find_beat_times(wav_path=stim_path)
# first_beat_regular = rising_edges_times[0]
# rising_edges_times -= rising_edges_times[0]
# falling_edges_times -= falling_edges_times[0]
# beat_times_regular = rising_edges_times
# omission_beat_regular = beat_times_regular[8]
# beat_times_regular = np.append(beat_times_regular, beat_times_regular[-1]+0.18)
# beat_times_regular = np.append(beat_times_regular, beat_times_regular[-1]+0.18)
# ax[0].plot(np.arange(len(regular_data))/fs-first_beat_regular, regular_data)
# for edge in beat_times_regular:
#     ax[0].axvline(edge, color='g')
# ax[0].set_title('ZF A 180ms Regular')
#
# stimulus = 'ZF A_20db_300ms_8b_1xomit_8b_silence'
# stim_path = os.path.join(STIM_LIB_PATH, f'{stimulus}.wav')
# fs, data_300 = scipy.io.wavfile.read(stim_path)
# rising_edges_times, falling_edges_times, _, _, _ = find_beat_times(wav_path=stim_path)
# first_beat_300 = rising_edges_times[0]
# rising_edges_times -= rising_edges_times[0]
# falling_edges_times -= falling_edges_times[0]
# diffs = np.diff(rising_edges_times)
# omission_loc = np.where(diffs>1.5*np.mean(diffs))[0][0]
# omission_beat_time = rising_edges_times[omission_loc] + np.mean(diffs[:5])
# rising_edges_times = sorted(np.append(rising_edges_times, omission_beat_time))
# beat_times_300 = rising_edges_times
# omission_beat_300 = beat_times_300[8]
# beat_times_300 = np.append(beat_times_300, beat_times_300[-1]+0.30)
# beat_times_300 = np.append(beat_times_300, beat_times_300[-1]+0.30)
# beat_times_300 = np.append(beat_times_300, beat_times_300[-1]+0.30)
# ax[3].plot(np.arange(len(data_300))/fs-first_beat_300, data_300)
# for edge in beat_times_300:
#     ax[3].axvline(edge, color='g')
# ax[3].set_title('ZF A 300ms Omit')
#
# #2. Omission
# stimulus = 'ZF A_20db_180ms_8b_1xomit_8b_silence'
# stim_path = os.path.join(STIM_LIB_PATH, f'{stimulus}.wav')
# fs, omission_data = scipy.io.wavfile.read(stim_path)
# rising_edges_times, falling_edges_times, _, _, _ = find_beat_times(wav_path=stim_path)
# first_beat_omission = rising_edges_times[0]
# rising_edges_times -= rising_edges_times[0]
# falling_edges_times -= falling_edges_times[0]
# diffs = np.diff(rising_edges_times)
# omission_loc = np.where(diffs>1.5*np.mean(diffs))[0][0]
# omission_beat_time = rising_edges_times[omission_loc] + np.mean(diffs[:5])
# rising_edges_times = sorted(np.append(rising_edges_times, omission_beat_time))
# beat_times_omission = rising_edges_times
# omission_beat_omit = beat_times_omission[8]
# beat_times_omission = np.append(beat_times_omission, beat_times_omission[-1]+0.18)
# beat_times_omission = np.append(beat_times_omission, beat_times_omission[-1]+0.18)
# beat_times_omission = np.append(beat_times_omission, beat_times_omission[-1]+0.18)
# ax[1].plot(np.arange(len(omission_data))/fs-first_beat_omission, omission_data)
# for edge in beat_times_omission:
#     ax[1].axvline(edge, color='g')
# ax[1].set_title('ZF A 180ms Omit')
#
# # 3. Irregular
# stimulus = 'ZF A irregular control'
# stim_path = os.path.join(STIM_LIB_PATH, f'{stimulus}.wav')
# fs, irregular_data = scipy.io.wavfile.read(stim_path)
# rising_edges_times, falling_edges_times, _, _, _ = find_beat_times(wav_path=stim_path)
# first_beat_jittered = rising_edges_times[0]
# rising_edges_times -= rising_edges_times[0]
# falling_edges_times -= falling_edges_times[0]
# omission_beat_time = rising_edges_times[omission_loc] + np.mean(diffs[:5])
# rising_edges_times = sorted(np.append(rising_edges_times, omission_beat_time))
# beat_times_jittered  = rising_edges_times
# omission_beat_jittered = rising_edges_times[8]
# beat_times_jittered = np.append(beat_times_jittered, beat_times_jittered[-1]+0.18)
# ax[2].plot(np.arange(len(irregular_data))/fs-first_beat_jittered, irregular_data)
# for edge in beat_times_jittered:
#     ax[2].axvline(edge, color='g')
# ax[2].set_title('ZF A 180ms Irregular')


