from SiNAPSE.core import Recording, Neuron, Database
import os
import numpy as np

db_path = r'C:\Users\tmerri03\Desktop\Temp Neural Files\awake_recordings.db'
stim_lib_path = r'R:\Data\tyler\Recordings\Stim\Stimuli Library'
recordings_path = r'N:\Data\RhythmPerception\Neural Recordings'
recording = r'w46o62 Recording #2 (77-H2, Field L, 1.4AP 1.2ML-R 2231um)'
bird = recording.split(' ')[0]
recording_path = os.path.join(recordings_path, bird, recording)
sound_element_path = r'C:\Users\tmerri03\Desktop\RhythmStimuli\Tempi Range Stimuli 8-25-26\Raw Files\ZF A_20db.wav'
stimulus = 'ZF A_20db_180ms_8b_1xomit_8b_silence'
tempo = 0.180
label_fontsize = 10
tick_fontsize = 8
db = Database(db_path, stim_lib_path)
rec = Recording(recording_path, samplerate=30000, db=db)
N = Neuron(recording, 226, db=db, rec=rec)
stims = N.load

#get spike times
if stimulus not in stims:
    raise ValueError('Stimulus not in recording')

padding = 0.2
_, spikes, duration = N.raster(stimulus, baseline=False, plot=False, ax=None, separate_trials=True, padding=padding)

import matplotlib.pyplot as plt
raw_data_fig, [stimulus_ax, raw_data_ax] = plt.subplots(2,1,figsize=(3,5), sharex=True)
N.padding=padding
# for i in range(0,19):
#     N.plot_raw_data(stimulus, ax=raw_data_ax[i], trial=i)
N.plot_stimulus(stimulus, ax=stimulus_ax)
N.plot_raw_data(stimulus, ax=raw_data_ax, trial=18)
raw_data_ax.spines['right'].set_visible(False)
raw_data_ax.spines['top'].set_visible(False)
raw_data_ax.spines['left'].set_visible(False)
raw_data_ax.spines['bottom'].set_visible(False)
raw_data_ax.set_xticks([])
raw_data_ax.set_yticks([])
plt.savefig('GRFP_raw_data.svg', format='svg')
plt.show()

N.plot_waveform()
plt.savefig('GRFP_waveform.svg', format='svg')
plt.show()

import scipy
fs, data = scipy.io.wavfile.read(sound_element_path)
sound_dur = len(data) / fs
print(f'Sound element duration: {sound_dur*1000}ms')

beat_times_pre = np.arange(0,8) * tempo + sound_dur + tempo
beat_times_post = np.arange(0,8) * tempo + beat_times_pre[-1] + 2*tempo
omit_time = beat_times_pre[-1] + tempo

import numpy as np
import matplotlib.pyplot as plt

#beat times
beat_times_pre = np.arange(0, 8) * tempo + sound_dur + tempo
beat_times_post = (
    np.arange(0, 8) * tempo
    + beat_times_pre[-1]
    + 2 * tempo
)

omit_time = beat_times_pre[-1] + tempo

all_beat_times = np.concatenate([
    beat_times_pre,
    beat_times_post
])


fig, ax = plt.subplots(
    3, 1,
    figsize=(4, 2),
    height_ratios=[0.2, 4, 1.5],
    sharex=True,
    gridspec_kw={"hspace": 0.05}
)

#order
raster_ax = ax[1]
psth_ax = ax[2]
beat_ax = ax[0]

all_beat_times = np.asarray(all_beat_times)

# Time of first beat
t0 = all_beat_times[0]

# Shift spikes so first beat = 0
spikes = [
    np.asarray(trial) - t0
    for trial in spikes
]

# Shift beat times and omission
all_beat_times = all_beat_times - t0
omit_time = omit_time - t0

# Plot raster
for t, trial in enumerate(spikes):
    raster_ax.vlines(
        trial,
        ymin=t,
        ymax=t + 1,
        linewidth=0.5
    )

raster_ax.set_ylabel("Trial", fontsize=label_fontsize)
raster_ax.set_yticks([0,len(spikes)])
raster_ax.tick_params(axis='y', labelsize=tick_fontsize)

#psth
all_spikes = np.concatenate(spikes)

bin_size = 0.01

bins = np.arange(
    -padding,
    padding+duration+bin_size,
    bin_size
)

counts, edges = np.histogram(
    all_spikes,
    bins=bins
)

psth = counts / (len(spikes) * bin_size)

from scipy.ndimage import gaussian_filter1d
psth_smooth = gaussian_filter1d(
    psth,
    sigma=1
)

centers = (edges[:-1] + edges[1:]) / 2

psth_ax.plot(
    centers,
    psth_smooth,
    linewidth=1
)

psth_ax.set_ylabel("Rate (hz)", fontsize=label_fontsize)
psth_ax.set_xlabel("Time (s)", fontsize=label_fontsize)
psth_ax.set_yticks([0, 50])
psth_ax.tick_params(axis='x', labelsize=tick_fontsize)
psth_ax.tick_params(axis='y', labelsize=tick_fontsize)

#beat lines
y0, y1 = 0.05, 0.25

# for bt in beat_times_pre:
#     beat_ax.fill_between(
#         [bt, bt + sound_dur],
#         y0, y1,
#         color="green",
#         alpha=0.8,
#         linewidth=0
#     )
#
# for bt in beat_times_post:
#     beat_ax.fill_between(
#         [bt, bt + sound_dur],
#         y0, y1,
#         color="green",
#         alpha=0.8,
#         linewidth=0
#     )

for bt in all_beat_times:
    beat_ax.fill_between(
        [bt, bt+sound_dur],
        y0, y1,
        color='green',
        alpha=0.8,
        linewidth=0
    )
# Omitted beat
beat_ax.fill_between(
    [omit_time, omit_time + sound_dur],
    y0, y1,
    color="red",
    alpha=0.8,
    linewidth=0
)

beat_ax.set_ylim(0, 0.25)
beat_ax.set_xticks([])
beat_ax.set_yticks([])

#format
raster_ax.spines['top'].set_visible(False)
raster_ax.spines['right'].set_visible(False)
raster_ax.spines['bottom'].set_visible(False)
psth_ax.spines['top'].set_visible(False)
psth_ax.spines['right'].set_visible(False)
beat_ax.spines['top'].set_visible(False)
beat_ax.spines['right'].set_visible(False)
beat_ax.spines['left'].set_visible(False)
beat_ax.spines['top'].set_visible(False)
beat_ax.spines['bottom'].set_visible(False)
raster_ax.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)
beat_ax.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)
psth_ax.set_xticks([0,1,2,3])
psth_ax.tick_params(axis='x', labelsize=tick_fontsize)
psth_ax.set_xlabel("Time (s)", fontsize=label_fontsize)

plt.tight_layout()
plt.savefig('GRFP.svg', format='svg')
plt.show()