import pandas as pd
import numpy as np
import os

path = r"C:\Users\tmerri03\Desktop\PLS Results\GRFP\trained\HVC_top_neurons_100.csv"
df = pd.read_csv(path)

stimulus = "exp1_ZF A_20db_180ms_8b_1xomit_8b_silence"
tempo = 0.180

import scipy
sound_element_path = r'C:\Users\tmerri03\Desktop\RhythmStimuli\Tempi Range Stimuli 8-25-26\Raw Files\ZF A_20db.wav'
fs, data = scipy.io.wavfile.read(sound_element_path)
sound_dur = len(data) / fs
print(f'Sound element duration: {sound_dur*1000}ms')
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

from SiNAPSE.core import Recording, Neuron, Database
db_path = r'C:\Users\tmerri03\Desktop\Temp Neural Files\awake_recordings.db'
stim_lib_path = r'R:\Data\tyler\Recordings\Stim\Stimuli Library'
recordings_path = r'N:\Data\RhythmPerception\Neural Recordings'
db = Database(db_path, stim_lib_path)

for r, row in df.iterrows():
    recording = row['recording']
    unit = row['unit']

    if not recording == 'y92y99 Recording #4 (77-H2, HVC, 0.2AP 2.4ML-L 684um)':
        continue

    rec = Recording(os.path.join(recordings_path, recording.split(' ')[0], recording), samplerate=30000, db=db)
    N = Neuron(recording, unit, rec=rec, db=db)
    stims = N.load

    import matplotlib.pyplot as plt
    padding=0.2
    _, spikes, duration = N.raster(stimulus, baseline=False, plot=False, ax=None, separate_trials=True, padding=padding)

    fig, ax = plt.subplots(
        2, 1,
        figsize=(4, 2),
        sharex=True,
        gridspec_kw={"hspace": 0.05}
    )

    # order
    raster_ax = ax[0]
    psth_ax = ax[1]


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

    raster_ax.set_ylabel("Trial")
    raster_ax.set_yticks([0, len(spikes)])
    raster_ax.tick_params(axis='y')

    # psth
    all_spikes = np.concatenate(spikes)

    bin_size = 0.01

    bins = np.arange(
        -padding,
        padding + duration + bin_size,
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

    psth_ax.set_ylabel("Rate (hz)")
    psth_ax.set_xlabel("Time (s)")
    psth_ax.set_yticks([0, 50])
    psth_ax.tick_params(axis='x')
    psth_ax.tick_params(axis='y')


    # format
    raster_ax.spines['top'].set_visible(False)
    raster_ax.spines['right'].set_visible(False)
    raster_ax.spines['bottom'].set_visible(False)
    psth_ax.spines['top'].set_visible(False)
    psth_ax.spines['right'].set_visible(False)
    raster_ax.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False
    )

    psth_ax.set_xticks([0, 1, 2, 3])
    psth_ax.tick_params(axis='x')
    psth_ax.set_xlabel("Time (s)")
    fig.suptitle(f'Unit {unit}')
    plt.tight_layout()
    plt.savefig(fr'C:\Users\tmerri03\Desktop\PLS Results\GRFP\trained\units\{unit}.svg', format='svg')
    plt.show()

