import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

from SiNAPSE.core import Neuron, Database, Recording

def plot_all_rasters(RECORDINGS_PATH, recording, db):
    import os
    import matplotlib
    matplotlib.use('Agg')

    from SiNAPSE.core import Recording, Neuron

    select_columns = ['unit_id', 'session_id']
    conditions = {
        'manual_isi_0_7': ('<', 1),
        'session_id': ('=', recording),
    }
    units = db.load_neurons_from_database(select_columns, conditions)

    output_path = r'C:\Users\tmerri03\Desktop\Neural Data\Awake Plots'
    bird = recording.split(' ')[0]
    output_path = os.path.join(output_path, bird, recording)

    if os.path.exists(output_path):
        print(f'Outputs already exist at {output_path}')
        return

    for unit, _ in units:
        rec_fp = os.path.join(RECORDINGS_PATH, bird, recording)

        rec = Recording(rec_fp, samplerate=30000, db=db)
        N = Neuron(recording, unit, rec=rec, db=db)
        stims = N.load
        N.plot(baseline=True)

        N.save_plots(output_path=output_path, format='png', name_prefix=f'{unit}_', clear_cache=True)

def plot_tuning_curves(all_tuning):
    fig, ax = plt.subplots(figsize=(6, 5))

    # --- individual neurons ---
    for tuning in all_tuning:
        ax.plot(
            tuning['frequency'],
            tuning['response strength'],
            color='gray',
            alpha=0.3
        )

    # --- align frequencies across neurons ---
    # (important if some neurons missing freqs)
    all_freqs = sorted(set(
        np.concatenate([t['frequency'].values for t in all_tuning])
    ))

    interp_curves = []

    for tuning in all_tuning:
        interp = np.interp(
            all_freqs,
            tuning['frequency'],
            tuning['response strength'],
            left=np.nan,
            right=np.nan
        )
        interp_curves.append(interp)

    interp_curves = np.array(interp_curves)

    # --- population mean ---
    mean_curve = np.nanmean(interp_curves, axis=0)

    ax.plot(
        all_freqs,
        mean_curve,
        color='red',
        linewidth=3,
        label='population mean'
    )

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Response strength')
    ax.legend()
    plt.show()

def plot_multiunit_activity(RECORDINGS_PATH, recording, channel=0):
    from SiNAPSE.core import Recording

    import numpy as np
    from open_ephys.analysis import Session
    import os
    import pandas as pd

    fs=30000
    bird = recording.split(' ')[0]
    rec_fp = os.path.join(RECORDINGS_PATH, bird, recording)

    oe_base = os.path.join(rec_fp, 'filtered')
    oe_base = os.path.join(oe_base, os.listdir(oe_base)[1])
    session = Session(oe_base)
    recordnode = session.recordnodes[0]
    oe_recording = recordnode.recordings[0]

    log = pd.read_json(os.path.join(rec_fp, 'stimuli.json'))
    stimuli = pd.unique(log['Stimuli Type'])

    def compute_thresholds_from_baseline(oe_recording, stim_windows, fs=30000,
                                         baseline_duration=0.5, factor=-4.5):
        """Estimate per-channel thresholds from pre-stimulus baselines."""
        baseline_chunks = []

        for _, trial in stim_windows.iterrows():
            start_time = trial['Start Time']
            pre_start = int((start_time - baseline_duration) * fs)
            pre_end = int(start_time * fs)

            if pre_start < 0:
                continue

            chunk = oe_recording.continuous[0].get_samples(
                start_sample_index=pre_start,
                end_sample_index=pre_end
            )
            baseline_chunks.append(chunk)

        baseline = np.concatenate(baseline_chunks, axis=0)  # (n_baseline_samples, n_channels)
        noise = np.median(np.abs(baseline) / 0.6745, axis=0)
        return factor * noise  # (n_channels,)

    def detect_crossings(samples, thresholds, window_start, fs=30000, refractory_ms=1.0):
        """
        samples    : (n_samples, n_channels) array for one stimulus window
        thresholds : (n_channels,) array of per-channel thresholds
        returns    : dict {channel: array of spike times in seconds relative to window start}
        """
        refractory_samples = int(refractory_ms * fs / 1000)
        spike_times = {}

        for ch in range(samples.shape[1]):
            sig = samples[:, ch]
            # Find all downward crossings
            crossings = np.where(sig < thresholds[ch])[0]

            if len(crossings) == 0:
                spike_times[ch] = np.array([])
                continue

            # Enforce refractory period — keep only crossings separated by > refractory_samples
            kept = [crossings[0]]
            for c in crossings[1:]:
                if c - kept[-1] > refractory_samples:
                    kept.append(c)

            spike_times[ch] = np.array(kept) / fs  # convert samples to seconds
            spike_times[ch] = spike_times[ch]
        return spike_times

    n_stimuli = len(stimuli)
    n_cols = int(np.ceil(np.sqrt(n_stimuli)))
    n_rows = int(np.ceil(n_stimuli / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows))
    axes = np.array(axes).flatten()  # make indexing consistent regardless of shape

    for ax, stimulus in zip(axes, stimuli):
        print(f'Gathering data for {stimulus}')
        stim_windows = log[log['Stimuli Type'] == stimulus].reset_index(drop=True)
        thresholds = compute_thresholds_from_baseline(oe_recording, stim_windows, fs=fs)

        for t, trial in stim_windows.iterrows():
            print(f' Trial: {t+1}')
            start_time = trial['Start Time']-0.5
            end_time = trial['End Time']+0.5
            samples = oe_recording.continuous[0].get_samples(start_sample_index=int(start_time*fs), end_sample_index=int(end_time*fs), selected_channels=[channel])
            mua_spike_times = detect_crossings(samples, thresholds, start_time)

            all_channel_spikes = np.concatenate([mua_spike_times[ch]-0.5 for ch in mua_spike_times])
            ax.vlines(all_channel_spikes, ymin=t, ymax=t+0.5)

            # ax.scatter(all_channel_spikes, np.full_like(all_channel_spikes, t),
            #            marker='|', s=8, color='black', linewidths=0.5)

        stim_duration = stim_windows.iloc[0]['End Time'] - stim_windows.iloc[0]['Start Time']
        ax.axvline(0, color='red', linewidth=0.6, linestyle='--')
        ax.axvline(stim_duration, color='red', linewidth=0.6, linestyle='--')
        ax.set_title(stimulus, fontsize=7)
        ax.set_xlabel('Time (s)', fontsize=6)
        ax.set_xlim(-0.5, stim_duration+0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=5)

    for ax in axes[n_stimuli:]:
        ax.set_visible(False)

    axes[0].set_ylabel('Trial', fontsize=6)
    plt.tight_layout()
    plt.savefig(f'ch{channel}_mua_raster_by_stimulus.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_tuning_latency(all_tuning):
    rows = []

    for i, tuning in enumerate(all_tuning):
        best_row = tuning.loc[np.abs(tuning['response strength'].idxmax())].copy()
        best_row['cell_id'] = i
        best_row['preferred_freq'] = best_row['frequency']

        rows.append(best_row)

    df = pd.DataFrame(rows)

    plt.figure(figsize=(8,6))
    sc = plt.scatter(
        df['latency'],
        df['response strength'],
        c=df['preferred_freq'],
        cmap='viridis'
    )

    plt.xlabel('Latency (s)')
    plt.ylabel('Response Strength at Preferred Frequency')
    plt.colorbar(sc, label='Preferred Frequency (Hz)')
    plt.show()

def plot_summary(N, save_path=None, baseline={'mean':0, 'std':0}, dpi=300, window_start=0, ntrials=20):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    if save_path is not None:
        save_fp = os.path.join(save_path, f'{N.session_id}_{N.unit_id}_summary.png')
        if os.path.exists(save_fp):
            return
    print(f'Plotting recording: {N.session_id}: {N.unit_id} summary')
    print(f'    Baseline FR: {baseline["mean"]}')
    print(f'    Baseline STD: {baseline["std"]}')

    stims = sorted(N.load)
    n_stims = len(stims)

    ncols = int(np.ceil(np.sqrt(n_stims)))
    n_plot_rows = int(np.ceil(n_stims / ncols))
    nrows = n_plot_rows * 3

    # --- better sizing ---
    fig_width = 10 * ncols
    fig_height = 2 * nrows

    # --- height ratios (key improvement) ---
    height_ratios = [1, 2, 2] * n_plot_rows

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        gridspec_kw={'height_ratios': height_ratios}
    )

    if ncols == 1:
        axes = axes.reshape(-1, 1)

    for s, stimulus in enumerate(stims):
        row = (s // ncols) * 3
        col = s % ncols

        stim_ax   = axes[row + 0, col]
        raster_ax = axes[row + 1, col]
        rs_ax     = axes[row + 2, col]

        # --- plotting ---
        _, time, data = N.plot_stimulus(
            stimulus, ax=stim_ax, plot=True, padding=1
        )

        _, trial_spikes, duration = N.raster(
            stimulus, ax=None, plot=False, padding=1
        )
        _, trial_spikes_separate, _ = N.raster(
            stimulus, ax=None, plot=False, padding=1, separate_trials=True
        )

        if len(trial_spikes_separate) < ntrials:
            ntrials = len(trial_spikes_separate)
        trials_subset = trial_spikes_separate[window_start: window_start+ntrials]
        for t, trial in enumerate(trials_subset):
            raster_ax.vlines(trial, ymin=t, ymax=t+0.5, color='black')

        _, [psth_time, psth_data] = N.psth(
            stimulus, trial_spikes, ax=None,
            color='blue', plot=False, padding=1
        )
        psth_data = np.array(psth_data)
        psth_data = (psth_data - baseline['mean']) / baseline['std']
        rs_ax.plot(psth_time, psth_data, color='blue')
        rs_ax.axhline(0, color='black', linewidth=0.5, alpha=0.5, linestyle='--')

        # --- cosmetics ---
        stim_ax.set_title(stimulus, fontsize=10)

        # share x-axis (important for time plots)
        raster_ax.sharex(rs_ax)
        stim_ax.sharex(rs_ax)

        # remove redundant ticks
        stim_ax.set_xticks([])
        raster_ax.set_xticks([])

    # --- turn off unused ---
    total_slots = n_plot_rows * ncols
    for s_unused in range(n_stims, total_slots):
        row = (s_unused // ncols) * 3
        col = s_unused % ncols
        for i in range(3):
            axes[row + i, col].axis("off")

    plt.tight_layout()

    # --- save ---
    if save_path is not None:
        fig.savefig(save_fp, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    return fig

def parse_json_file(json_path,
                    raw_sound_path = r'C:\Users\tmerri03\Desktop\RhythmStimuli\Awake Recs\Raw Files',
                    stims_fp = r'R:\Data\tyler\Recordings\Stim\Stimuli Library'):
    import json
    import numpy as np
    from scipy.io import wavfile

    with open(json_path, 'r') as file:
        stim_data = json.load(file)
    stims_to_plot = stim_data.keys()

    ret_dict={}
    for s in stims_to_plot:
        stimulus_info = stim_data.get(s)

        samplerate, data = wavfile.read(os.path.join(raw_sound_path, f'{stimulus_info['raw_sound']}.wav'))
        raw_sound_duration = len(data)/samplerate

        gap_duration = stimulus_info['tempo']/1000 - raw_sound_duration
        omission_duration = stimulus_info['omit_beats']*stimulus_info['tempo']/1000 + gap_duration

        beat_times = []
        beat_times.extend(np.arange(stimulus_info['pre_beats']) * stimulus_info['tempo']/1000 + omission_duration)
        beat_times.extend(np.arange(stimulus_info['post_beats']) * stimulus_info['tempo']/1000 + (stimulus_info['pre_beats']+stimulus_info['omit_beats'])*stimulus_info['tempo']/1000 + omission_duration)

        stimulus_info['beat_start_times'] = np.array(beat_times)
        stimulus_info['beat_end_times'] = np.array(beat_times) + raw_sound_duration
        stimulus_info['raw_sound_duration'] = raw_sound_duration
        stimulus_info['wav_fp'] = os.path.join(stims_fp, f'{s}.wav')
        ret_dict[s] = stimulus_info
    return ret_dict

def plot_offsets(N, save_path=None, baseline={'mean':0, 'std':0}, dpi=300, window_start=0, ntrials=20, padding=0):
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.io import wavfile

    if save_path is not None:
        save_fp = os.path.join(save_path, f'{N.session_id}_{N.unit_id}_offsets.png')
        if os.path.exists(save_fp):
            return
    print(f'Plotting offset responses: {N.session_id}: {N.unit_id}')

    #parse json file
    json_path = os.path.join(__file__, '..', '.settings', 'stim.json')
    stimulus_info=parse_json_file(json_path)

    n_stimuli = len(stimulus_info)

    fig, axes = plt.subplots(
        n_stimuli,
        2,
        figsize=(10, 3 * n_stimuli),
        dpi=dpi,
        squeeze=False, sharex=True
    )

    for row, stimulus in enumerate(stimulus_info):

        info = stimulus_info[stimulus]

        _, trial_spikes_separate, _ = N.raster(
            stimulus,
            ax=None,
            plot=False,
            padding=padding,
            separate_trials=True,
        )

        trials_subset = trial_spikes_separate[
            window_start:window_start + ntrials
        ]

        half_tempo = info["tempo"] / 1000 / 2

        aligned_spikes = []

        for beat in info["beat_end_times"]:
            for trial in trials_subset:
                trial = np.asarray(trial)

                window = (
                        (trial >= beat - half_tempo)
                        & (trial <= beat + half_tempo)
                )

                aligned = trial[window] - beat
                aligned_spikes.append(aligned)

        ##################################################
        # Raster
        ##################################################

        ax = axes[row, 0]

        for i, spikes in enumerate(aligned_spikes):
            ax.vlines(spikes, i + 0.5, i + 1.5,
                      color="k", linewidth=0.5)

        ax.axvline(0, color="red", ls="--")
        duration = info['raw_sound_duration']
        ax.axvspan(
            -duration,
            0,
            color='gray',
            alpha=0.2
        )
        ax.set_xlim(-half_tempo, half_tempo)
        ax.set_ylim(0.5, len(aligned_spikes) + 0.5)

        ax.set_title(stimulus)
        ax.set_ylabel("Beat × Trial")

        if row == n_stimuli - 1:
            ax.set_xlabel("Time from beat offset (s)")

        ##################################################
        # PSTH
        ##################################################

        ax = axes[row, 1]

        if len(aligned_spikes):
            all_spikes = np.concatenate(aligned_spikes)

            bins = np.linspace(
                -half_tempo,
                half_tempo,
                51,
            )

            counts, edges = np.histogram(
                all_spikes,
                bins=bins,
            )

            centers = (edges[:-1] + edges[1:]) / 2
            bin_width = np.diff(edges)[0]

            firing_rate = counts / (
                    len(aligned_spikes) * bin_width
            )

            ax.plot(
                centers,
                firing_rate,
                color="k",
                lw=2,
            )

        ax.axvline(0, color="red", ls="--", linewidth=0.7)
        duration = info['raw_sound_duration']
        ax.axvspan(
            -duration,
            0,
            color='gray',
            alpha=0.2
        )

        ax.set_xlim(-half_tempo, half_tempo)

        ax.set_title(f"{stimulus} PSTH")
        ax.set_ylabel("FR (Hz)")

        if row == n_stimuli - 1:
            ax.set_xlabel("Time from beat offset (s)")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_fp, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_waveforms(rec, units, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    if save_path is not None:
        save_fp = os.path.join(save_path, f'{Path(rec.rec_fp).name}_waveforms.png')
        if os.path.exists(save_fp):
            # print('skipping')
            return
    print(f'Plotting waveforms for {Path(rec.rec_fp).name}')

    nunits = len(units)
    ncols = int(np.ceil(np.sqrt(nunits)))
    nrows = int(np.ceil(nunits / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 2.5 * nrows),
        squeeze=False
    )

    axes = axes.ravel()

    recording = Path(rec.rec_fp).name

    for ax, (unit, _, _) in zip(axes, units):
        N = Neuron(recording, unit, rec=rec, db=rec.db)
        N.load
        N.plot_waveform(ax=ax)
        ax.set_title(f"Unit {unit}", fontsize=10)

    # Hide unused axes
    for ax in axes[nunits:]:
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_fp = os.path.join(save_path, f'{Path(rec.rec_fp).name}_waveforms.png')
        plt.savefig(save_fp, dpi=300, bbox_inches="tight")



def plot_probe(rec, units, save_path=None):
    from pathlib import Path
    import os
    import matplotlib.pyplot as plt
    from probeinterface import get_probe
    from probeinterface.plotting import plot_probe as plot_probe_layout

    if save_path is not None:
        save_fp = os.path.join(save_path, f"{Path(rec.rec_fp).name}_probe.png")
        if os.path.exists(save_fp):
            return

    print(f"Plotting probe for {Path(rec.rec_fp).name}")

    # Load probe once
    probe_name = units[0][2]
    probe = get_probe("cambridgeneurotech", probe_name)

    fig, ax = plt.subplots(figsize=(4, 10))

    # Draw probe
    plot_probe_layout(probe, ax=ax)

    # Draw units
    for unit, recording, probe_name, x, y in units:
        ax.scatter(x, y, color="red", s=40, zorder=10)
        ax.text(
            x + 5,
            y,
            str(unit),
            fontsize=8,
            va="center",
            color="red"
        )

    ax.set_title(Path(rec.rec_fp).name)
    ax.set_aspect("equal")

    positions = probe.contact_positions

    xmin = positions[:, 0].min()
    xmax = positions[:, 0].max()
    ymin = positions[:, 1].min()
    ymax = positions[:, 1].max()

    padding = 50  # µm

    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_ylim(ymin - padding, ymax + padding)

    if save_path is not None:
        plt.savefig(save_fp, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()