"""
polar_analysis.py

Beat-omission phase-locking analysis for single units, packaged so it can be
run for any (recording, unit) pair without touching a notebook.

Typical usage
-------------
    from polar_analysis import NeuronOmissionAnalysis

    stim_subset = [
        'ZF A_20db_180ms_8b_1xomit_8b_silence',
        'ZF A_20db_300ms_8b_1xomit_8b_silence',
        '2khz_20db_180ms_8b_1xomit_8b_silence',
        '2khz_20db_300ms_8b_1xomit_8b_silence',
        'ZF A 180ms 12b_2xomit_4b_silence',
        'ZF A 180ms 4b_2xomit_12b_silence',
    ]

    analysis = NeuronOmissionAnalysis(
        db_path=r'C:\\Users\\tmerri03\\Desktop\\Temp Neural Files\\awake_recordings.db',
        stim_lib_path=r'R:\\Data\\tyler\\Recordings\\Stim\\Stimuli Library',
        recordings_path=r'N:\\Data\\RhythmPerception\\Neural Recordings',
        raw_sounds_path=r'C:\\Users\\tmerri03\\Desktop\\RhythmStimuli\\Awake Recs\\Raw Files\\regex',
        recording=r'N:\\Data\\RhythmPerception\\Neural Recordings\\w46o62\\w46o62 Recording #2 (77-H2, Field L, 1.4AP 1.2ML-R 2231um)',
        unit=231,
        stim_subset=stim_subset,
    )
    analysis.run()
    analysis.plot_summary_scatter()
    analysis.plot_summary_polar()

    # drill into one stimulus if needed:
    analysis.analyses['ZF A_20db_180ms_8b_1xomit_8b_silence'].plot()
    analysis.analyses['ZF A_20db_180ms_8b_1xomit_8b_silence'].plot_polar()
"""

import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io.wavfile


from SiNAPSE.core import Neuron, Recording, Database
from _neurons import Response, LatencyCalculator


# ---------------------------------------------------------------------- #
#  Per-stimulus analysis (single beat-omission sequence for one unit)
# ---------------------------------------------------------------------- #
class OmissionAnalysis:
    """
    Beat-omission phase-locking analysis for a single (neuron, stimulus) pair.

    Pipeline: get_spike_times() -> compute_beat_times() ->
    compute_sound_latencies() -> compute_phase_locking() -> plot() / plot_polar()
    """

    def __init__(self, N, stimulus, tempo_ms, n_beats_before, n_omissions,
                 n_beats_after, padding=1):
        """
        N              : neuron object
        stimulus       : stimulus identifier
        tempo_ms       : beat period in ms (e.g. 180)
        n_beats_before : number of beats before the omission
        n_omissions    : number of consecutive omitted beats
        n_beats_after  : number of beats after the omission
        padding        : padding passed to N.raster (1 = spike times relative to padding onset)
        """
        self.N = N
        self.stimulus = stimulus
        self.tempo_ms = tempo_ms
        self.tempo_s = tempo_ms / 1000
        self.n_beats_before = n_beats_before
        self.n_beats_after = n_beats_after
        self.n_omissions = n_omissions
        self.padding = padding

        self.trial_spikes = None
        self.duration = None
        self.beat_times = None       # in seconds, relative to trial onset (padding=0 frame)
        self.omission_times = None   # expected beat times during omission window

    # -------------------------------------------------------------- #
    #  Step 1 — load spikes
    # -------------------------------------------------------------- #
    def get_spike_times(self):
        _, self.trial_spikes, self.duration = self.N.raster(
            stimulus=self.stimulus, baseline=False,
            ax=None, plot=False, separate_trials=True,
            padding=self.padding
        )
        return self

    # -------------------------------------------------------------- #
    #  Step 2 — compute beat times
    # -------------------------------------------------------------- #
    def compute_beat_times(self, silence_duration_s, sound_duration_s,
                            gap_duration_s, n_decay_windows=None,
                            n_pre_windows=None):
        """
        n_decay_windows : number of extra beat positions after post_times to
                          track decay. Defaults to filling the remaining
                          trial duration.
        """
        first_beat = silence_duration_s
        ibi = self.tempo_s

        self.pre_times = np.array([first_beat - ibi - i * ibi
                                    for i in range(n_pre_windows)])

        self.beat_times = np.array([first_beat + i * ibi
                                     for i in range(self.n_beats_before)])

        self.omission_times = np.array([first_beat + (self.n_beats_before + i) * ibi
                                         for i in range(self.n_omissions)])

        self.post_times = np.array([first_beat + (self.n_beats_before + self.n_omissions + i) * ibi
                                     for i in range(self.n_beats_after)])

        # decay windows: continue beat sequence after post_times until end of trial
        post_end_idx = self.n_beats_before + self.n_omissions + self.n_beats_after
        if n_decay_windows is None:
            n_decay_windows = int((self.duration - first_beat) / ibi) - post_end_idx
            n_decay_windows = max(n_decay_windows, 0)

        self.decay_times = np.array([first_beat + (post_end_idx + i) * ibi
                                      for i in range(n_decay_windows)])
        self.decay_times = self.decay_times[self.decay_times < self.duration + self.padding]

        return self

    # -------------------------------------------------------------- #
    #  Step 2.5 — sound-onset latency
    # -------------------------------------------------------------- #
    def compute_sound_latencies(self, latency_control_sound, search_window=0.100,
                                 n_bootstrap=1000, percentile=99, plot=True):
        """
        Computes latency for `latency_control_sound` using bootstrap
        thresholding from this unit's own baseline spike trains.
        """
        baseline_trials, ntrials, nstimuli = self.N.collect_baseline(duration=search_window)

        calc = LatencyCalculator(self.N)
        calc.fit_baseline(
            baseline_trials,
            search_window=search_window,
            n_bootstrap=n_bootstrap,
            percentile=percentile,
        )

        _, spks, _ = self.N.raster(
            stimulus=latency_control_sound, baseline=False,
            ax=None, plot=False, separate_trials=False,
            padding=0
        )

        latency = calc.calculate_latency(spks, search_window=search_window)

        if plot:
            calc.plot(spks, search_window=search_window, latency=latency)

        self.latency_calc = calc  # keep around for inspection/debugging
        self.latency = latency

        return self

    # -------------------------------------------------------------- #
    #  Step 3 — phase locking per beat position
    # -------------------------------------------------------------- #
    def compute_phase_locking(self, window_s=None, latency_s=20 / 1000):
        if self.beat_times is None:
            raise RuntimeError("Call compute_beat_times() first.")
        if window_s is None:
            window_s = self.tempo_s

        all_times = np.concatenate([self.pre_times, self.beat_times,
                                     self.omission_times, self.post_times,
                                     self.decay_times])
        labels = (['baseline'] * len(self.pre_times) +
                  ['beat'] * len(self.beat_times) +
                  ['omission'] * len(self.omission_times) +
                  ['beat'] * len(self.post_times) +
                  ['decay'] * len(self.decay_times))

        self.phase_locking = []

        for beat_t, label in zip(all_times, labels):
            all_phases = []
            for spikes in self.trial_spikes:
                spikes = np.array(spikes)
                near = spikes[
                    (spikes >= beat_t + latency_s) &
                    (spikes < beat_t + latency_s + window_s)
                ]
                phases = ((near - beat_t) / self.tempo_s) * 2 * np.pi
                all_phases.extend(phases.tolist())

            phases = np.array(all_phases)

            if len(phases) > 0:
                z = np.mean(np.exp(1j * phases))
                vs = np.abs(z)
                mean_phase = np.angle(z)                                   # radians, -pi to pi
                mean_phase_ms = (mean_phase / (2 * np.pi)) * self.tempo_ms  # ms into beat cycle
            else:
                vs = np.nan
                mean_phase = np.nan
                mean_phase_ms = np.nan

            self.phase_locking.append({
                'time': beat_t,
                'label': label,
                'vector_strength': vs,
                'mean_phase': mean_phase,        # radians
                'mean_phase_ms': mean_phase_ms,  # ms -- most interpretable
                'n_spikes': len(phases),
                'phases': phases,                # raw phases, used for the polar plot
            })

        self.latency_s = latency_s
        self._compare_beat_vs_omission()  # run comparison automatically
        return self

    def _compare_beat_vs_omission(self, n_bootstrap=1000, verbose=False):
        """
        Compare the pooled beat response against the first omission and first decay
        windows. Stores bootstrap p-values for both comparisons.
        """

        beat_entries = [d for d in self.phase_locking if d['label'] == 'beat']
        beat_phases = np.concatenate([d['phases'] for d in beat_entries]) if beat_entries else np.array([])

        if len(beat_phases) == 0:
            self.phase_comparison = None
            return

        beat_z = np.mean(np.exp(1j * beat_phases))
        beat_mean = np.angle(beat_z)
        beat_vs = np.abs(beat_z)

        self.phase_comparison = {}

        for label in ("baseline", "omission", "decay"):

            entries = [d for d in self.phase_locking if d['label'] == label]

            if len(entries) == 0:
                continue

            # ONLY use the first window
            phases = entries[0]['phases']

            if len(phases) == 0:
                continue

            z = np.mean(np.exp(1j * phases))
            mean_phase = np.angle(z)
            vs = np.abs(z)

            # wrapped angular difference
            diff = np.angle(np.exp(1j * (mean_phase - beat_mean)))
            diff_ms = diff / (2 * np.pi) * self.tempo_ms

            # ---------------- Bootstrap ---------------- #

            all_phases = np.concatenate([beat_phases, phases])
            n_beat = len(beat_phases)

            null_diffs = np.zeros(n_bootstrap)

            for i in range(n_bootstrap):
                shuffled = np.random.permutation(all_phases)

                b = np.angle(np.mean(np.exp(1j * shuffled[:n_beat])))
                x = np.angle(np.mean(np.exp(1j * shuffled[n_beat:])))

                null_diffs[i] = np.abs(
                    np.angle(np.exp(1j * (x - b)))
                )

            observed = np.abs(diff)
            p_value = np.mean(null_diffs <= observed)

            self.phase_comparison[label] = {
                'beat_mean_phase_ms': beat_mean / (2 * np.pi) * self.tempo_ms,
                f'{label}_mean_phase_ms': mean_phase / (2 * np.pi) * self.tempo_ms,
                'difference_ms': diff_ms,
                'p_value': p_value,
                'beat_vs': beat_vs,
                f'{label}_vs': vs,
                'tempo_ms': self.tempo_ms,
            }

        if verbose:
            print(f"\n--- Phase comparison ({self.stimulus}) ---")

            for label in ("omission", "decay"):

                if label not in self.phase_comparison:
                    continue

                d = self.phase_comparison[label]

                print(f"\nBeat vs {label}")
                print(f"  Beat mean phase:   {d['beat_mean_phase_ms']:+.1f} ms")
                print(f"  {label.capitalize()} phase: {d[f'{label}_mean_phase_ms']:+.1f} ms")
                print(f"  Difference:        {d['difference_ms']:+.1f} ms")
                # print(f"  Bootstrap p-value: {d['p_value']:.3f}")

    # -------------------------------------------------------------- #
    #  Step 4 — single-stimulus plots
    # -------------------------------------------------------------- #
    def plot(self, ax=None, show=False):
        """Raster + per-window vector-strength scatter for this stimulus."""

        if not hasattr(self, 'phase_locking'):
            raise RuntimeError("Call compute_phase_locking() first.")

        times = np.array([d['time'] for d in self.phase_locking])
        vs = np.array([d['vector_strength'] for d in self.phase_locking])
        labels = [d['label'] for d in self.phase_locking]

        colors = {
            'baseline': 'forestgreen',
            'beat': 'steelblue',
            'omission': 'tomato',
            'decay': 'gray'
        }

        point_colors = [colors[l] for l in labels]

        # mean phase from beat windows -> predicted spike time offset
        beat_phases = np.concatenate([
            d['phases'] for d in self.phase_locking if d['label'] == 'beat'
        ])

        if len(beat_phases) > 0:
            mean_beat_phase = np.angle(np.mean(np.exp(1j * beat_phases)))
            mean_beat_phase %= (2 * np.pi)
            mean_beat_offset_s = (mean_beat_phase / (2 * np.pi)) * self.tempo_s
        else:
            mean_beat_offset_s = None

        # --------- NEW ---------
        # Create axes only if none were supplied
        created_fig = False

        if ax is None:
            fig, axes = plt.subplots(
                2, 1,
                figsize=(10, 6),
                sharex=True
            )
            created_fig = True
        else:
            axes = ax
            fig = axes[0].figure
        # -----------------------

        # --- top: raster ---
        for i, spikes in enumerate(self.trial_spikes):
            axes[0].scatter(
                np.array(spikes) * 1000,
                np.full(len(spikes), i),
                s=1,
                c='steelblue',
                alpha=0.5
            )

        for t in self.pre_times:
            axes[0].axvline(t * 1000, color=colors['baseline'], lw=0.5, alpha=1, ls=':')
        for t in self.beat_times:
            axes[0].axvline(t * 1000, color=colors['beat'], lw=0.5, alpha=1)
        for t in self.omission_times:
            axes[0].axvline(t * 1000, color=colors['omission'], lw=1, ls='--', alpha=1)
        for t in self.post_times:
            axes[0].axvline(t * 1000, color=colors['beat'], lw=0.5, alpha=1)
        for t in self.decay_times:
            axes[0].axvline(t * 1000, color=colors['decay'], lw=0.5, alpha=1, ls=':')

        if mean_beat_offset_s is not None:
            for t in self.omission_times:
                predicted_ms = (t + mean_beat_offset_s) * 1000
                axes[0].axvline(predicted_ms, color='tomato', lw=1.5,
                                 ls=':', alpha=0.9,
                                 label='predicted (beat mean phase)' if t == self.omission_times[0] else None)

        axes[0].set_ylabel('Trial')
        axes[0].set_title(self.stimulus)
        axes[0].legend(loc='upper right', fontsize=7)

        # --- bottom: vector strength ---
        axes[1].scatter((times + self.tempo_s / 2) * 1000, vs,
                         c=point_colors, s=40, zorder=3)
        axes[1].axhline(0, color='gray', lw=0.5)

        for t in self.pre_times:
            axes[1].axvspan((t + self.latency_s) * 1000,
                             (t + self.tempo_s + self.latency_s) * 1000,
                             alpha=0.07, color='gray')
        for t in self.omission_times:
            axes[1].axvspan((t + self.latency_s) * 1000,
                             (t + self.tempo_s + self.latency_s) * 1000,
                             alpha=0.1, color='tomato')
            if mean_beat_offset_s is not None:
                predicted_ms = (t + self.latency_s + mean_beat_offset_s) * 1000
                axes[1].axvline(predicted_ms, color='tomato', lw=1.5, ls=':', alpha=0.9)
        for t in self.decay_times:
            axes[1].axvspan((t + self.latency_s) * 1000,
                             (t + self.tempo_s + self.latency_s) * 1000,
                             alpha=0.07, color='gray')

        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Vector strength')
        axes[1].set_title('Phase locking per beat position  (blue = beat, red = omission, dotted = predicted)')
        axes[1].set_ylim(0, 1)

        if show:
            plt.tight_layout()
            plt.show()

        return fig, axes

    def plot_polar(self, windows=('beat', 'omission'), bins=24, ax=None, show=True):
        """
        Polar plot of spike phase distributions.

        baseline, beat
            pooled across all windows of that label.

        omission, decay
            plotted separately for each window.
        """
        if not hasattr(self, 'phase_locking'):
            raise RuntimeError("Call compute_phase_locking() first.")

        import numpy as np
        import matplotlib.pyplot as plt

        colors = {
            'baseline': 'forestgreen',
            'beat': 'steelblue',
            'omission': 'tomato',
            'decay': 'gray'
        }

        pooled_labels = {'baseline', 'beat'}
        split_labels = {'omission', 'decay'}

        created_fig = False
        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='polar')
            created_fig = True

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        edges = np.linspace(0, 2 * np.pi, bins + 1)
        width = edges[1] - edges[0]
        centers = edges[:-1] + width / 2

        plot_data = []

        ###########################################################
        # pooled labels
        ###########################################################

        for label in pooled_labels.intersection(windows):

            entries = [d for d in self.phase_locking if d['label'] == label]

            if len(entries) == 0:
                continue

            phases = np.concatenate([d['phases'] for d in entries])

            counts, _ = np.histogram(np.mod(phases, 2 * np.pi), bins=edges)
            density = counts / counts.sum()

            z = np.mean(np.exp(1j * phases))
            plot_data.append(dict(
                label=label,
                density=density,
                phases=phases,
                mean_phase=np.angle(z) % (2 * np.pi),
                vs=np.abs(z),
                color=colors[label],
                alpha=0.45,
                legend=f"{label.capitalize()} (pooled)"
            ))

        ###########################################################
        # split labels
        ###########################################################

        for label in split_labels.intersection(windows):

            entries = [d for d in self.phase_locking if d['label'] == label]

            if len(entries) == 0:
                continue

            # darker first, lighter later
            alphas = np.linspace(0.85, 0.3, len(entries))

            for i, (entry, alpha) in enumerate(zip(entries, alphas), start=1):
                phases = entry['phases']

                counts, _ = np.histogram(np.mod(phases, 2 * np.pi), bins=edges)
                density = counts / counts.sum()

                z = np.mean(np.exp(1j * phases))

                plot_data.append(dict(
                    label=label,
                    density=density,
                    phases=phases,
                    mean_phase=np.angle(z) % (2 * np.pi),
                    vs=np.abs(z),
                    color=colors[label],
                    alpha=alpha,
                    legend=f"{label.capitalize()} {i}"
                ))

        ###########################################################

        rmax = max(max(d['density'].max(), d['vs']) for d in plot_data)

        for d in plot_data:
            ax.bar(
                centers,
                d['density'],
                width=width * 0.9,
                color=d['color'],
                edgecolor=d['color'],
                linewidth=0.5,
                alpha=d['alpha'],
                label=f"{d['legend']} (n={len(d['phases'])})"
            )

            ax.annotate(
                '',
                xy=(d['mean_phase'], d['vs']),
                xytext=(0, 0),
                arrowprops=dict(
                    facecolor=d['color'],
                    edgecolor=d['color'],
                    width=2,
                    headwidth=8,
                    alpha=d['alpha']
                )
            )

        ax.set_ylim(0, rmax * 1.15)

        ax.set_title(
            f"{self.stimulus}\nPhase distribution and mean vector",
            fontsize=10
        )

        ax.legend(
            loc='upper right',
            bbox_to_anchor=(1.35, 1.1),
            fontsize=7
        )

        if created_fig and show:
            plt.tight_layout()
            plt.show()

        return ax


# ---------------------------------------------------------------------- #
#  Stimulus-string parsing helper
# ---------------------------------------------------------------------- #
def parse_stimulus_string(s, raw_sounds_fp, stim_lib_fp):
    """
    Parse a stimulus filename into the parameters OmissionAnalysis needs
    (tempo, beat counts, sound durations, control stimulus for latency, ...).
    """
    fs, data = scipy.io.wavfile.read(os.path.join(stim_lib_fp, f'{s}.wav'))
    total_duration = len(data) / fs

    s = s.replace(' ', '_').replace('2khz', '2_khz')

    # extract structured fields by pattern, not position
    tempo = int(re.search(r'(\d+)ms', s).group(1))
    n_before = int(re.search(r'(\d+)b_\d+xomit', s).group(1))
    n_omissions = int(re.search(r'(\d+)xomit', s).group(1))
    n_after = int(re.search(r'xomit_(\d+)b', s).group(1))

    # sound name is everything before the tempo token
    raw_sound = re.search(r'^(.+?)_\d+ms_', s).group(1)
    raw_sound = raw_sound.replace('2_khz', '2khz')

    fs, data = scipy.io.wavfile.read(os.path.join(raw_sounds_fp, f'{raw_sound}.wav'))
    raw_sound_duration = len(data) / fs
    gap_duration = tempo / 1000 - raw_sound_duration

    if total_duration > (n_before + n_omissions + n_after) * tempo / 1000:
        omission_duration = n_omissions * (tempo / 1000) + gap_duration
    else:
        omission_duration = 0

    latency_sound_prefix = raw_sound.replace('ZF_A_20db', 'ZF A_20db')

    return {
        'tempo': tempo,
        'n_beats_before': n_before,
        'n_omissions': n_omissions,
        'n_beats_after': n_after,
        'raw_sound': raw_sound,
        'raw_sound_duration': raw_sound_duration,
        'gap_duration': gap_duration,
        'omission_duration': omission_duration,
        'latency_control_sound': f'{latency_sound_prefix}_180_onebeatcontrol_silence',
    }


# ---------------------------------------------------------------------- #
#  Cross-stimulus summary plots (module-level, take a dict of analyses)
# ---------------------------------------------------------------------- #
def plot_summary_scatter(analyses, ax=None, show=True):
    """
    One scatter plot, all stimuli overlaid: vector strength vs. beat index,
    aligned so beat index 0 = the first omitted beat (makes stimuli with
    different tempos directly comparable). Circles = beat windows,
    X markers = omission windows. One color per stimulus.
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(9, 5))

    cmap = matplotlib.colormaps['tab10']
    colors = cmap(range(max(len(analyses), 1)))

    for i, (stim_name, oa) in enumerate(analyses.items()):
        color = cmap(i)
        t0 = oa.omission_times[0]

        beat_idx, vs, labels = [], [], []
        for d in oa.phase_locking:
            if d['label'] not in ('beat', 'omission'):
                continue
            beat_idx.append((d['time'] - t0) / oa.tempo_s)
            vs.append(d['vector_strength'])
            labels.append(d['label'])

        beat_idx = np.array(beat_idx)
        vs = np.array(vs)
        is_omit = np.array(labels) == 'omission'

        ax.scatter(beat_idx[~is_omit], vs[~is_omit], color=color, marker='o',
                   s=35, alpha=0.8, label=f'{stim_name} (beat)')
        ax.scatter(beat_idx[is_omit], vs[is_omit], color=color, marker='X',
                   s=70, alpha=0.95, edgecolor='black', linewidth=0.5,
                   label=f'{stim_name} (omission)')

    ax.axvline(0, color='gray', ls='--', lw=1, alpha=0.6)
    ax.set_xlabel('Beat index (0 = first omitted beat)')
    ax.set_ylabel('Vector strength')
    ax.set_ylim(0, 1)
    ax.set_title('Phase locking across stimuli  (o = beat, X = omission)')
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.02, 0.5))

    if created_fig and show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_summary_polar(analyses, ax=None, show=True):
    """
    One polar plot, all stimuli overlaid: mean-vector arrow per stimulus.
    Bold arrow = omission window, faint arrow = beat window (reference).
    Direction = mean phase (timing), length = vector strength (tightness).
    """
    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='polar')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    cmap = matplotlib.colormaps['tab10']
    colors = cmap(range(max(len(analyses), 1)))
    rmax = 0.0

    for i, (stim_name, oa) in enumerate(analyses.items()):
        color = cmap(i)
        for label, alpha, lwidth in (('beat', 0.35, 1.5), ('omission', 0.95, 2.5)):
            phases = np.concatenate(
                [d['phases'] for d in oa.phase_locking if d['label'] == label]
            ) if any(d['label'] == label for d in oa.phase_locking) else np.array([])
            if len(phases) == 0:
                continue

            z = np.mean(np.exp(1j * phases))
            vs = np.abs(z)
            mean_phase = np.angle(z) % (2 * np.pi)
            rmax = max(rmax, vs)

            ax.annotate('', xy=(mean_phase, vs), xytext=(0, 0),
                        arrowprops=dict(facecolor=color, edgecolor=color,
                                         alpha=alpha, width=lwidth, headwidth=8))

        ax.plot([], [], color=color, label=stim_name)  # legend swatch

    ax.set_ylim(0, max(rmax * 1.15, 0.1))
    ax.set_title('Mean phase-locking vector per stimulus\n(faint = beat, bold = omission)',
                 fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=7)

    if created_fig and show:
        plt.tight_layout()
        plt.show()

    return ax


# ---------------------------------------------------------------------- #
#  Top-level, per-neuron entry point
# ---------------------------------------------------------------------- #
class NeuronOmissionAnalysis:
    """
    Run the full beat-omission phase-locking pipeline for one (recording, unit)
    pair across a set of stimuli, and produce cross-stimulus summary plots.

    Parameters
    ----------
    db_path : str
        Path to the recordings database (.db file).
    stim_lib_path : str
        Path to the stimulus library folder (contains the full stimulus .wav files).
    recordings_path : str
        Path to the root "Neural Recordings" folder.
    raw_sounds_path : str
        Path to the folder of raw (un-sequenced) sound .wav files, used to
        work out per-beat sound/gap durations.
    recording : str
        Either the recording folder name, or a full path to it (only the
        folder name is used).
    unit : int
        Unit / cluster ID for this neuron.
    stim_subset : list of str
        Stimulus identifiers to analyze for this neuron.
    samplerate : int, default 30000
        Recording sample rate in Hz.

    Attributes (populated after .run())
    ------------------------------------
    analyses : dict {stimulus_name: OmissionAnalysis}
    """

    def __init__(self, db_path, stim_lib_path, recordings_path, raw_sounds_path,
                 recording, unit, stim_subset, samplerate=30000):
        self.db_path = db_path
        self.stim_lib_path = stim_lib_path
        self.recordings_path = recordings_path
        self.raw_sounds_path = raw_sounds_path
        self.unit = unit
        self.stim_subset = list(stim_subset)
        self.samplerate = samplerate
        self.phase_comparison = None
        # full path -> just the recording folder name (matches DB convention)
        recording_path = Path(recording)
        self.recording_name = recording_path.name
        self.recording_path = str(recording_path) if recording_path.is_absolute() else str(
            Path(recordings_path) / self.recording_name
        )

        # --- wire up SiNAPSE objects ---
        self.db = Database(self.db_path, self.stim_lib_path)
        self.rec = Recording(self.recording_path, samplerate=self.samplerate, db=self.db)
        self.N = Neuron(self.recording_name, self.unit, rec=self.rec, db=self.db)
        self.N.load
        self.R = Response(self.stim_lib_path, self.recordings_path, self.recording_name, db=self.db)

        # parse stimulus metadata up front (tempo, beat counts, control sound, ...)
        self.stims = {
            s: parse_stimulus_string(s, raw_sounds_fp=self.raw_sounds_path,
                                      stim_lib_fp=self.stim_lib_path)
            for s in self.stim_subset
        }

        self.analyses = {}  # populated by .run()

    # -------------------------------------------------------------- #
    #  Run the pipeline for every stimulus in stim_subset
    # -------------------------------------------------------------- #
    def run(self, n_pre_windows=5, n_decay_windows=5, latency_search_window=0.100,
            n_bootstrap=1000, percentile=99, verbose=False):
        """
        Runs get_spike_times -> compute_beat_times -> compute_sound_latencies
        -> compute_phase_locking for every stimulus in self.stim_subset.
        Stores results in self.analyses. Returns self.
        """
        self.analyses = {}
        self.phase_comparison = {}
        for stimulus in self.stim_subset:
            stim_data = self.stims[stimulus]

            oa = OmissionAnalysis(
                self.N, stimulus=stimulus,
                tempo_ms=stim_data['tempo'],
                n_beats_before=stim_data['n_beats_before'],
                n_omissions=stim_data['n_omissions'],
                n_beats_after=stim_data['n_beats_after'],
            )
            oa.get_spike_times()
            oa.compute_beat_times(
                silence_duration_s=stim_data['omission_duration'],
                sound_duration_s=stim_data['raw_sound_duration'],
                gap_duration_s=stim_data['raw_sound_duration'],
                n_pre_windows=n_pre_windows,
                n_decay_windows=n_decay_windows,
            )
            oa.compute_sound_latencies(
                latency_control_sound=stim_data['latency_control_sound'],
                search_window=latency_search_window,
                n_bootstrap=n_bootstrap,
                percentile=percentile,
                plot=False,
            )
            oa.compute_phase_locking()
            self.phase_comparison[stimulus] = oa.phase_comparison
            if not verbose:
                pass  # phase-locking comparison print is inside OmissionAnalysis; harmless to leave

            self.analyses[stimulus] = oa

        return self

    # -------------------------------------------------------------- #
    #  Cross-stimulus summary plots
    # -------------------------------------------------------------- #
    def plot_summary_scatter(self, ax=None, show=True):
        if not self.analyses:
            raise RuntimeError("Call .run() first.")
        return plot_summary_scatter(self.analyses, ax=ax, show=show)

    def plot_summary_polar(self, ax=None, show=True):
        if not self.analyses:
            raise RuntimeError("Call .run() first.")
        return plot_summary_polar(self.analyses, ax=ax, show=show)

    # -------------------------------------------------------------- #
    #  Convenience: single-stimulus plots
    # -------------------------------------------------------------- #
    def plot_stimulus(self, stimulus):
        """Raster + vector-strength scatter for one stimulus."""
        return self.analyses[stimulus].plot()

    def plot_stimulus_polar(self, stimulus, **kwargs):
        """Polar phase plot for one stimulus."""
        return self.analyses[stimulus].plot_polar(**kwargs)