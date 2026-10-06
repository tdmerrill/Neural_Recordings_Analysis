from SiNAPSE.core import Database, Neuron, Recording
import numpy as np
import os, re, json
from scipy.signal import hilbert, resample
from scipy.stats import zscore
from scipy.signal import correlate
from scipy.io import wavfile

# SQL interface
def init_db(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS neuron_features
        (
            session_id TEXT,
            unit_id INTEGER,

            -- metadata (not clustering features)
            bird_id TEXT,
            brain_region TEXT,
            training_status TEXT,

            -- basic physiology
            baseline_fr REAL,
            response_probability REAL,
            mean_pairwise_trial_corr REAL,
            spike_width REAL,

            -- excitation / inhibition
            peak_excitation REAL,
            peak_suppression REAL,

            excitation_latency REAL,
            suppression_latency REAL,

            excitation_duration REAL,
            suppression_duration REAL,

            excitation_inhibition_index REAL,

            -- response shape
            response_centroid REAL,
            response_skewness REAL,
            num_response_peaks INTEGER,

            -- adaptation dynamics
            adaptation_index REAL,
            adaptation_slope REAL,
            steady_state_response REAL,
            recovery_after_silence REAL,

            -- temporal coding
            vector_strength REAL,
            preferred_phase REAL,

            stimulus_correlation REAL,
            stimulus_lag REAL,

            -- spectral properties
            best_frequency REAL,
            bandwidth REAL,

            -- natural sound selectivity
            natural_sound_preference REAL,

            -- PSTH waveform embedding
            psth_pc1 REAL,
            psth_pc2 REAL,
            psth_pc3 REAL,

            -- assigned after clustering
            cluster_id INTEGER,

            UNIQUE(session_id, unit_id)
        )
    """)

    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_neuron_features
        ON neuron_features(session_id, unit_id)
    """)

    conn.commit()
    conn.close()

def update_SQL(db_path, data):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO neuron_features
        (
            session_id,
            unit_id,

            bird_id,
            brain_region,
            training_status,

            baseline_fr,
            response_probability,
            mean_pairwise_trial_corr,
            spike_width,

            peak_excitation,
            peak_suppression,

            excitation_latency,
            suppression_latency,

            excitation_duration,
            suppression_duration,

            excitation_inhibition_index,

            response_centroid,
            response_skewness,
            num_response_peaks,

            adaptation_index,
            adaptation_slope,
            steady_state_response,
            recovery_after_silence,

            vector_strength,
            preferred_phase,

            stimulus_correlation,
            stimulus_lag,

            best_frequency,
            bandwidth,

            natural_sound_preference,

            psth_pc1,
            psth_pc2,
            psth_pc3,

            cluster_id
        )

        VALUES
        (
            ?,  -- session_id
            ?,  -- unit_id

            ?,  -- bird_id
            ?,  -- brain_region
            ?,  -- training_status

            ?,  -- baseline_fr
            ?,  -- response_probability
            ?,  -- mean_pairwise_trial_corr
            ?,  -- spike_width

            ?,  -- peak_excitation
            ?,  -- peak_suppression

            ?,  -- excitation_latency
            ?,  -- suppression_latency

            ?,  -- excitation_duration
            ?,  -- suppression_duration

            ?,  -- excitation_inhibition_index

            ?,  -- response_centroid
            ?,  -- response_skewness
            ?,  -- num_response_peaks

            ?,  -- adaptation_index
            ?,  -- adaptation_slope
            ?,  -- steady_state_response
            ?,  -- recovery_after_silence

            ?,  -- vector_strength
            ?,  -- preferred_phase

            ?,  -- stimulus_correlation
            ?,  -- stimulus_lag

            ?,  -- best_frequency
            ?,  -- bandwidth

            ?,  -- natural_sound_preference

            ?,  -- psth_pc1
            ?,  -- psth_pc2
            ?,  -- psth_pc3

            ?   -- cluster_id
        )

        ON CONFLICT(session_id, unit_id)
        DO UPDATE SET

            bird_id = excluded.bird_id,
            brain_region = excluded.brain_region,
            training_status = excluded.training_status,

            baseline_fr = excluded.baseline_fr,
            response_probability = excluded.response_probability,
            mean_pairwise_trial_corr = excluded.mean_pairwise_trial_corr,
            spike_width = excluded.spike_width,
            
            peak_excitation = excluded.peak_excitation,
            peak_suppression = excluded.peak_suppression,

            excitation_latency = excluded.excitation_latency,
            suppression_latency = excluded.suppression_latency,

            excitation_duration = excluded.excitation_duration,
            suppression_duration = excluded.suppression_duration,

            excitation_inhibition_index = excluded.excitation_inhibition_index,

            response_centroid = excluded.response_centroid,
            response_skewness = excluded.response_skewness,
            num_response_peaks = excluded.num_response_peaks,

            adaptation_index = excluded.adaptation_index,
            adaptation_slope = excluded.adaptation_slope,
            steady_state_response = excluded.steady_state_response,
            recovery_after_silence = excluded.recovery_after_silence,

            vector_strength = excluded.vector_strength,
            preferred_phase = excluded.preferred_phase,

            stimulus_correlation = excluded.stimulus_correlation,
            stimulus_lag = excluded.stimulus_lag,

            best_frequency = excluded.best_frequency,
            bandwidth = excluded.bandwidth,

            natural_sound_preference = excluded.natural_sound_preference,

            psth_pc1 = excluded.psth_pc1,
            psth_pc2 = excluded.psth_pc2,
            psth_pc3 = excluded.psth_pc3,

            cluster_id = excluded.cluster_id
        """,
        (
            data["session_id"],
            data["unit_id"],

            data.get("bird_id"),
            data.get("brain_region"),
            data.get("training_status"),

            data.get("baseline_fr"),
            data.get("response_probability"),
            data.get("mean_pairwise_trial_corr"),
            data.get("spike_width"),

            data.get("peak_excitation"),
            data.get("peak_suppression"),

            data.get("excitation_latency"),
            data.get("suppression_latency"),

            data.get("excitation_duration"),
            data.get("suppression_duration"),

            data.get("excitation_inhibition_index"),

            data.get("response_centroid"),
            data.get("response_skewness"),
            data.get("num_response_peaks"),

            data.get("adaptation_index"),
            data.get("adaptation_slope"),
            data.get("steady_state_response"),
            data.get("recovery_after_silence"),

            data.get("vector_strength"),
            data.get("preferred_phase"),

            data.get("stimulus_correlation"),
            data.get("stimulus_lag"),

            data.get("best_frequency"),
            data.get("bandwidth"),

            data.get("natural_sound_preference"),

            data.get("psth_pc1"),
            data.get("psth_pc2"),
            data.get("psth_pc3"),

            data.get("cluster_id"),
        ),
    )

    conn.commit()
    conn.close()

#helper functions
def lookup_bird(bird_id,
                lookup_fp = None):
    import os, json

    if lookup_fp is None:
        lookup_fp = os.path.join(__file__, '..', 'birds.json')

    with open(lookup_fp, 'r') as f:
        lookup = json.load(f)

        for status in lookup.keys():
            if bird_id in lookup[status]:
                return status

def find_tone_order(stim_lib_path, tone_fn='tone_order.txt'):
    order_path = os.path.join(stim_lib_path, tone_fn)
    if not os.path.exists(order_path):
        print(f'Could not find {tone_fn} in {stim_lib_path}')
        return

    tones = {}
    with open(order_path, 'r') as f:
        content = f.read()

    pattern = r'File (\d+) tone order \(Hz\):\s*([\d,\s]+)'
    for match in re.finditer(pattern, content):
        file_num = int(match.group(1))
        freqs = [int(x.strip()) for x in match.group(2).split(',')]
        tones[f'tones{file_num}'] = freqs

    return tones

def hilbert_envelope(stimulus_data, fs=44100):
    audio = stimulus_data.astype(float)

    # Mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Envelope
    envelope = np.abs(hilbert(audio))

    # Smooth (optional)
    window_ms = 5
    window = int(fs * window_ms / 1000)
    window = max(window, 1)
    envelope = np.convolve(envelope, np.ones(window) / window, mode='same')
    return envelope

def calculate_baseline_fr(unit,
                          recording,
                          recordings_path,
                          db=None,
                          duration=2,
                          return_raster=False):
    rec_path = os.path.join(recordings_path, recording.split(' ')[0], recording)
    rec = Recording(rec_path, samplerate=30000, db=db)

    N = Neuron(recording, unit, rec=rec, db=db)
    stims = N.load
    baseline_trials, ntrials, nstimuli = N.collect_baseline(duration=duration)
    trial_duration = duration * nstimuli

    trial_frs = [
        len(spikes) / trial_duration
        for spikes in baseline_trials
    ]
    mean_fr = np.mean(trial_frs)
    std_fr = np.std(trial_frs, ddof=1)

    return mean_fr, std_fr

def calculate_response_probability(unit,
                                   recording,
                                   recordings_path,
                                   baseline_mean,
                                   baseline_std,
                                   db=None):
    rec_path = os.path.join(recordings_path, recording.split(' ')[0], recording)
    rec = Recording(rec_path, samplerate=30000, db=db)

    N = Neuron(recording, unit, rec=rec, db=db)
    stims = N.load

    R = []
    for stimulus in stims:
        _, spikes, duration = N.raster(stimulus,baseline=False,plot=False,ax=None,separate_trials=True,padding=0)
        expected_spikes = baseline_mean*duration
        threshold_spikes = expected_spikes+3*baseline_std

        total = len(spikes)
        response = 0
        for trial in spikes:
            if len(trial) > threshold_spikes:
                response += 1
        responsiveness = response/total
        R.append(responsiveness)
    return np.array(R).max()

def calculate_pcc(unit,
                  recording,
                  recordings_path,
                  db=None):
    rec_path = os.path.join(recordings_path, recording.split(' ')[0], recording)
    rec = Recording(rec_path, samplerate=30000, db=db)

    N = Neuron(recording, unit, rec=rec, db=db)
    stims = N.load

    pccs=[]
    for stimulus in stims:
        _, spikes, duration = N.raster(stimulus, baseline=False, plot=False, ax=None, separate_trials=True, padding=0)
        # Define bins (e.g., 10 ms bins)
        bin_size = 0.010  # seconds
        bins = np.arange(0, duration, bin_size)

        # Convert each trial's spike times to binned spike trains
        binned_trials = []

        for trial_spikes in spikes:
            counts, _ = np.histogram(trial_spikes, bins=bins)
            binned_trials.append(counts)

        binned_trials = np.array(binned_trials)

        # Pairwise correlation matrix (trial x trial)
        valid_trials = binned_trials[np.std(binned_trials, axis=1) > 0] #remove trials with no variance to avoid warning
        corr_matrix = np.corrcoef(valid_trials)

        # Need at least a 2D correlation matrix
        if np.ndim(corr_matrix) != 2:
            return np.nan

        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

        if upper_triangle.size == 0:
            return np.nan

        mean_trial_correlation = np.nanmean(upper_triangle)
        pccs.append(mean_trial_correlation)
    return np.array(pccs).mean()

def calculate_peak_responses(unit,
                             recording,
                             recordings_path,
                             baseline_mean,
                             baseline_std,
                             db=None):
    rec_path = os.path.join(recordings_path, recording.split(' ')[0], recording)
    rec = Recording(rec_path, samplerate=30000, db=db)
    N = Neuron(recording, unit, rec=rec, db=db)
    stims = N.load

    max_response = []
    min_response = []
    for stimulus in stims:
        _, spikes, duration = N.raster(stimulus, baseline=False, plot=False, ax=None, separate_trials=False, padding=0)
        _, [time, response] = N.psth(stimulus, spikes, plot=False, ax=None)
        response = (response-baseline_mean)/baseline_std

        # Identify peak excitation and suppression
        peak_excitation = np.max(response)
        peak_suppression = np.min(response)

        max_response.append(peak_excitation)
        min_response.append(peak_suppression)


    maximum_excitation = np.array(max_response).max()
    minimum_excitation = np.array(min_response).min()
    e_i_idx = (maximum_excitation-minimum_excitation)/(maximum_excitation+minimum_excitation)
    return maximum_excitation, minimum_excitation, e_i_idx

def calculate_tuning(unit,
                             recording,
                             recordings_path,
                             stim_library,
                             baseline_mean,
                             baseline_std,
                             db=None,
                             buffer_ms=10):
    rec_path = os.path.join(recordings_path, recording.split(' ')[0], recording)
    rec = Recording(rec_path, samplerate=30000, db=db)
    N = Neuron(recording, unit, rec=rec, db=db)
    stims = N.load

    tone_order = find_tone_order(stim_library)
    if tone_order is None:
        return None

    tone_duration = 0.4
    window = tone_duration - buffer_ms/1000
    tone_stims = [s for s in stims if 'tones' in s]
    if not tone_stims:
        return None

    tuning_dict = {}
    for stim in tone_stims:
        freqs = tone_order.get(stim, [])
        if len(freqs) == 0:
            continue

    _, trial_spikes, _ = N.raster(
        stim,
        baseline=False,
        plot=False,
        padding=0
    )
    _, _, _ = N.raster(
        stim,
        baseline=True,
        plot=False,
        padding=0
    )
    _, sep_trial_spikes, _ = N.raster(
        stim,
        baseline=False,
        plot=False,
        padding=0,
        separate_trials=True
    )
    tone_times = np.arange(len(freqs)) * tone_duration
    for freq, onset in zip(freqs, tone_times):

        offset = onset + window

        # firing rate during tone
        fr = np.sum(
            (trial_spikes >= onset) &
            (trial_spikes < offset)
        ) / window

        # z-score relative to provided baseline stats
        if baseline_std > 0:
            zscore = (fr - baseline_mean) / baseline_std
        else:
            zscore = np.nan

        # latency
        first_spikes = []

        for trial in sep_trial_spikes:

            trial = np.asarray(trial)

            mask = (
                (trial >= onset) &
                (trial < offset)
            )

            if np.any(mask):
                first_spikes.append(
                    trial[mask][0] - onset
                )

        latency = (
            np.mean(first_spikes)
            if len(first_spikes)
            else np.nan
        )

        tuning_dict[freq] = {
            "zscore": zscore,
            "firing_rate": fr,
            "latency": latency
        }
    return tuning_dict

def calculate_stimulus_correlation(unit,
                                   recording,
                                   recordings_path,
                                   stim_library,
                                   baseline_mean,
                                   baseline_std,
                                   db=None,
                                   fs=44100,
                                   psth_binsize=0.001):
    rec_path = os.path.join(recordings_path, recording.split(' ')[0], recording)
    rec = Recording(rec_path, samplerate=30000, db=db)
    N = Neuron(recording, unit, rec=rec, db=db)
    stims = N.load

    c, l = [], []
    for stimulus in stims:
        _, stimulus_time, stimulus_data = N.plot_stimulus(stimulus, plot=False, ax=None)
        envelope = hilbert_envelope(stimulus_data, fs=fs)

        _, spikes, duration = N.raster(stimulus, baseline=False, plot=False, ax=None)
        _, [psth_time, psth_data] = N.psth(stimulus, spikes, plot=False, ax=None)

        # Bin the envelope to match the PSTH
        samples_per_bin = int(fs * psth_binsize)

        n_complete = len(envelope) // samples_per_bin
        envelope_binned = envelope[:n_complete * samples_per_bin]
        envelope_binned = envelope_binned.reshape(n_complete, samples_per_bin).mean(axis=1)

        # Match lengths
        n = min(len(envelope_binned), len(psth_data))
        envelope_binned = envelope_binned[:n]
        psth_data = psth_data[:n]

        # Z-score both
        envelope_binned = zscore(envelope_binned)
        psth_data = zscore(psth_data)

        # Zero-lag correlation
        r = np.corrcoef(envelope_binned, psth_data)[0, 1]

        # Maximum cross-correlation
        xcorr = correlate(psth_data, envelope_binned, mode="full")
        xcorr /= np.sqrt(np.sum(psth_data ** 2) * np.sum(envelope_binned ** 2))

        lags = np.arange(-n + 1, n)
        best_idx = np.argmax(xcorr)

        best_r = xcorr[best_idx]
        best_lag_bins = lags[best_idx]
        best_lag_ms = best_lag_bins * psth_binsize * 1000

        l.append(best_lag_ms)
        c.append(best_r)

    return np.array(c).max(), np.min(np.array(l)[np.array(l) > 0])

def calculate_adaptation(unit, recording, recordings_path, stim_library, baseline_mean,  baseline_std, db=None):
    rec_path = os.path.join(recordings_path, recording.split(' ')[0], recording)
    rec = Recording(rec_path, samplerate=30000, db=db)
    N = Neuron(recording, unit, rec=rec, db=db)
    stims = N.load

    sounds = ['ZF A_20db', '2khz_20db']
    tempi = [180, 300]

    stim_details = json.load(open(r'C:\Users\tmerri03\PycharmProjects\Neural-Recordings-Analysis\pipeline\stimulus_details.json'))
    stim_path = rf'C:\Users\tmerri03\Desktop\RhythmStimuli\Awake Recs\Raw Files'

    slopes = []
    for sound in sounds:
        for tempo in tempi:
            stim = f'{sound}_{tempo}ms_8b_1xomit_8b_silence'
            details = stim_details.get(stim)

            sr, sound_data = wavfile.read(f'{stim_path}/{sound.lower()}.wav')
            sound_duration = sound_data.shape[0] / sr
            gap_duration = tempo/1000 - sound_duration
            omission_duration = details['omit_beats'] * tempo/1000 + gap_duration
            beat_times = np.arange(0, details['pre_beats']) * tempo/1000 + omission_duration

            _, spikes, duration = N.raster(stim, baseline=False, plot=False, ax=None)
            _, [psth_time, psth_data] = N.psth(stim, spikes, plot=False, ax=None)
            psth_z = (psth_data - baseline_mean) / baseline_std

            frs=[]
            for beat in beat_times:
                mask = (psth_time > beat) & (psth_time <= beat + duration)
                beat_psth = psth_z[mask]
                beat_fr = np.max(beat_psth)
                frs.append(beat_fr)
            slope = np.polyfit(np.arange(len(frs)), frs, 1)[0] if len(frs) >= 2 else np.nan
            slopes.append(slope)

    return np.median(slopes)

def calculate_spike_width(unit, recording, db=None):
    cols = ['spike_width_pp']
    conds = {
        'manual_isi_0_7': ('<', 1),
        'session_id': ('=', recording),
        'unit_id': ('=', unit)
    }
    neurons = db.load_neurons_from_database(select_columns= cols, conditions=conds)

    if not len(neurons) == 1:
        return None

    return neurons[0][0]

def calculate_pca_scores(pcaf, unit, recording):
    pc1_score = pcaf.get_pca_scores(unit, recording)[0]
    pc2_score = pcaf.get_pca_scores(unit, recording)[1]
    pc3_score = pcaf.get_pca_scores(unit, recording)[2]
    return pc1_score, pc2_score, pc3_score

#running functions
def analyze_neurons(
    db_path = r'C:\\Users\\tmerri03\\Desktop\\Temp Neural Files\\awake_recordings.db',
    stim_lib_path = r'R:\\Data\\tyler\\Recordings\\Stim\\Stimuli Library',
    recordings_path=r'N:\\Data\\RhythmPerception\\Neural Recordings',
    raw_sounds_path=r'C:\\Users\\tmerri03\\Desktop\\RhythmStimuli\\Awake Recs\\Raw Files\\regex',
    elements = None, #list of tuples: (neuron_id, recording_id)
    db=None, #database objset (SiNAPSE.core.Database)
):
    init_db(db_path)

    from pipeline._pca_features import pca_features
    pcaf = pca_features(db_path, stim_lib_path, recordings_path)
    pcaf.calculate_PCA_feature_scores(elements)

    for neuron_id, recording_id in elements:

        try:
            data = analyze_single_neuron(
                db_path = db_path,
                stim_lib_path = stim_lib_path,
                recordings_path = recordings_path,
                raw_sounds_path = raw_sounds_path,
                unit=neuron_id,
                recording=recording_id,
                db=db,
                pcaf=pcaf
            )
            update_SQL(db_path, data)
        except:
            print(f"Something went wrong --> skipping {recording_id} neuron {neuron_id}")
            continue
        # return

def analyze_single_neuron(
    db_path = r'C:\\Users\\tmerri03\\Desktop\\Temp Neural Files\\awake_recordings.db',
    stim_lib_path = r'R:\\Data\\tyler\\Recordings\\Stim\\Stimuli Library',
    recordings_path=r'N:\\Data\\RhythmPerception\\Neural Recordings',
    raw_sounds_path=r'C:\\Users\\tmerri03\\Desktop\\RhythmStimuli\\Awake Recs\\Raw Files\\regex',
    unit=None,
    recording=None,
    db=None, #database object (SiNAPSE.core.Database),
    pcaf=None #pca_features object (pipeline._pca_features.pca_features)
):
    print(f'Beginning analysis for recording:', recording, 'unit:', unit)
    data = {}

    # --- identifiers --- #
    data['session_id'] = recording
    data['unit_id'] = unit
    data['cluser_id'] = None #to be filled after PCA/clustering
    data['bird_id'] = recording.split(' ')[0]
    data['training_status'] = lookup_bird(data['bird_id'])

    # --- basic physiology --- #
    mean_fr, std_fr = calculate_baseline_fr(unit, recording, recordings_path, db=db)
    data['baseline_fr'] =  mean_fr

    response_prob = calculate_response_probability(unit, recording, recordings_path, mean_fr, std_fr, db=db)
    data['response_probability'] = response_prob

    pcc = calculate_pcc(unit, recording, recordings_path, db=db)
    data['mean_pairwise_trial_corr'] = pcc

    spike_width = calculate_spike_width(unit, recording, db=db)
    data['spike_width'] = spike_width

    # --- excitation / inhibition --- #
    peak_excitation, peak_inhibition, e_i_idx = calculate_peak_responses(unit, recording, recordings_path, mean_fr, std_fr, db=db)
    data['peak_excitation'] = peak_excitation
    data['peak_suppression'] = peak_inhibition
    data['excitation_inhibition_index'] = e_i_idx
    data['excitation_duration'] = None #set later?
    data['suppression_duration'] = None #set later?

    # --- frequency tuning --- #
    tuning_dict = calculate_tuning(
                        unit=unit,
                        recording=recording,
                        recordings_path=recordings_path,
                        stim_library=stim_lib_path,
                        baseline_mean=mean_fr,
                        baseline_std=std_fr,
                        db=db,
                    )
    best_freq, best_data = max(tuning_dict.items(), key=lambda x: x[1]["zscore"]);
    worst_freq, worst_data = min(tuning_dict.items(), key=lambda x: x[1]["zscore"]);
    worst_latency = worst_data["latency"]
    best_latency = best_data["latency"]

    pref_freq, pref_data = max(tuning_dict.items(), key=lambda x: abs(x[1]["zscore"]))
    data['best_frequency'] = pref_freq
    peak = abs(pref_data["zscore"])
    threshold = 0.5 * peak
    freqs = sorted(tuning_dict.keys())
    responsive = [
        f for f in freqs
        if abs(tuning_dict[f]["zscore"]) >= threshold
    ]
    bandwidth = responsive[-1] - responsive[0] if responsive else np.nan
    data['bandwidth'] = bandwidth

    #latency estimates may be best in response to preferred tones
    data['excitation_latency'] = best_latency
    data['suppression_latency'] = worst_latency

    # --- temporal coding ---
    #stimulus correlation, stimulus lag, vector strength, preferred phase
    corr, lag = calculate_stimulus_correlation(unit=unit,
                                                recording=recording,
                                                recordings_path=recordings_path,
                                                stim_library=stim_lib_path,
                                                baseline_mean=mean_fr,
                                                baseline_std=std_fr,
                                                db=db)
    data['stimulus_correlation'] = corr
    data['stimulus_lag'] = lag

    # returns the median slope of the first 8 beats for 2khz and sound a regular stim
    adaptation = calculate_adaptation(
        unit=unit,
        recording=recording,
        recordings_path=recordings_path,
        stim_library=stim_lib_path,
        baseline_mean=mean_fr,
        baseline_std=std_fr,
        db=db
    )
    data['adaptation_slope'] = adaptation

    # --- response PCA scores ---
    pc1_score, pc2_score, pc3_score = calculate_pca_scores(pcaf, unit, recording)
    data['psth_pc1'] = pc1_score
    data['psth_pc2'] = pc2_score
    data['psth_pc3'] = pc3_score

    # --- features not yet implemented: explicit None placeholders so SQL/clustering still work --- #
    data['brain_region'] = 'L3/NC'
    data['response_centroid'] = None
    data['response_skewness'] = None
    data['num_response_peaks'] = None
    data['adaptation_index'] = None
    data['steady_state_response'] = None
    data['recovery_after_silence'] = None
    data['vector_strength'] = None
    data['preferred_phase'] = None
    data['natural_sound_preference'] = None


    # import pprint
    # pprint.pprint(data)

    return data
