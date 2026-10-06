def analyze_neuron(
        db_path = r'C:\\Users\\tmerri03\\Desktop\\Temp Neural Files\\awake_recordings.db',
        stim_lib_path = r'R:\\Data\\tyler\\Recordings\\Stim\\Stimuli Library',
        recordings_path=r'N:\\Data\\RhythmPerception\\Neural Recordings',
        raw_sounds_path=r'C:\\Users\\tmerri03\\Desktop\\RhythmStimuli\\Awake Recs\\Raw Files\\regex',
        unit=None,
        recording=None,
        stim_subset=None
):

    print('Beginning analysis for recording:', recording, 'unit:', unit)

    from polar_analysis import NeuronOmissionAnalysis
    from pathlib import Path

    if recording is None or unit is None:
        print('specify a recording path and unit number')
        return

    if stim_subset is None:
        stim_subset = [
            'ZF A_20db_180ms_8b_1xomit_8b_silence',
            'ZF A_20db_300ms_8b_1xomit_8b_silence',
            '2khz_20db_180ms_8b_1xomit_8b_silence',
            '2khz_20db_300ms_8b_1xomit_8b_silence',
        ]

    analysis = NeuronOmissionAnalysis(
        db_path=db_path,
        stim_lib_path=stim_lib_path,
        recordings_path=recordings_path,
        raw_sounds_path=raw_sounds_path,
        recording=recording,
        unit=unit,
        stim_subset=stim_subset,
    )
    analysis.run()

    for stimulus in stim_subset:
        results = analysis.phase_comparison.get(stimulus)

        if results is None:
            continue

        results['session_id'] = Path(recording).name
        results['unit_id'] = unit
        results['stimulus'] = stimulus
        data = reformat_data(results)

        update_SQL(db_path, data)

        print(f'    Updated database for recording: {recording}, unit: {unit}, stimulus: {stimulus}')
    print(f'    Finished analysis for recording: {recording}, unit: {unit}')
    return analysis

def reformat_data(results):
    data={}

    data['omission_phase_ms'] = None
    data['omission_vs'] = None
    data['decay_phase_ms'] = None
    data['decay_vs'] = None
    data['beat_phase_ms'] = None
    data['beat_vs'] = None
    data['baseline_vs'] = None
    data['baseline_phase_ms'] = None

    if 'omission' in results.keys():
        data['omission_phase_ms'] = results['omission']['omission_mean_phase_ms']
        data['omission_vs'] = results['omission']['omission_vs']

        data['beat_phase_ms'] = results['omission']['beat_mean_phase_ms']
        data['beat_vs'] = results['omission']['beat_vs']

    if 'decay' in results.keys():
        data['decay_vs'] = results['decay']['decay_vs']
        data['decay_phase_ms'] = results['decay']['decay_mean_phase_ms']

        data['beat_phase_ms'] = results['decay']['beat_mean_phase_ms']
        data['beat_vs'] = results['decay']['beat_vs']
        data['tempo_ms'] = results['decay']['tempo_ms']

    if 'baseline' in results.keys():
        data['baseline_phase_ms'] = results['baseline']['baseline_mean_phase_ms']
        data['baseline_vs'] = results['baseline']['baseline_vs']

        data['beat_vs'] = results['baseline']['beat_vs']
        data['beat_phase_ms'] = results['baseline']['beat_mean_phase_ms']
        data['tempo_ms'] = results['baseline']['tempo_ms']

    data['session_id'] = results['session_id']
    data['unit_id'] = results['unit_id']
    data['stimulus'] = results['stimulus']

    try:
        data['candidate'] = int(candidacy(data))
    except Exception as e:
        print(f'    ❌️ Error determining candidate status for recording: {data['session_id']}, unit: {data['unit_id']}')
        print(e)
        data['candidate'] = None

    return data

def candidacy(data, factor=2):
    import numpy as np

    beat_phase = data['beat_phase_ms']
    beat_vs = data['beat_vs']
    baseline_vs = data['baseline_vs']
    tempo_ms = data['tempo_ms']

    # Must have stronger phase locking during the beat than baseline
    if beat_vs <= factor*baseline_vs:
        return False



    for label in ('omission', 'decay'):

        phase = data[f'{label}_phase_ms']
        vs = data[f'{label}_vs']

        # skip missing values
        if phase is None or vs is None:
            continue

        if vs <= factor*baseline_vs:
            return False

        # circular difference in ms
        diff_ms = ((phase - beat_phase + tempo_ms / 2)
                   % tempo_ms) - tempo_ms / 2

        # convert to degrees
        diff_deg = abs(diff_ms / tempo_ms * 360)

        if diff_deg <= 45 and vs >= 0.5 * beat_vs:
            return True

    return False

def init_db(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
                CREATE TABLE IF NOT EXISTS omission_analysis
                (
                    session_id
                    TEXT,
                    unit_id
                    INTEGER,
                    stimulus
                    TEXT,
                    beat_phase_ms
                    REAL,
                    decay_phase_ms
                    REAL,
                    omission_phase_ms
                    REAL,
                    beat_vs
                    REAL,
                    decay_vs
                    REAL,
                    omission_vs
                    REAL,
                    candidate
                    INTEGER
                )
                """)

    cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_omission
                    ON omission_analysis(session_id, unit_id, stimulus)
                """)

    conn.commit()
    conn.close()

def update_SQL(db_path, data):
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO omission_analysis (session_id,
                                       unit_id,
                                       stimulus,
                                       beat_phase_ms,
                                       decay_phase_ms,
                                       omission_phase_ms,
                                       beat_vs,
                                       decay_vs,
                                       omission_vs,
                                       candidate
                                       )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(session_id, unit_id, stimulus)
        DO
        UPDATE SET
            beat_phase_ms = excluded.beat_phase_ms,
            decay_phase_ms = excluded.decay_phase_ms,
            omission_phase_ms = excluded.omission_phase_ms,
            beat_vs = excluded.beat_vs,
            decay_vs = excluded.decay_vs,
            omission_vs = excluded.omission_vs,
            candidate = excluded.candidate
        """,
        (
            data['session_id'],
            data['unit_id'],
            data['stimulus'],
            data['beat_phase_ms'],
            data['decay_phase_ms'],
            data['omission_phase_ms'],
            data['beat_vs'],
            data['decay_vs'],
            data['omission_vs'],
            data['candidate'],
        ),
    )
    conn.commit()
    conn.close()

def run(recording, unit, db_path, stim_lib_path, recordings_path):
    init_db(db_path)
    return analyze_neuron(
        db_path = db_path,
        stim_lib_path = stim_lib_path,
        recordings_path = recordings_path,
        recording = recording,
        unit = unit,
    )