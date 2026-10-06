DB_PATH = r"C:\Users\tmerri03\Desktop\Temp Neural Files\awake_recordings.db"
STIMULUS_LIBRARY = r'R:\Data\tyler\Recordings\Stim\Stimuli Library'
RECORDINGS_PATH = r'N:\Data\RhythmPerception\Neural Recordings'

# --- interface with database ---
def write_to_db(db_path, recording, unit, pcc, depth):
    import sqlite3

    print(f'Writing to recording: {recording} and unit: {unit}')
    print(f'    depth: {depth}, pcc: {pcc}')
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "UPDATE neurons SET pcc = ? WHERE session_id = ? AND unit_id = ?",
        (pcc, recording, unit),
    )
    cur.execute(
        "UPDATE neurons SET depth = ? WHERE session_id = ? AND unit_id = ?",
        (depth, recording, unit),
    )
    conn.commit()
    conn.close()

def load_features(db_path, session_id=None, columns=None):
    """Reads selected columns from neuron_features into a DataFrame."""
    import sqlite3
    import pandas as pd
    conn = sqlite3.connect(db_path)

    if columns is None:
        columns = ["*"]
    else:
        columns = ", ".join(columns)

    query = f"SELECT {columns} FROM neurons"

    if session_id is not None:
        query += " WHERE session_id = ?"
        df = pd.read_sql_query(
            query,
            conn,
            params=(session_id,)
        )
    else:
        df = pd.read_sql_query(query, conn)

    conn.close()

    return df

# --- calcuations for metrics ---
def calculate_depth(y_coord, recording_name):
    import re
    match = re.search(r'(\d+(?:\.\d+)?)\s*um\b', recording_name)
    return (float(match.group(1)) - y_coord) if match else None

def calculate_pcc(stimulus, N=None):
    pcc = N.compute_pcc(stimulus, padding=0)
    return pcc if pcc is not None else None

# --- visualization ---
def plot_depth_pcc_distribution(db_path):
    df = load_features(db_path, columns=['session_id', 'unit_id', 'pcc', 'depth'])

    import matplotlib.pyplot as plt
    import numpy as np

    plot_df = df[['depth', 'pcc']].dropna()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        plot_df['depth'],
        plot_df['pcc'],
        alpha=0.7,
        s=45
    )

    ax.set_xlabel('Depth (µm)')
    ax.set_ylabel('PCC')
    ax.set_title('PCC vs. Recording Depth')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # #regression line
    # x = plot_df['depth'].to_numpy()
    # y = plot_df['pcc'].to_numpy()
    #
    # m, b = np.polyfit(x, y, 1)
    # x_fit = np.linspace(x.min(), x.max(), 100)
    #
    # ax.plot(
    #     x_fit,
    #     m * x_fit + b,
    #     linewidth=2
    # )

    #invert axis
    ax.invert_xaxis()

    fig.tight_layout()
    plt.show()

def main():
    from SiNAPSE.core import Database, Recording, Neuron
    import os

    db = Database(DB_PATH, STIMULUS_LIBRARY)
    recordings = db.recordings
    for recording in recordings:
        select_columns = ['unit_id', 'session_id', 'unit_loc_y', 'pcc', 'depth']
        conditions = {
            'manual_isi_0_7': ('<', 1),
            'session_id': ('=', recording),
        }
        units = db.load_neurons_from_database(select_columns, conditions, table='neurons')

        rec = Recording(os.path.join(RECORDINGS_PATH, recording.split(' ')[0], recording), samplerate=30000, db=db)
        for unit, _, y_coord, pcc, depth in units:

            if pcc is not None and depth is not None:
                continue

            N = Neuron(recording, unit, db=db, rec=rec)
            stims = N.load

            depth, pcc = None, None
            if 'ZF A_20db_180ms_8b_1xomit_8b_silence' in stims:
                depth = calculate_depth(y_coord, recording)
                pcc = calculate_pcc('ZF A_20db_180ms_8b_1xomit_8b_silence', N=N)
            write_to_db(DB_PATH, recording, unit, pcc, depth)

    plot_depth_pcc_distribution(DB_PATH)

if __name__ == '__main__':
    main()
