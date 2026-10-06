# globals
RECORDINGS_PATH = r'R:\Data\RhythmPerception\Neural Recordings\Recordings'
DB_PATH = r'C:\Users\tmerri03\Desktop\Temp Neural Files\awake_recordings.db'
STIM_LIB_PATH = r'R:\Data\tyler\Recordings\Stim\Stimuli Library'

#database
from _generate_database import *

db = generate_database(DB_PATH, STIM_LIB_PATH)
select_columns=['unit_id', 'session_id']
conditions = {
    'manual_isi_0_7': ('<', 1)
}
neurons = db.load_neurons_from_database(select_columns=select_columns, conditions=conditions,)
#%%
from pipeline.cluster_responses import analyze_neurons
analyze_neurons(
    db_path = DB_PATH,
    stim_lib_path = STIM_LIB_PATH,
    recordings_path = RECORDINGS_PATH,
    elements = neurons,
    db=db
)