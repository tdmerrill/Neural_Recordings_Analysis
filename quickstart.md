# Installation

### Anaconda (virtual environment manager)
It is highly recommended to use Anaconda to manage your Python environment.
You can [download Anaconda here](https://www.anaconda.com/products/distribution).
Miniconda is acceptable, and  can be installed directly from the PyCharm application in `Settings > Python > Interpreter`

Once Installed, open the Anaconda Prompt and create a new environment with Python 3.12:
```bash
conda create -n sinapse python=3.12
conda activate sinapse
```


### SiNAPSE
SiNAPSE is a python library built on top of SpikeInterface to sort, maintain, and visualize neural data.
You can find the source code for [SiNAPSE on GitHub](https://github.com/tdmerrill/SiNAPSE/tree/master)

To install:
```bash
pip install SiNAPSE
```
Sorting with a GPU is recommended, though not technically required. If you have a GPU, you will need to install the following dependencies:
```bash
pip uninstall torch
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```
If you do not have a GPU, sorting performance will be much slower, and you may want to make adjustments to the batch
size during sorting to compensate.

### Analysis Tools
If you'd like to use any additional analysis tools I've written to make plotting or further analysis easier,
you can find the [Neural_Recordings_Analysis repository on GitHub](https://github.com/tdmerrill/Neural_Recordings_Analysis)

These tools are not required, but if you want to use them you can clone the repo to access the code:
```bash
git clone https://github.com/tdmerrill/Neural-Recordings-Analysis
```

# Using SiNAPSE
Make sure to write your code in the virtual environment you created above.

There are 3 main objects in SiNAPSE: `Database`, `Recording`, and `Neuron`. 
Each of these objects has a set of methods that can be used to manipulate and analyze the data.

### Database
The `Database` object is the main entry point for interacting with your data. It is used to
create, load, and iteract with a SQL database that contains all of your neurons along with their metadata.

You can initialize a new database with the following code:
```python
from sinapse.core import Database

path_to_database = "path/to/database.db"
path_to_stimuli = "path/to/stimuli/directory"
db = Database(path_to_database, path_to_stimuli)
```

Now we have a Database object that we can use access our data. The database.db file will be created
at the location specified by `path_to_database`. The `path_to_stimuli` argument is the path to a 
directory containing any stimuli files that you played during your recordings (.wav files).

There is really only one method to know about the Database object:
- `load_neurons_from_database(list: select_columns, dict: conditions)`: Returns a list of units that meet the conditions you specify.

Also, there are two properties that may be helpful. You can return the  property of the object by calling it like a function, but without the parentheses (see example below).
- `recordings`: Returns a list of all recordings stored in the database.
- `metrics`: Returns a list of all types of data stored for each neuron (columns in the database).

Example usage:
```python
# Load all neurons that 
#  1. have a <1% ISI violations with a refractory period of 0.7ms 
#  2. are from a particular recording

select_columns = ['unit_id', 'session_id']
conditions = {
    'manual_isi_0_7': ('<', 1),
    'session_id': ('=', recording1),
}
units = db.load_neurons_from_database(select_columns, conditions)
```

Now we may loop through the neurons and do something for each:
```python
for unit_id, session_id in units:
    # Do something with each neuron
```

### Recording
The `Recording` object is used to represent a single recording session. 
It contains mainly internal methods. Importantly you can use the `Recording` object 
sort your data via SpikeInterface (kilsort4 algorithm):

```python
from SiNAPSE.core import Recording

path_to_recording = "path/to/recording"
rec = Recording(path_to_recording, samplerate=30000, db=db) #OpenEphys uses a default samplerate of 30kHz; you must pass a reference to the Database object you created above.

# Sorting the recording must be done on the local drive, so the data is first copied over.
#   You can specify the local path to copy the data to, or it will default to the home directory.
local_path = "path/to/local/"
rec.sort(local_path=local_path)
```

Running`rec.sort()` will initiate the sorting process. The sorted data will be copied back to your recording folder.
After automatical spike sorting, you must manually curate the data using the SpikeInterface GUI, which will launch
automatically. This is very similar to another tool many labs use called [Phy](https://github.com/cortex-lab/phy).

In particular, you will want to look for waveforms that look like real action potentials, and remove any that look like noise. 
You can also merge units that are likely to be the same neuron, and split units that are likely to be multiple neurons.

### Neuron
The `Neuron` object is used to represent a single neuron. 
It contains methods that can help wtih accessing data like spike times and waveform shape, as well
as stimulus information. You can create the `Neuron` object by passing in the `unit_id` and `session_id` of the neuron you want to access:
```python
from SiNAPSE.core import Neuron

N = Neuron(session_id, unit_id, rec=rec, db=db) #you must pass references to your Recording and Database objects to the Neuron object so it can access the data.
```

Much of the functionality of the Neuron class is internal or for a specific use case, but there are two main visulization methods to be aware of:
- `plot_waveform()`: Plots the average waveform of the neuron.
- `plot()`: Plots a raster plot and PSTH.
    - `list: stimuli_to_plot`: A list of stimuli to plot. Plots all by default.
    - `float: padding`: The amount of time to pad before and after the stimulus in seconds. Default is 0.5s.'
    - `float: psth_sigma`: The size of the smoothing gaussian for the PSTH. Default is 15ms.
    - `float: psth_dt`: The bin size for the PSTH. Default is 1ms.
    - `bool: baseline`: Include a plot for the baseline activity in a window precending the stimulus. Default is False.
    - `bool: raw_data`: Plot the raw data for one trial. Default is False.

Here's an example of how I might generate and save plots for all stimuli for one neuron:
```python
stims_to_plot = ['stim1', 'stim2', 'stim3']
N.plot(stimuli_to_raster=stims_to_plot, baseline=True) #if you are generating and saving many plots, use show=False. They will still save as files.

save_path = "path/to/save/plots"
N.save_plots(output_path=save_path, name_prefix=f'{session_id}_{unit_id}_', clear_cache=True) #clear_cache will clear the plots from memory after saving, which is useful if you are generating a lot of plots.
```

# Example
Here is an example of how you might use SiNAPSE to plot figures for every recording in your recordings folder.
```python
from SiNAPSE.core import Database, Recording, Neuron

import os #for file management
import numpy as np #for arrays and numerical operations
import matplotlib.pyplot as plt #for plotting

RECORDINGS_FOLDER = "path/to/recordings"
DATABSE_PATH = "path/to/database.db"
STIMULUS_LIBRARY_FOLDER = "path/to/stimuli"

db = Database(DATABSE_PATH, STIMULUS_LIBRARY_FOLDER)
for recording_folder in os.listdir(RECORDINGS_FOLDER):
    recording_path = os.path.join(RECORDINGS_FOLDER, recording_folder)
    rec = Recording(recording_path, samplerate=30000, db=db)

    # Sort the recording if it hasn't been sorted yet
    if not rec.is_sorted():
        rec.sort(local_path="path/to/local/")

    # Load all single units from the database for this recording
    select_columns = ['unit_id', 'session_id']
    conditions = {
        'manual_isi_0_7': ('<', 1),
        'session_id': ('=', recording_folder),
    }
    units = db.load_neurons_from_database(select_columns, conditions)

    for unit_id, session_id in units:
        N = Neuron(session_id, unit_id, rec=rec, db=db)
        N.plot(baseline=True)
        save_path = os.path.join("path/to/save/plots", f"{session_id}_{unit_id}_plots")
        N.save_plots(output_path=save_path, clear_cache=True)
```

# Using Neural_Recordings_Analysis
SiNAPSE provides the main framework for interacting with your data, but the Neural_Recordings_Analysis repo 
provides a set of tools for analyzing and visualizing your data. Here are some examples of what you can do:

1. Evaluate the stability of your recordings by calculating drift in the baseline activity.
    -You may choose to take a subset of your trials based on this metric.
2. Generate summary plots for all neurons in a recording
   - Each stimulus will have a raster plot and PSTH
   - PSTH plots are z-scored to the baseline actity such that the y-axis is in units of standard deviations from the baseline.
3. Plot a waveform summary for each recording.
4. Plot the location of each unit on the probe for each recording.

```python
RECORDINGS_PATH = r'R:\Data\RhythmPerception\Neural Recordings\Recordings'
DB_PATH = r'C:\Users\tmerri03\Desktop\Temp Neural Files\awake_recordings.db'
STIM_LIB_PATH = r'R:\Data\tyler\Recordings\Stim\Stimuli Library'

from Neural_Recordings_Analysis._generate_database import *
db = generate_database(DB_PATH, STIM_LIB_PATH)
recordings = sorted(db.recordings, reverse=True)

from Neural_Recordings_Analysis._neurons import Response
for recording in recordings:
    R = Response(STIM_LIB_PATH, RECORDINGS_PATH, recording, db=db, output=r'C:\Users\tmerri03\Desktop\Neural Data\Awake Plots')
    R.evaluate_neurons()
    R.summary
    R.waveforms
    R.probe
```
You may specify an output path for the plots with the `output` parameter. If you don't specify an output path,
the plots will be saved to your home directory in `C:\Users\<username>\.Neural_Recordings_Analysis\outputs`.
```python

# Citations
SiNAPSE and other tools heavily use code written by other labs and developers.
Please cite the following sources if you use SiNAPSE:

[Kilsort4](https://github.com/mouseland/kilosort)

[SpikeInterface](https://github.com/SpikeInterface/spikeinterface)

[ProbeInterface](https://github.com/SpikeInterface/probeinterface)