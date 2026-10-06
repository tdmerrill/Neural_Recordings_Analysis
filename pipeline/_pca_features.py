from SiNAPSE.core import Database, Recording, Neuron

import os
import numpy as np

class pca_features():
    def __init__(self, db_path, stim_lib_path, recordings_path):
        self.db = Database(db_path, stim_lib_path)
        self.recordings_path = recordings_path

    def baseline_fr(self, elements, duration=2):
        baselines = {}
        for unit, recording in elements:
            if recording not in baselines.keys():
                baselines[recording] = {}
            if unit not in baselines[recording].keys():
                baselines[recording][unit] = {}

            bird = recording.split(' ')[0]
            recording_path = os.path.join(self.recordings_path, bird, recording)
            rec = Recording(recording_path, samplerate=30000, db=self.db)
            N = Neuron(recording, unit, rec=rec, db=self.db)
            stims = N.load
            baseline_trials, ntrials, nstimuli = N.collect_baseline(duration=duration)
            trial_duration = duration * nstimuli

            trial_frs = [
                len(spikes) / trial_duration
                for spikes in baseline_trials
            ]
            mean_fr = np.mean(trial_frs)
            std_fr = np.std(trial_frs, ddof=1)

            baselines[recording][unit]['mean'] = mean_fr
            baselines[recording][unit]['std'] = std_fr

        return baselines

    def find_overlapping_stim(self, elements):
        import os

        overlapping_stims = None

        for neuron_id, recording_id in elements:
            bird = recording_id.split(' ')[0]
            recording_path = os.path.join(self.recordings_path, bird, recording_id)
            rec = Recording(recording_path, samplerate=30000, db=self.db)
            N = Neuron(recording_id, neuron_id, db=self.db, rec=rec)
            stims = set(N.load)

            if overlapping_stims is None:
                overlapping_stims = stims
            else:
                overlapping_stims &= stims

            # Early exit if there are no common stimuli
            if not overlapping_stims:
                break

        return sorted(overlapping_stims) if overlapping_stims else []

    def calculate_PCA_feature_scores(self, elements):
        print("Building feature matrix:")

        cache_dir = os.path.join(os.path.dirname(__file__), ".pca_cache")
        os.makedirs(cache_dir, exist_ok=True)

        save_path = os.path.join(cache_dir, "neuron_feature_matrix.npz")

        if os.path.exists(save_path):
            print("     Loading cached feature matrix...")
            data = np.load(save_path, allow_pickle=True)

            X = data["X"]
            overlapping_stims = data["stimuli"].tolist()
            neuron_ids = data["neuron_ids"]
            session_ids = data["session_ids"]

        else:
            overlapping_stims = self.find_overlapping_stim(elements)
            print(f'    Found {len(overlapping_stims)} overlapping stimuli.')
            baselines = self.baseline_fr(elements, duration=2)
            print(f'    Calculated baseline firing rates for {len(baselines)} recordings.')


            X = []
            neuron_ids = []
            session_ids = []

            print(f'    There are {len(elements)} neurons. This may take a while to compute...')
            for neuron_id, recording_id in elements:
                bird = recording_id.split(' ')[0]
                recording_path = os.path.join(self.recordings_path, bird, recording_id)
                rec = Recording(recording_path, samplerate=30000, db=self.db)
                N = Neuron(recording_id, neuron_id, db=self.db, rec=rec)
                N.load

                mean = baselines[recording_id][neuron_id]["mean"]
                std = baselines[recording_id][neuron_id]["std"]
                if not np.isfinite(std) or std <= 1e-10:
                    print(f"Skipping {recording_id} unit {neuron_id}")
                    continue

                neuron_vector = []
                for stimulus in overlapping_stims:
                    _, spikes, duration = N.raster(stimulus, baseline=False, plot=False, padding=0)
                    _, [psth_time, psth_data] = N.psth(stimulus, spikes, plot=False, padding=0)
                    psth_z = (psth_data - mean) / std
                    neuron_vector.extend(psth_z)

                    # break #single stimulus for testing --- should always be the same

                X.append(neuron_vector)
                neuron_ids.append(neuron_id)
                session_ids.append(recording_id)

            X = np.asarray(X)

            np.savez_compressed(
                save_path,
                X=X,
                stimuli=np.array(overlapping_stims, dtype=object),
                neuron_ids=np.array(neuron_ids),
                session_ids=np.array(session_ids)
            )

            print(f'    Finished creating PCA substrate with shape {X.shape}')

        print(f'Running PCA:')

        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        Xz = StandardScaler().fit_transform(X)

        pca = PCA(n_components=20)
        scores = pca.fit_transform(Xz)

        self.pca_scores = {}

        for recording_id, neuron_id, score in zip(session_ids, neuron_ids, scores):
            if recording_id not in self.pca_scores:
                self.pca_scores[recording_id] = {}
            self.pca_scores[recording_id][neuron_id] = score

    def get_pca_scores(self, unit, recording):
        return self.pca_scores[recording][unit]