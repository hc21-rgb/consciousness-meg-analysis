"""
Preprocessing and Epoching
"""

import os
import numpy as np
import pandas as pd
import gc
import re
import mne
from pathlib import Path
from collections import defaultdict
from mne import concatenate_epochs


############# needs to be specified ##########

# participant/participant ID
subject = 11

#############################################

# analysis duration prior to awakening
Duration = 20

# ica method
ica_method = 'infomax'

# initialize lists
epochs_list = []
data_list = []
metadata_list = []

# read metadata
file_path = "/Users/Data/Dream/Metadata.xlsx"
df = pd.read_excel(file_path)
exp_days = sorted(map(int, df[df['Participant ID'] == subject]['Exp Day'].dropna().unique()))

for ExpDay in exp_days:
    ExpDates = df.loc[
        (df['Participant ID'] == subject) & (df['Exp Day'] == ExpDay),
        'Exp date'
    ]
    ExpDate = ExpDates.iloc[0].strftime("%y%m%d")

    fifdir = Path.home() / "Data" / "Dream" / "MEG" / ExpDate
    fif_files = list(fifdir.rglob("*tsss.fif"))
    fif_files = [f.name for f in fif_files]

    def take_runs(fif_files):
        run_groups = defaultdict(list)
        for fname in fif_files:
            match = re.match(r'^run(\d*)?_?raw', fname)
            group_name = f'run{match.group(1)}' if match and match.group(1) else 'run0'
            run_groups[group_name].append(fname)

        for key in run_groups:
            files = run_groups[key]
            main_idx = [i for i, f in enumerate(files) if not re.search(r'-\d+_tsss\.fif$', f)]
            other_idx = list(set(range(len(files))) - set(main_idx))
            sorted_files = []
            if main_idx:
                sorted_files.append(files[main_idx[0]])
            if other_idx:
                other_files = [files[i] for i in other_idx]
                nums = [int(re.search(r'-(\d+)', f).group(1)) if re.search(r'-(\d+)', f) else float('inf') for f in other_files]
                sorted_files.extend([f for _, f in sorted(zip(nums, other_files))])
            run_groups[key] = sorted_files

        sorted_keys = sorted(run_groups.keys(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
        return {key: run_groups[key] for key in sorted_keys}

    runs = take_runs(fif_files)
    run_files = list(runs.values())

    for i in range(len(run_files)):
        run_id = i + 1

        df_run = df[
            (df['Participant ID'] == subject) &
            (df['Exp Day'] == ExpDay) &
            (df['run'] == run_id) &
            (df['exclusion_flag'] != 1)
        ]

        if df_run.empty:
            print(f"Skipping ExpDay {ExpDay}, run {run_id}")
            continue

        def time_str_to_seconds(t):
            parts = list(map(int, str(t).split(';')))
            return sum([a * b for a, b in zip([3600, 60, 1][-len(parts):], parts)])

        time_event = [time_str_to_seconds(t) for t in df_run['Awake Time_adj (to be used for analysis)']]
        asleep_dream = df_run['Asleep and dreaming'].tolist()
        asleep_whitedream = df_run['Dreaming but cannot recall'].tolist()
        unsure_dream = df_run['Unsure if dreaming'].tolist()
        asleep_nodream = df_run['Asleep with no dreams'].tolist()
        unsure_asleep = df_run['Unsure if asleep'].tolist()
        wake_thinking = df_run['Not asleep_thinking about something'].tolist()
        wake_nothinking = df_run['Not asleep_not thinking about anything'].tolist()
        sleepstage = df_run['U-Sleep_last30s'].tolist()
        

        file_list = run_files[i]
        for f in file_list:
            print(f)
        raw_list = [mne.io.read_raw_fif(str(fifdir / fname), preload=True, on_split_missing="ignore") for fname in file_list]
        for raw in raw_list:
            raw.set_meas_date(None)
        raw_crun = mne.concatenate_raws(raw_list)
                        
        # set time to 0
        raw_data = raw_crun.get_data()
        info = raw_crun.info.copy()
        raw_crun_zeroed = mne.io.RawArray(raw_data, info)
        
        # notch filter
        raw_crun_zeroed.notch_filter(freqs=np.arange(60, 241, 60))
         
        # bandpath filter
        raw_crun_zeroed.filter(l_freq=0.3, h_freq=50)

        # downsampling
        raw_crun_zeroed.resample(sfreq=500, npad="auto")
         
        del raw, raw_crun, raw_data
        gc.collect()
         
        # bad channel
        print("Bad channels:", raw_crun_zeroed.info['bads'])

        ### ICA
        ica = mne.preprocessing.ICA(n_components=20, method=ica_method, random_state=97)
        ica.fit(raw_crun_zeroed, picks='meg')
            
        # ICA coponent scores
        ecg_inds_ECG_cpts, scores_ECG_ctps = ica.find_bads_ecg(raw_crun_zeroed, method='ctps')
        ecg_inds_ECG_corr, scores_ECG_corr = ica.find_bads_ecg(raw_crun_zeroed, method='correlation')
        ecg_inds_EOGL, scores_EOGL = ica.find_bads_eog(raw_crun_zeroed, ch_name='EEG020', measure='correlation')
        ecg_inds_EOGR, scores_EOGR = ica.find_bads_eog(raw_crun_zeroed, ch_name='EEG021', measure='correlation')
        ecg_inds_muscle, scores_muscle = ica.find_bads_muscle(raw_crun_zeroed, threshold=0.5, start=None, stop=None, l_freq=7, h_freq=45, sphere=None, verbose=None)

        # for auto ICA exclusion
        ECG_ctps = np.abs(np.array(scores_ECG_ctps))
        ECG_ctps_index = np.argmax(ECG_ctps)

        ECG_corr = np.abs(np.array(scores_ECG_corr))
        ECG_corr_index = np.where(ECG_corr >= 0.4)[0]

        EOGL_corr = np.abs(np.array(scores_EOGL))
        EOGL_corr_index = np.where(EOGL_corr >= 0.4)[0]

        EOGR_corr = np.abs(np.array(scores_EOGR))
        EOGR_corr_index = np.where(EOGR_corr >= 0.4)[0]

        muscle_s = np.abs(np.array(scores_muscle))
        muscle_index = np.where(muscle_s >= 2.5)[0]
        if len(muscle_index) > 0:
            muscle_max_index = np.argmax(muscle_s)
        else:
            muscle_max_index = []  

        all_indices = np.concatenate([
            [ECG_ctps_index],       
            ECG_corr_index,
            EOGL_corr_index,
            EOGR_corr_index,
            muscle_max_index,
        ])

        ica_exclusion_comps = np.unique(all_indices)
        print("ICA exclusion compnent:", ica_exclusion_comps)

        ica.exclude = ica_exclusion_comps #automatic exclusion
        reconst_raw = raw_crun_zeroed.copy()# ica.apply() changes the Raw object in-place, so let's make a copy first:
        ica.apply(reconst_raw)
        
        sfreq = reconst_raw.info['sfreq']
        valid_sec = [s for s in time_event if (s - Duration) >= reconst_raw.first_time and s <= reconst_raw.times[-1]]
        if not valid_sec:
            continue        
        
        event_samples = [int(s * sfreq) for s in valid_sec]
        events = np.array([[samp, 0, 1] for samp in event_samples], dtype=int)

        # take metadadta
        metadata = pd.DataFrame({
            'participant': [subject] * len(event_samples),
            'ExpDay': [ExpDay] * len(event_samples),
            'run': [run_id] * len(event_samples),
            'time': time_event[:len(event_samples)],
            'dream': asleep_dream[:len(event_samples)],
            'whitedream': asleep_whitedream[:len(event_samples)],
            'unsure_dream': unsure_dream[:len(event_samples)],
            'asleep_nodream': asleep_nodream[:len(event_samples)],
            'unsure_asleep': unsure_asleep[:len(event_samples)],
            'wake_thinking': wake_thinking[:len(event_samples)],
            'wake_nothinking': wake_nothinking[:len(event_samples)],
            'sleepstage': sleepstage[:len(event_samples)]            
        })

        epochs = mne.Epochs(
            raw=reconst_raw,
            events=events,
            event_id=1,
            tmin=-Duration,
            tmax=-0,
            baseline=None,
            preload=True,
            reject_by_annotation=False,
            metadata=metadata
        )

        epochs_list.append(epochs)
        data_list.append(epochs.get_data())
        metadata_list.append(metadata)

        del raw_crun_zeroed, reconst_raw
        gc.collect()


# collect across runs and days
epochs_all = concatenate_epochs(epochs_list, on_mismatch="ignore")

# save epoch
save_dir = "/Users/Data/Dream/MNE/Epoch"
os.makedirs(save_dir, exist_ok=True)
outfilename = os.path.join(save_dir, f"Dream{subject}_Notch_BP_500Hz_autoICA_{Duration}s_epo.fif")    
epochs_all.save(outfilename, overwrite=True)

print(f"Total epochs: {len(epochs_all)}")