"""
Make noise covariance
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
import gc


############# needs to be specified ##########

# participant/participant ID
subject = 11

#############################################

file_path = "/Users/Data/Dream/Metadata.xlsx"

df = pd.read_excel(file_path)
exp_days = sorted(map(int, df[df['Participant ID'] == subject]['Exp Day'].dropna().unique()))


ExpDates = []
for ExpDay in exp_days:   
    expdate_series = df.loc[
        (df['Participant ID'] == subject) & (df['Exp Day'] == ExpDay),
        'Exp date'
    ]
    if not expdate_series.empty:
        ExpDate = expdate_series.iloc[0].strftime("%y%m%d")
        ExpDates.append(ExpDate)
    else:
        print(f"[!] Exp date not found for Exp Day {ExpDay}")


### take fif files from runs in order 
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


### compute noise covariance from the fif file of the first run for each experimental day
for ind in exp_days:
    ind = ind-1
    day_dir = Path.home() / "Data" / "Dream" / "MEG" / ExpDates[ind]
    fif_files = list(day_dir.rglob("*tsss.fif"))
    fif_files = [f.name for f in fif_files]
    
    runs = take_runs(fif_files)
    run_files = list(runs.values())
    
    file_list = run_files[0]
    first_fif = file_list[0]   
    fif_dir = Path.home() / "Data" / "Dream" / "MEG" / ExpDates[ind] / first_fif
    raw = mne.io.read_raw_fif(fif_dir, preload=True, on_split_missing='ignore')
    
    # crop raw file
    raw_crop = raw.copy().crop(tmin=0, tmax=120)
        
    ## make noise covariance
    # notch filter
    raw_crop.notch_filter(freqs=np.arange(60, 241, 60))
     
    # bandpath filter
    raw_crop.filter(l_freq=0.3, h_freq=50)
    
    ## downsampling
    raw_crop.resample(sfreq=500, npad="auto")
    
    # extract silent period more than 60 sec
    quiet_segments = []
    min_duration = 60.0  # >60
    
    onsets = raw_crop.annotations.onset
    durations = raw_crop.annotations.duration
    descriptions = raw_crop.annotations.description
    bad_mask = np.array(['BAD' in d.upper() for d in descriptions])
    bad_intervals = [(onset, onset + dur) for onset, dur, is_bad in zip(onsets, durations, bad_mask) if is_bad]
    
    total_duration = raw_crop.times[-1]
    start = 0.0
    
    for onset, offset in sorted(bad_intervals):
        if onset - start >= min_duration:
            raw_crop = raw_crop.copy().crop(tmin=start, tmax=onset)
            quiet_segments.append(raw_crop)
        start = max(start, offset)
    
    if total_duration - start >= min_duration:
        raw_crop = raw_crop.copy().crop(tmin=start, tmax=total_duration)
        quiet_segments.append(raw_crop)
    
    if not quiet_segments:
        raise RuntimeError("No quiet segment was detected.")
    
    # make covariance by concatenating segments
    raw_for_cov = mne.concatenate_raws(quiet_segments)
    
    # check noise covariance
    noise_cov = mne.compute_raw_covariance(
        raw_for_cov,
        method='shrunk',
        rank='info',
        verbose=True
    )
    
    mne.viz.plot_cov(noise_cov, info=raw_for_cov.info)
    
    # check rank
    rank_dict = mne.compute_rank(raw_for_cov)
    print(rank_dict)

    # save noise covariance
    save_dir = "/Users/Data/Dream/MNE/NoiseCov"
    os.makedirs(save_dir, exist_ok=True)
    outfilename = os.path.join(save_dir, f"Dream{subject}_day{ind+1}_noise_cov.fif")
    noise_cov.save(outfilename, overwrite=True)