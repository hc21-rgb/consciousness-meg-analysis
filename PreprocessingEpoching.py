#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1. Specify the participant number and the analysis duration
2. Load the metadata, MEG, and EOG data
3. Preprocess the data
4. Perform ICA
5. Save the epoched data

Steps 2–5 are performed automatically. The ICA quality control (QC) results are saved in a CSV file.
"""

import gc
import os
import re
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np
import pandas as pd


# =============================================================================
# Settings
# =============================================================================
participant = 11
duration = 20  # seconds before awakening/event
ica_method = "infomax"

file_path = Path.home() / "Data" / "Dream" / "DreamMetaData_pilot.xlsx"
save_dir = Path.home() / "Data" / "Dream" / "MNE" / "Epoch" / f"D{participant}"
save_dir.mkdir(parents=True, exist_ok=True)
save_file_name = f"D{participant}_{duration}s_ICA_QC.csv"

# Keep the original behavior here. If meaningful BAD annotations exist and
# epochs overlapping them should be rejected, change this to True.
reject_by_annotation_epochs = True


# =============================================================================
# Helper functions
# =============================================================================
def time_str_to_seconds(t):
    """Convert strings such as '1;23;45' or '23;45' to seconds."""
    parts = list(map(int, str(t).split(";")))
    return sum(a * b for a, b in zip([3600, 60, 1][-len(parts):], parts))


def take_runs(fif_files):
    """Group and sort split FIF files by run.

    Supported examples
    ------------------
    run_raw_tsss.fif, run_raw-1_tsss.fif, ...
        -> run 1
    session_raw_tsss.fif, session_raw-1_tsss.fif, ...
        -> run 1
    run2_raw_tsss.fif, run2_raw-1_tsss.fif, ...
        -> run 2
    session2_raw_tsss.fif, session2_raw-1_tsss.fif, ...
        -> run 2
    """
    run_groups = defaultdict(list)

    for fname in fif_files:
        match = re.match(
            r"^(?:run|session)(\d*)_?raw(?:-\d+)?_tsss\.fif$",
            fname,
        )
        if match is None:
            print(f"Skipping unrecognized tSSS file: {fname}")
            continue

        # No explicit number means the first run/session.
        run_id = int(match.group(1)) if match.group(1) else 1
        run_groups[run_id].append(fname)

    def split_sort_key(fname):
        """Put the main FIF first, then -1, -2, -3, ... in order."""
        split_match = re.search(r"-(\d+)_tsss\.fif$", fname)
        if split_match is None:
            return (0, 0)
        return (1, int(split_match.group(1)))

    for run_id in run_groups:
        run_groups[run_id] = sorted(run_groups[run_id], key=split_sort_key)

    return dict(sorted(run_groups.items()))

def check_continuous_raws(raw_list, file_list):
    """Check that split FIF chunks are exactly sample-contiguous."""
    for j in range(1, len(raw_list)):
        prev_raw = raw_list[j - 1]
        curr_raw = raw_list[j]

        if curr_raw.info["sfreq"] != prev_raw.info["sfreq"]:
            raise RuntimeError(
                f"Sampling frequency mismatch between {file_list[j - 1]} "
                f"and {file_list[j]}."
            )

        expected_first = prev_raw.last_samp + 1
        if curr_raw.first_samp != expected_first:
            raise RuntimeError(
                "FIF chunks are not sample-contiguous:\n"
                f"  {file_list[j - 1]} last_samp = {prev_raw.last_samp}\n"
                f"  {file_list[j]} first_samp = {curr_raw.first_samp}\n"
                f"  expected first_samp = {expected_first}"
            )


def remove_concat_boundaries(raw):
    """Remove artificial boundaries added by concatenate_raws().

    This is appropriate here because the chunks are first checked to be
    exactly sample-contiguous parts of the same continuous run.
    Other annotations are preserved.
    """
    descriptions = np.asarray(raw.annotations.description)
    boundary_idx = np.where(
        np.isin(descriptions, ["BAD boundary", "EDGE boundary"])
    )[0]

    if len(boundary_idx):
        raw.annotations.delete(boundary_idx)


# =============================================================================
# Initialization
# =============================================================================
ica_qc_rows = []
total_epochs = 0

# Create the output directory before run-wise saving starts.
os.makedirs(save_dir, exist_ok=True)

# Read metadata
df = pd.read_excel(file_path)
exp_days = sorted(
    map(
        int,
        df[df["Participant ID"] == participant]["Exp Day"]
        .dropna()
        .unique(),
    )
)


# =============================================================================
# Main loop: experimental day -> run
# =============================================================================
for ExpDay in exp_days:
    ExpDates = df.loc[
        (df["Participant ID"] == participant)
        & (df["Exp Day"] == ExpDay),
        "Exp date",
    ]

    if ExpDates.empty or pd.isna(ExpDates.iloc[0]):
        print(f"Skipping ExpDay {ExpDay}: Exp date is missing")
        continue

    ExpDate = ExpDates.iloc[0].strftime("%y%m%d")

    fifdir = Path.home() / "Data" / "Dream" / "MEG" / ExpDate
    fif_files = [f.name for f in fifdir.rglob("*tsss.fif")]

    runs = take_runs(fif_files)
    print(f"ExpDay {ExpDay}: detected run groups = {list(runs.keys())}")
    for _run_id, _files in runs.items():
        print(f"  run {_run_id}: {_files}")

    for run_id, file_list in runs.items():

        df_run = df[
            (df["Participant ID"] == participant)
            & (df["Exp Day"] == ExpDay)
            & (df["Run"] == run_id)
            & (df["Exclusion_flag"] != 1)
        ].copy()

        if df_run.empty:
            print(f"Skipping ExpDay {ExpDay}, run {run_id}")
            continue

        # Convert awakening times to seconds from the beginning of this run.
        df_run["_event_sec"] = [
            time_str_to_seconds(t)
            for t in df_run["Awake Time_adj"]
        ]

        print(f"\n===== ExpDay {ExpDay}, run {run_id} =====")
        for f in file_list:
            print(f)

        # =====================================================================
        # Read and concatenate manually split chunks of the same run
        # =====================================================================
        raw_list = [
            mne.io.read_raw_fif(
                str(fifdir / fname),
                preload=True,
                on_split_missing="ignore",
            )
            for fname in file_list
        ]

        # Verify that these are truly continuous pieces before concatenating.
        check_continuous_raws(raw_list, file_list)

        # Do NOT rebuild with RawArray. Keep first_samp and annotations.
        raw_crun = mne.concatenate_raws(raw_list)
        
        print("\nAnnotations immediately after concatenation:")
        print(raw_crun.annotations)

        # concatenate_raws() inserts BAD/EDGE boundary annotations at joins.
        # These chunks are verified to be contiguous pieces of the same run,
        # so remove only those artificial boundary annotations.
        remove_concat_boundaries(raw_crun)
        
        print("\nAnnotations after removing artificial boundaries:")
        print(raw_crun.annotations)
        
        print(
            "Raw first_samp before preprocessing:",
            raw_crun.first_samp,
            "duration:",
            raw_crun.times[-1],
            "sec",
        )

        # =====================================================================
        # Preprocessing
        # =====================================================================
        # 60-Hz line noise
        raw_crun.notch_filter(freqs=[60])
        
        # Band-pass filter
        raw_crun.filter(l_freq=0.3, h_freq=50)
        
        # Downsampling
        raw_crun.resample(sfreq=500, npad="auto")

        print(
            "Raw first_samp after resampling:",
            raw_crun.first_samp,
            "sfreq:",
            raw_crun.info["sfreq"],
        )
        print("Bad channels:", raw_crun.info["bads"])

        # =====================================================================
        # ICA
        # =====================================================================
        ica = mne.preprocessing.ICA(
            n_components=20,
            method=ica_method,
            random_state=97,
        )
        ica.fit(raw_crun, picks="meg")

        inds_ECG_ctps, scores_ECG_ctps = ica.find_bads_ecg(
            raw_crun,
            method="ctps",
            threshold="auto",            
        )
        inds_ECG_corr, scores_ECG_corr = ica.find_bads_ecg(
            raw_crun,
            method="correlation",
            measure="zscore",
            threshold="auto",
        )
        inds_EOGL, scores_EOGL = ica.find_bads_eog(
            raw_crun,
            ch_name="EEG020",
            measure="zscore",
            threshold="auto",
        )
        inds_EOGR, scores_EOGR = ica.find_bads_eog(
            raw_crun,
            ch_name="EEG021",
            measure="zscore",
            threshold="auto",
        )
        inds_muscle, scores_muscle = ica.find_bads_muscle(
            raw_crun,
            threshold=0.5,
            start=None,
            stop=None,
            l_freq=7,
            h_freq=45,
            sphere=None,
            verbose=None,
        )

        ECG_ctps_index = np.asarray(inds_ECG_ctps, dtype=int)
        ECG_corr_index = np.asarray(inds_ECG_corr, dtype=int)
        EOGL_corr_index = np.asarray(inds_EOGL, dtype=int)
        EOGR_corr_index = np.asarray(inds_EOGR, dtype=int)
        muscle_index = np.asarray(inds_muscle, dtype=int)

        all_indices = np.concatenate(
            [
                ECG_ctps_index,
                ECG_corr_index,
                EOGL_corr_index,
                EOGR_corr_index,
                muscle_index,
            ]
        )

        ica_exclusion_comps = np.unique(all_indices).astype(int)
        ica.exclude = ica_exclusion_comps.tolist()

        print("\n===== ICA artifact detection =====")
        print(f"ECG CTPS       : {inds_ECG_ctps} (n={len(inds_ECG_ctps)})")
        print(f"ECG correlation: {inds_ECG_corr} (n={len(inds_ECG_corr)})")
        print(f"EOG Left       : {inds_EOGL} (n={len(inds_EOGL)})")
        print(f"EOG Right      : {inds_EOGR} (n={len(inds_EOGR)})")
        print(f"Muscle         : {inds_muscle} (n={len(inds_muscle)})")

        ctps_set = set(inds_ECG_ctps)
        corr_set = set(inds_ECG_corr)
        print("--- ECG comparison ---")
        print("CTPS only       :", sorted(ctps_set - corr_set))
        print("Correlation only:", sorted(corr_set - ctps_set))
        print("Both            :", sorted(ctps_set & corr_set))
        print("Final exclusion :", ica_exclusion_comps)
        print("Total excluded  :", len(ica_exclusion_comps))

        # Save compact run-level QC information in memory for later CSV output.
        ica_qc_rows.append(
            {
                "participant": participant,
                "ExpDay": ExpDay,
                "run": run_id,
                "ECG_CTps_components": ",".join(map(str, inds_ECG_ctps)),
                "ECG_corr_components": ",".join(map(str, inds_ECG_corr)),
                "EOG_L_components": ",".join(map(str, inds_EOGL)),
                "EOG_R_components": ",".join(map(str, inds_EOGR)),
                "muscle_components": ",".join(map(str, inds_muscle)),
                "final_exclusion": ",".join(map(str, ica_exclusion_comps)),
                "n_excluded": len(ica_exclusion_comps),
                "max_ECG_CTps": float(np.max(scores_ECG_ctps)),
                "max_abs_ECG_corr": float(np.max(np.abs(scores_ECG_corr))),
                "max_abs_EOG_L": float(np.max(np.abs(scores_EOGL))),
                "max_abs_EOG_R": float(np.max(np.abs(scores_EOGR))),
                "max_muscle": float(np.max(scores_muscle)),
            }
        )

        # Apply ICA to a copy of the preprocessed run.
        reconst_raw = raw_crun.copy()
        ica.apply(reconst_raw)

        # =====================================================================
        # Build events WITHOUT resetting first_samp
        # =====================================================================
        # df_run['_event_sec'] is relative to the beginning of the loaded run,
        # while MNE events[:, 0] must use the Raw object's sample numbering.
        # Therefore:
        #   sample = raw.first_samp + index relative to raw start
        # time_as_index(..., use_rounding=True) is used to avoid truncation.
        valid_mask = (
            ((df_run["_event_sec"] - duration) >= 0)
            & (df_run["_event_sec"] <= reconst_raw.times[-1])
        )

        if (~valid_mask).any():
            invalid_times = df_run.loc[~valid_mask, "_event_sec"].tolist()
            print(
                "Excluded events outside available epoch range (sec):",
                invalid_times,
            )

        df_valid = df_run.loc[valid_mask].copy().reset_index(drop=True)

        if df_valid.empty:
            print(f"No valid events in ExpDay {ExpDay}, run {run_id}")
            del raw_list, raw_crun, reconst_raw, ica
            gc.collect()
            continue

        relative_samples = reconst_raw.time_as_index(
            df_valid["_event_sec"].to_numpy(dtype=float),
            use_rounding=True,
        )
        event_samples = relative_samples + reconst_raw.first_samp

        events = np.column_stack(
            [
                event_samples.astype(int),
                np.zeros(len(event_samples), dtype=int),
                np.ones(len(event_samples), dtype=int),
            ]
        )

        # Build metadata from exactly the same filtered rows used for events.
        # This fixes the original [:len(event_samples)] alignment problem.
        metadata = pd.DataFrame(
            {
                "participant": [participant] * len(df_valid),
                "ExpDay": [ExpDay] * len(df_valid),
                "run": [run_id] * len(df_valid),
                "time": df_valid["_event_sec"].to_numpy(),
                "dream": df_valid["Asleep and dreaming"].to_numpy(),
                "whitedream": df_valid["Dreaming but cannot recall"].to_numpy(),
                "unsure_dream": df_valid["Unsure if dreaming"].to_numpy(),
                "asleep_nodream": df_valid["Asleep with no dreams"].to_numpy(),
                "unsure_asleep": df_valid["Unsure if asleep"].to_numpy(),
                "wake_thinking": df_valid[
                    "Not asleep_thinking about something"
                ].to_numpy(),
                "wake_nothinking": df_valid[
                    "Not asleep_not thinking about anything"
                ].to_numpy(),
            }
        )

        epochs = mne.Epochs(
            raw=reconst_raw,
            events=events,
            event_id=1,
            tmin=-duration,
            tmax=0,
            baseline=None,
            preload=True,
            reject_by_annotation=reject_by_annotation_epochs,
            metadata=metadata,
        )

        # =====================================================================
        # Save Epochs separately for each experimental day and run
        # =====================================================================
        run_outfilename = os.path.join(
            save_dir,
            f"D{participant}_Day{ExpDay}_run{run_id}_"
            f"Notch_BP_500Hz_ICAzscauto_{duration}s_"
            "epo.fif",
        )

        epochs.save(run_outfilename, overwrite=True)
        total_epochs += len(epochs)

        print(f"Saved run-wise epochs: {run_outfilename}")
        print(f"Number of epochs in this run: {len(epochs)}")

        # Free run-level objects that are no longer needed.
        del epochs, raw_list, raw_crun, reconst_raw, ica
        gc.collect()


# =============================================================================
# Final checks and run-level ICA QC
# =============================================================================
if total_epochs == 0:
    raise RuntimeError("No epochs were created. Check run IDs and event times.")

print(f"Total epochs saved across all days/runs: {total_epochs}")

# Save ICA QC for every run (not just the final run).
if ica_qc_rows:
    qc_filename = os.path.join(
        save_dir,
        save_file_name,
    )
    pd.DataFrame(ica_qc_rows).to_csv(qc_filename, index=False)
    print(f"Saved ICA QC: {qc_filename}")
