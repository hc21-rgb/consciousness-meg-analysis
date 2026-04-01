#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is based on the MNE tutorials:
https://mne.tools/stable/auto_tutorials/index.html (accessed March 31, 2026)
"""

import subprocess
from pathlib import Path
import mne


########## need to specify ##########

# participant ID
subject = "Dream11"

nifti_file = "/Users/Data/MRI/Dream11/Dream11/DICOM_T1w.nii.gz"

#####################################


# setting
FREESURFER_HOME = "/Applications/freesurfer"
SUBJECTS_DIR = f"{FREESURFER_HOME}/subjects"


# reconstruct FreeSurfer by recon-all
cmd = f"""
export FREESURFER_HOME={FREESURFER_HOME} && \
source $FREESURFER_HOME/SetUpFreeSurfer.sh && \
export SUBJECTS_DIR={SUBJECTS_DIR} && \
recon-all -i {nifti_file} -s {subject} -all
"""

print("FreeSurfer...")
res = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
print(res.stdout)
print(res.stderr)

# fenerate source space and BEM for MNE
mne.set_config("SUBJECTS_DIR", SUBJECTS_DIR)
src = mne.setup_source_space(subject, spacing="oct6", subjects_dir=SUBJECTS_DIR, add_dist=False)

# create BEM surface 
mne.bem.make_watershed_bem(subject, subjects_dir=SUBJECTS_DIR, overwrite=True)

# creae BEM model and solution 
bem_model = mne.make_bem_model(subject=subject, ico=4, conductivity=(0.3,), subjects_dir=SUBJECTS_DIR)
bem_sol = mne.make_bem_solution(bem_model)

# save
bem_path = Path(SUBJECTS_DIR) / subject / "bem" / f"{subject}-bem-sol.fif"
mne.write_bem_solution(bem_path, bem_sol)

print("Source space and BEM for MNE are created")

# manual step
print("Coregistration by MNE GUI ")
mne.gui.coregistration(subject='Dream05', subjects_dir='/Applications/freesurfer/subjects')


print(">>> mne.gui.coregistration(subject='Dream05', subjects_dir='...')")
