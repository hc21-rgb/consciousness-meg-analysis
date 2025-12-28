"""
perform FreeSurfer reconstruction with recon-all and manual coregistration

"""

import os
import subprocess
from pathlib import Path
import mne


############# needs to be specified ##########

# participant/participant ID
subject = "Dream11"

#############################################


# path setting
fshome = "/Applications/freesurfer"
os.environ['FREESURFER_HOME'] = fshome
os.environ['SUBJECTS_DIR'] = f"{fshome}/subjects"
os.environ["PATH"] = f"{fshome}/bin:" + os.environ["PATH"]

FREESURFER_HOME = "/Applications/freesurfer"
SUBJECTS_DIR = f"{FREESURFER_HOME}/subjects"
nifti_file = f"/Users/Data/Dream/MRI/{subject}/NIFTI/{subject}_T1.nii.gz"


### recon-all
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

# generate source space and BEM for MNE
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

### manual coregistration
print("Coregistration by MNE GUI ")
mne.gui.coregistration(subject=subject, subjects_dir='/Applications/freesurfer/subjects')
# see <https://mne.tools/stable/generated/mne.gui.coregistration.html#mne.gui.coregistration> last access(2025/12/28)


print(">>> mne.gui.coregistration(subject=subject, subjects_dir='...')")