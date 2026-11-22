PYTEPFIT DATA (VENDORED)
========================

Source  : https://github.com/GriffithsLab/PyTepFit, folder data/
Version : commit 66b94e488c82478e6f302015b6cd8e8d9d33792a (2023-04-26)
License : MIT, Copyright (c) 2023 Davide Momi -- see LICENSE in this folder
Used by : syn_brain (core/anatomy.py); the same files neurofe's
          examples/example_train_jr.py expects in its --data_dir

In git (copied unchanged, sha256 checked against the source):

  leadfield                                       50 KB   EEG leadfield, 62 channels x 200 parcels
                                                          (numpy pickle; np.load(..., allow_pickle=True))
  Schaefer2018_200Parcels_7Networks_count.csv    100 KB   structural connectome, streamline counts
  Schaefer2018_200Parcels_7Networks_distance.csv 490 KB   inter-parcel distances, mm

Not in git (large; fetch when needed):

  only_high_trial.mat                            11.5 MB  real TMS-EEG evoked responses, 20 subjects
                                                          (needed only by scripts/calibrate_teacher.py)

    bash syn_brain/scripts/fetch_pytepfit_data.sh

syn_brain reads this folder by default. To use another copy:

    export SYN_BRAIN_PYTEPFIT_DATA=/path/to/PyTepFit/data
