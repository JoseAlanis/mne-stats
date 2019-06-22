"""
================================================================
Plot bootstrapped beta coefficients for a linear model estimator
================================================================

"""

# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

from mne import create_info
from mne.viz import plot_compare_evokeds, plot_evoked_topo
from mne.channels import read_montage
from mne.decoding import Vectorizer
from mne.datasets import limo
from mne.evoked import EvokedArray
from mne.io.pick import pick_types

###############################################################################
# list with subjects ids that should be imported
subjects = [2]
# create a dictionary containing participants data
limo_epochs = {str(subj): limo.load_data(subject=subj) for subj in subjects}

# get key from subjects dict for easy slicing
# subjects = list(limo_epochs.keys())

# interpolate missing channels
for subject in limo_epochs.values():
    subject.interpolate_bads(reset_bads=True)

# montage for info
montage = read_montage('biosemi128')

# pick channels that should be included in the analysis
picks_eeg = pick_types(limo_epochs['2'].info, eeg=True)
# channel names and types
ch_names = limo_epochs['2'].info['ch_names'][0:len(picks_eeg)]
ch_types = ['eeg'] * len(ch_names)
# channels to be excluded
exclude = ['EXG1', 'EXG2', 'EXG3', 'EXG4']
# sampling rate
sfreq = limo_epochs['2'].info['sfreq']

# create evoked info for results object
evoked_info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types,
                          montage=montage)

# save epochs information (needed for creating a homologous
# epochs object containing linear regression result)
epochs_info = limo_epochs['2'].copy().drop_channels(exclude).info
tmin = limo_epochs['2'].tmin.copy()

###############################################################################
# use epochs metadata to create design matrix for linear regression analyses

# add intercept
design = limo_epochs['2'].metadata.copy().assign(intercept=1)
# effect code contrast for categorical variable (i.e., condition a vs. b)
design['face a - face b'] = np.where(design['face'] == 'A', 1, -1)
# create design matrix with named predictors
predictors = ['intercept', 'face a - face b', 'phase-coherence']
design = design[predictors]

###############################################################################
# --- run linear regression analysis using scikit-learn ---
# data to be analysed
face_a = limo_epochs['2']['Face/A'].get_data(picks_eeg)
# face_b = limo_epochs['2']['Face/B'].get_data(picks_eeg)

# number of trials per condition
n_epochs = face_a.shape[0]

# number of channels and number of time points in each epoch
# we'll this this information later to bring the results of the
# the linear regression algorithm into an eeg-like format
# (i.e., channels x times points)
n_channels = len(picks_eeg)
n_times = len(limo_epochs['2'].times)

# vectorize (channel) data for linear regression
Y = Vectorizer().fit_transform(face_a)

# set random state for replication
random_state = 42
np.random.seed(random_state)

# number of random samples
boot = 2000

# bootstrap estimator
resampled_data = []
for i in range(boot):
    resamples = np.random.choice(range(n_epochs), n_epochs)
    resampled_data.append(Y[resamples, :].mean(axis=0))

# compute low and high percentile
lower, upper = np.quantile(resampled_data, [.025, .975], axis=0)

# create evoked objects for percentile
# lower bound
lower = lower.reshape((n_channels, n_times))
lower = EvokedArray(lower, info=evoked_info, tmin=tmin)
# upper bound
upper = upper.reshape((n_channels, n_times))
upper = EvokedArray(upper, info=evoked_info, tmin=tmin)

# erp for condition
face_a_erp = limo_epochs['2']["Face/A"].average()

# electrode to plot
pick = face_a_erp.ch_names.index('B29')

fig, ax = plt.subplots(figsize=(10, 7), sharex=True, sharey=True)
ax = plot_compare_evokeds(face_a_erp, pick, ylim=dict(eeg=[-3, 5]),
                          colors=['b'], axes=ax)
ax.axes[0].fill_between(limo_epochs['2']["Face/A"].times,
                        upper.data[pick]*1e6,
                        lower.data[pick]*1e6, alpha=0.25)
plt.plot()
