"""
=======================================================
Plot bootstrapped confidence interval for condition ERP
=======================================================

"""

# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

from mne.viz import plot_compare_evokeds
from mne.decoding import Vectorizer
from mne.datasets import limo

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

###############################################################################
# data to be analysed
face_a = limo_epochs['2']['Face/A'].get_data()

# number of trials per condition
n_epochs = face_a.shape[0]

# number of channels and number of time points in each epoch
# we'll this this information later to bring the results of the
# the linear regression algorithm into an eeg-like format
# (i.e., channels x times points)
n_channels = len(limo_epochs['2'].info['ch_names'])
n_times = len(limo_epochs['2'].times)

# vectorize (channel) data for linear regression
Y = Vectorizer().fit_transform(face_a)

###############################################################################
# run bootstrap for centrality estimator

# set random state for replication
random_state = 42
random = np.random.RandomState(random_state)

# number of random samples
boot = 2000

# initialize bootstrap
resampled_data = []
for i in range(boot):
    # pick random samples with replacement from data
    resamples = random.choice(range(n_epochs), n_epochs, replace=True)
    # compute centrality estimator on re-sampled data
    resampled_data.append(Y[resamples, :].mean(axis=0))

###############################################################################
# compute confidence interval

# compute low and high percentile
lower, upper = np.quantile(resampled_data, [.025, .975], axis=0)

# create evoked objects for percentile
# lower bound
lower = lower.reshape((n_channels, n_times))
# upper bound
upper = upper.reshape((n_channels, n_times))

###############################################################################
# plot results

# erp for condition
face_a_erp = limo_epochs['2']["Face/A"].average()

# electrode to plot
pick = face_a_erp.ch_names.index('B8')

# create figure
fig, ax = plt.subplots(figsize=(10, 7))
ax = plot_compare_evokeds(face_a_erp, pick,
                          ylim=dict(eeg=[-17.5, 10]),
                          show_sensors='upper right',
                          axes=ax)
ax.axes[0].fill_between(limo_epochs['2']["Face/A"].times,
                        upper[pick]*1e6,
                        lower[pick]*1e6, alpha=0.25)
plt.plot()
