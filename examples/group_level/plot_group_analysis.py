"""
===============================================================
Plot group-level ERPs across percentiles of continuous variable
===============================================================

"""

# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mne import combine_evoked
from mne.datasets import limo
from mne.viz import plot_compare_evokeds

###############################################################################
# Here, we'll import multiple subjects from the LIMO-dataset and explore the
# group-level averages (i.e., grand-averages) for percentiles of a continuous
# variable.

# subject ids
subjects = range(2, 19)
# create a dictionary containing participants data for easy slicing
limo_epochs = {str(subj): limo.load_data(subject=subj) for subj in subjects}

# interpolate missing channels
for subject in limo_epochs.values():
    subject.interpolate_bads(reset_bads=True)
    # only keep eeg channels
    subject.pick_types(eeg=True)

###############################################################################
# create factor for phase-variable

name = "phase-coherence"
factor = 'factor-' + name
for subject in limo_epochs.values():
    df = subject.metadata
    df[factor] = pd.cut(df[name], 11, labels=False) / 10
    # overwrite metadata
    subject.metadata = df

################################################################################
# compute grand averages for phase-coherence factor

# get levels of phase_coherence factor
pc_factor = limo_epochs[str(subjects[0])].metadata[factor]
# create dict of colors for plot
colors = {str(val): val for val in sorted(pc_factor.unique())}

# evoked responses per subject
evokeds = list()
for subject in limo_epochs.values():
    subject_evo = {str(val): subject[subject.metadata[factor] == val].average()
                   for val in colors.values()}
    evokeds.append(subject_evo)

# evoked responses per level of phase-coherence
factor_evokeds = list()
for val in colors:
    factor_evo = {val: [evokeds[ind][val] for ind in range(len(evokeds))]}
    factor_evokeds.append(factor_evo)

# average phase-coherence betas
weights = np.repeat(1 / len(subjects), len(subjects))
grand_averages = {val: combine_evoked(factor_evokeds[i][val], weights=weights)
                  for i, val in enumerate(colors)}

# pick channel to plot
electrodes = ['A19', 'C22', 'B8']

# create figs
for electrode in electrodes:
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_compare_evokeds(grand_averages,
                         axes=ax,
                         ylim=dict(eeg=[-12.5, 12.5]),
                         colors=colors,
                         split_legend=True,
                         picks=electrode,
                         cmap=(name + " Percentile", "magma"))
    plt.show()

################################################################################
# plot individual ERPs for three exemplary subjects

# create figs
for i, subj in enumerate(evokeds[0:3]):
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_compare_evokeds(subj,
                         axes=ax,
                         title='subject %s' % (i + 2),
                         ylim=dict(eeg=[-20, 20]),
                         colors=colors,
                         split_legend=True,
                         picks=electrodes[2],
                         cmap=(name + " Percentile", "magma"))
plt.show()
