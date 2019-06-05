import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne.datasets import limo
from mne.viz import plot_compare_evokeds
from mne.stats import linear_regression

###############################################################################
# create a list with participants data
limo_epochs = list()
for subject in list(range(2, 6)):
    limo_epochs.append(limo.load_data(subject=subject))

# indices for loop
epochs_ind = list(range(0, len(limo_epochs)))
for ind in epochs_ind:
    limo_epochs[ind].drop_channels(['EXG1', 'EXG2', 'EXG3', 'EXG4'])
    limo_epochs[ind].interpolate_bads(reset_bads=True)

# check metadata
print(limo_epochs[0].metadata.head())

# create factor for phase-variable
name = "phase-coherence"
factor = 'factor-' + name
for ind in epochs_ind:
    df = limo_epochs[ind].metadata
    df[factor] = pd.cut(df[name], 11, labels=False) / 10

################################################################################
# --- compute and plot grand averages for phase-coherence factor
# create dict of colors for plot
colors = {str(val): val for val in limo_epochs[0].metadata[factor].unique()}

# evokeds per subject
evokeds = list()
for ind in epochs_ind:
    subject_evo = {val: limo_epochs[ind][limo_epochs[ind].metadata[factor] == float(val)].average() for val in colors}  # noqa
    evokeds.append(subject_evo)

# evokeds per level of phase-coherence
factor_evokeds = list()
for val in colors:
    factor_evo = {val: [evokeds[ind][val] for ind in epochs_ind]}
    factor_evokeds.append(factor_evo)

# dict for average
grand_averages = list()
for val in colors:
    mne.grand_average(grand_averages[val])

# compute grand averages
grand_averages = {val: mne.grand_average(factor_evokeds[i][val]) for i, val in enumerate(colors)}  # noqa

# pick channel to plot
pick = limo_epochs[0]['Face/A'].ch_names.index('B11')

# plot activity at electrode 'B11'
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
plot_compare_evokeds(grand_averages,  axes=ax,
                     colors=colors, split_legend=True, picks=pick,
                     truncate_yaxis='max_ticks',
                     cmap=(name + " Percentile", "magma"))
plt.show()

###############################################################################
# --- create design for linear regression [WIP]
design = limo_epochs[0].metadata.copy()
design = design.assign(intercept=1)  # add intercept
design['face a - face b'] = np.where(design['face'] == 'A', 1, -1)
names = ['intercept', 'face a - face b', 'phase-coherence']

# fit linear model
reg = linear_regression(limo_epochs, design[names], names=names)

ts_args = dict(xlim=(-.25, 0.5))
# Visualise effect of phase-coherence.
reg['phase-coherence'].beta.plot_joint(ts_args=ts_args,
                                       title='Effect of phase-coherence',
                                       times=[.23])
# intercept + beta of phase coherence
prediction = mne.combine_evoked([reg['phase-coherence'][0],
                                 reg['intercept'][0]],
                                weights='equal')

# plot difference between predicted values
prediction.plot_joint(times=[.15], title='Predicted values - phase coherence')
