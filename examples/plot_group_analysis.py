
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne.datasets import limo
from mne.viz import plot_compare_evokeds

###############################################################################
# list with subjects ids that should be imported
subjects = list(range(2, 19))
# create a dictionary containing participants data for easy slicing
limo_epochs = {str(subject): limo.load_data(subject=subject) for subject in subjects}  # noqa

# get key from subjects dict for easy slicing
subjects = list(limo_epochs.keys())

###############################################################################
# drop EOGs and interpolate missing channels
for subject in subjects:
    limo_epochs[subject].drop_channels(['EXG1', 'EXG2', 'EXG3', 'EXG4'])
    limo_epochs[subject].interpolate_bads(reset_bads=True)

###############################################################################
# check metadata
print(limo_epochs[subjects[0]].metadata.head())

# create factor for phase-variable
name = "phase-coherence"
factor = 'factor-' + name
for subject in subjects:
    df = limo_epochs[subject].metadata
    df[factor] = pd.cut(df[name], 11, labels=False) / 10
    # overwrite metadata
    limo_epochs[subject].metadata = df

################################################################################
# --- compute and plot grand averages for phase-coherence factor
# create dict of colors for plot
colors = {str(val): val for val in sorted(df[factor].unique())}

# evoked responses per subject
evokeds = list()
for subject in subjects:
    subject_evo = {val: limo_epochs[subject][limo_epochs[subject].metadata[factor] == float(val)].average() for val in colors}  # noqa
    evokeds.append(subject_evo)

# evoked responses per level of phase-coherence
factor_evokeds = list()
for val in colors:
    factor_evo = {val: [evokeds[ind][val] for ind in list(range(len(evokeds)))]}  # noqa
    factor_evokeds.append(factor_evo)

# compute grand averages
grand_averages = {val: mne.grand_average(factor_evokeds[i][val]) for i, val in enumerate(colors)}  # noqa

# pick channel to plot
electrodes = ['A19', 'C22', 'B8']
# initialize figure
fig, axs = plt.subplots(len(electrodes), 1, figsize=(10, 15))
for electrode in list(range(len(electrodes))):
    plot_compare_evokeds(grand_averages,
                         axes=axs[electrode],
                         ylim=dict(eeg=[-12.5, 12.5]),
                         colors=colors,
                         split_legend=True,
                         picks=electrodes[electrode],
                         truncate_yaxis='max_ticks',
                         cmap=(name + " Percentile", "magma"))
plt.show()

# plot individual erps
fig, axs = plt.subplots(17, 1, figsize=(5, 20))
for ind in list(range(0, len(evokeds))):
    plot_compare_evokeds(evokeds[ind],
                         axes=axs[ind],
                         title='subject %s' % (subjects[ind]),
                         ylim=dict(eeg=[-15, 15]),
                         colors=colors,
                         split_legend=True,
                         picks=electrodes[2],
                         truncate_yaxis='max_ticks',
                         cmap=(name + " Percentile", "magma"))
plt.show()
