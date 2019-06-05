import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne.datasets import limo
from mne.viz import plot_compare_evokeds
from mne.stats import linear_regression

# load epochs
limo_epochs = limo.load_data(subject=2)
limo_epochs.drop_channels(['EXG1', 'EXG2', 'EXG3', 'EXG4'])
limo_epochs.interpolate_bads(reset_bads=True)
print(limo_epochs.metadata.head())

# create factor for phase-variable
name = "phase-coherence"
df = limo_epochs.metadata
df[name] = pd.cut(df[name], 11, labels=False) / 10
colors = {str(val): val for val in df[name].unique()}
limo_epochs.metadata = df.assign(Intercept=1)  # Add an intercept for later
evokeds = {val: limo_epochs[limo_epochs.metadata[name] == float(val)].average() for val in colors}

# pick channel to plot
pick = limo_epochs['Face/A'].ch_names.index('B11')

# plot activity at electrode 'B11'
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
plot_compare_evokeds(evokeds,  axes=ax,
                     colors=colors, split_legend=True, picks=pick,
                     truncate_yaxis='max_ticks',
                     cmap=(name + " Percentile", "magma"))
plt.show()

# create design
design = limo_epochs.metadata.copy()
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