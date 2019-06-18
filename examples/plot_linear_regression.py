"""
================================================================
Plot beta coefficients from linear model estimation with sklearn
================================================================

"""

# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from sklearn.linear_model import LinearRegression

from mne.decoding import Vectorizer, get_coef
from mne.datasets import limo
from mne.stats import linear_regression
from mne.evoked import EvokedArray
from mne.io.pick import pick_types

###############################################################################
# list with subjects ids that should be imported
subjects = list(range(2, 3))
# create a dictionary containing participants data
limo_epochs = {str(subj): limo.load_data(subject=subj) for subj in subjects}

# get key from subjects dict for easy slicing
# subjects = list(limo_epochs.keys())

# interpolate missing channels
for subject in limo_epochs.values():
    subject.interpolate_bads(reset_bads=True)

# pick channels that should be included in the analysis
picks_eeg = pick_types(limo_epochs['2'].info, eeg=True)
# channels to be excluded
exclude = ['EXG1', 'EXG2', 'EXG3', 'EXG4']

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
features = ['intercept', 'face a - face b', 'phase-coherence']
design = design[features]

###############################################################################
# --- run linear regression analysis using scikit-learn ---

# design matrix
design_matrix = design.copy()
# name of predictors
names = features.copy()

# data to be analysed
data = limo_epochs['2'].get_data(picks_eeg)
# place holder for results
out = EvokedArray(np.zeros(data.shape[1:]), epochs_info, tmin)

# vectorised (channel) data for linear regression
Y = Vectorizer().fit_transform(data)

# fit linear model with sklearn
# use fit intercept=false because b0 already in design matrix
linear_model = LinearRegression(fit_intercept=False)
fit = linear_model.fit(design_matrix, Y)

# extract coefficients (i.e., betas) for estimator
betas = get_coef(fit, 'coef_').T

# store coefficients for each feature in a dictionary
beta = dict()
for coef, feature in zip(betas, features):
    beta[feature] = coef.reshape(data.shape[1:])

# loop through predictors and save results in evoked object
lm_betas = {}
for feature in features:
    out_ = out.copy()
    out_.data[:] = beta[feature]
    lm_betas[feature] = out_

###############################################################################
# --- plot results of linear regression ---
# only show -250 to 500 ms
ts_args = dict(xlim=(-.25, 0.5))

# visualise effect of phase-coherence for sklearn estimation method.
lm_betas['phase-coherence'].plot_joint(ts_args=ts_args,
                                       title='Phase-coherence (sklearn betas)',
                                       times=[.23])

###############################################################################
# replicate analysis using mne.stats.linear_regression
reg = linear_regression(limo_epochs['2'], design, names=features)

# visualise effect of phase-coherence for mne.stats method.
reg['phase-coherence'].beta.plot_joint(ts_args=ts_args,
                                       title='Phase-coherence (mne.stats betas)',  # noqa
                                       times=[.23])
