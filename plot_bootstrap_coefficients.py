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

from sklearn.linear_model import LinearRegression, Ridge

from mne.viz import plot_compare_evokeds
from mne.decoding import Vectorizer, get_coef
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

# pick channels that should be included in the analysis
picks_eeg = pick_types(limo_epochs['2'].info, eeg=True)
# channels to be excluded
exclude = ['EXG1', 'EXG2', 'EXG3', 'EXG4']

# save epochs information (needed for creating a homologous
# epochs object containing linear regression result)
epochs_info = limo_epochs['2'].copy().drop_channels(exclude).info
tmin = limo_epochs['2'].tmin

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
data = limo_epochs['2'].get_data(picks_eeg)

# number of epochs in data set
n_epochs = data.shape[0]

# number of channels and number of time points in each epoch
# we'll use this information later to bring the results of the
# the linear regression algorithm into an eeg-like format
# (i.e., channels x times points)
n_channels = len(picks_eeg)
n_times = len(limo_epochs['2'].times)

# vectorize (channel) data for linear regression
Y = Vectorizer().fit_transform(data)

# -- initialize bootstrap procedure --
# set random state for replication
random_state = 42
np.random.seed(random_state)

# number of random samples
boot = 2000

# choose which estimator to use (either "ridge" or "ols"
estimator = 'ridge'

# run bootstrap for regression coefficients
boot_betas = []
for i in range(boot):
    resamples = np.random.choice(range(n_epochs), n_epochs)
    # set up model: ridge or ols
    if estimator == 'ridge':
        linear_model = Ridge(fit_intercept=False, alpha=0)
    elif estimator == 'ols':
        linear_model = LinearRegression(fit_intercept=False)
    # fit model to bootstrap samples
    linear_model.fit(X=design.iloc[resamples], y=Y[resamples, :])
    # extract coefficients
    boot_betas.append(get_coef(linear_model, 'coef_'))

# compute low and high percentiles
lower, upper = np.quantile(boot_betas, [.025, .975], axis=0)

###############################################################################
# set up model: ridge or ols
if estimator == 'ridge':
    linear_model = Ridge(fit_intercept=False, alpha=0)
elif estimator == 'ols':
    linear_model = LinearRegression(fit_intercept=False)
# fit model
linear_model.fit(design, Y)

# extract the coefficients for linear model estimator
betas = get_coef(linear_model, 'coef_')

# project coefficients back to a channels x time points space.
lm_betas = dict()
ci = dict(lower_bound=dict(), upper_bound=dict())
# loop through predictors
for ind, predictor in enumerate(predictors):
    # extract coefficients for predictor in question
    beta = betas[:, ind]
    lower_bound = lower[:, ind]
    upper_bound = upper[:, ind]
    # back projection to channels x time points
    beta = beta.reshape((n_channels, n_times))
    lower_bound = lower_bound.reshape((n_channels, n_times))
    upper_bound = upper_bound.reshape((n_channels, n_times))
    # create evoked object containing the back projected coefficients
    # for each predictor
    lm_betas[predictor] = EvokedArray(beta, epochs_info, tmin)
    # dictionary containing upper and lower confidence boundaries
    ci['lower_bound'][predictor] = lower_bound
    ci['upper_bound'][predictor] = upper_bound

###############################################################################
# --- plot results of linear regression ---
# only show -250 to 500 ms
ts_args = dict(xlim=(-.25, 0.5))

# predictor to plot
predictor = 'phase-coherence'
# electrode to plot
pick = limo_epochs['2'].info['ch_names'].index('B8')

# visualise effect of phase-coherence for sklearn estimation method.
lm_betas[predictor].plot_joint(ts_args=ts_args,
                               title='Phase-coherence (sklearn betas)',
                               times=[.23])

# plot effect of phase-coherence on electrode B8 with 95% confidence interval
fig, ax = plt.subplots(figsize=(10, 7))
ax = plot_compare_evokeds(lm_betas[predictor], pick,
                          ylim=dict(eeg=[-11, 1]),
                          colors=['b'], axes=ax)
ax.axes[0].fill_between(limo_epochs['2']["Face/A"].times,
                        ci['lower_bound'][predictor][pick]*1e6,
                        ci['upper_bound'][predictor][pick]*1e6, alpha=0.25)
plt.plot()
