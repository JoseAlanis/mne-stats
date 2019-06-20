"""
================================================================
Plot bootstrapped beta coefficients for a linear model estimator
================================================================

"""

# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from sklearn.linear_model import LinearRegression

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
data = limo_epochs['2'].get_data(picks_eeg)

# number of epochs in dataset
n_epochs = data.shape[0]

# number of channels and number of time points in each epoch
# we'll this this information later to bring the results of the
# the linear regression algorithm into an eeg-like format
# (i.e., channels x times points)
n_channels = len(picks_eeg)
n_times = len(limo_epochs['2'].times)

# vectorize (channel) data for linear regression
Y = Vectorizer().fit_transform(data)

# --- fit linear model with sklearn ---
linear_model = LinearRegression(fit_intercept=False)
linear_model.fit(design, Y)

# extract the resulting coefficients (i.e., betas)
betas = get_coef(linear_model, 'coef_')

# loop through the columns (i.e., the predictors) of coefficient matrix
# and extract coefficients for each predictor.
lm_betas = dict()
for ind, predictor in enumerate(predictors):
    # extract coefficients
    beta = betas[:, ind]
    # back projection to channels x time points
    beta = beta.reshape((n_channels, n_times))
    # create evoked object containing the back projected coefficients
    # for each predictor
    lm_betas[predictor] = EvokedArray(beta, epochs_info, tmin)


###############################################################################
# -- initialize bootstrap procedure --
boot_linear_model = LinearRegression(fit_intercept=False)

# set random state for replication
random_state = 42
np.random.seed(random_state)

# number of random samples
boot = 100
# shape of the resulting data (number of random samples x shape of Y)
boot_shape = (boot, n_epochs, n_channels * n_times)

# create sets of random samples
resamples = []
for i in range(boot):
    resamples.append(list(np.random.choice(range(n_epochs), n_epochs, replace=True)))  # noqa

# fit linear model to each of the random samples of Y
boot_reg = [boot_linear_model.fit(design.iloc[resamples[i]], Y[resamples[i], :]) for i in range(boot)]  # noqa

# extract coefficients
boot_betas = [get_coef(reg, 'coef_') for reg in boot_reg]

# create a centrality estimator (here the mean) for the linear models fitted
# on the random samples.
centrality_estimator = np.mean(np.asarray(boot_betas), axis=0)

boot_lm_betas = dict()
for ind, predictor in enumerate(predictors):
    # extract coefficients
    beta = centrality_estimator[:, ind]
    # back projection to channels x time points
    beta = beta.reshape((n_channels, n_times))
    # create evoked object containing the back projected coefficients
    # for each predictor
    boot_lm_betas[predictor] = EvokedArray(beta, epochs_info, tmin)

###############################################################################
# --- plot results of linear regression ---
# only show -250 to 500 ms
ts_args = dict(xlim=(-.25, 0.5))

# visualise effect of phase-coherence for sklearn estimation method.
lm_betas['phase-coherence'].plot_joint(ts_args=ts_args,
                                       title='Phase-coherence (sklearn betas)',
                                       times=[.23])

###############################################################################
# visualise effect of phase-coherence for bootstrapped  method.
boot_lm_betas['phase-coherence'].plot_joint(ts_args=ts_args,
                                            title='Phase-coherence (boot betas)',  # noqa
                                            times=[.23])
