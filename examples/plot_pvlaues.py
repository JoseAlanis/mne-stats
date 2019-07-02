"""
================================================
Plot bootstrapped p-values for beta coefficients
================================================
"""

# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from mne.io.pick import pick_types
from mne.decoding import Vectorizer, get_coef
from mne.datasets import limo
from mne.evoked import EvokedArray
from mne.stats import fdr_correction
from mne.viz import plot_compare_evokeds

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

# number of trials and number of predictors
n_trials, n_predictors = design.shape
# degrees of freedom
dfs = float(n_trials - n_predictors)

# vectorize (channel) data for linear regression
Y = Vectorizer().fit_transform(data)

# set up model and fit linear model
linear_model = LinearRegression(fit_intercept=False)
linear_model.fit(design, Y)

###############################################################################
# extract the coefficients for linear model estimator
betas = get_coef(linear_model, 'coef_')

# compute predictions
predictions = linear_model.predict(design)

# compute sum of squared residuals and mean squared error
residuals = linear_model.predict(design) - Y
# sum of squared residuals
ssr =  np.sum(residuals ** 2, axis=0)
# mean squared error
sqrt_mse = np.sqrt(ssr / dfs)

# raw error terms from each predictor in design matrix:
# here, we take the inverse of a design) matrix projections
# (i.e., A^T*A)^-1 and extract the square root of the diagonal values.
error_terms = \
    np.sqrt(
        np.diag(
            np.linalg.pinv(
                np.dot(design.T, design)
            )
        )
    )

###############################################################################
# define dictionaries for results
lm_betas, stderrs, t_vals, p_vals, p_vals_evoked = (dict() for _ in range(5))

# define point asymptotic to zero to use as zero
tiny = np.finfo(np.float64).tiny

# loop through predictors to extract parameters
for ind, predictor in enumerate(predictors):

    # extract coefficients for predictor in question
    beta = betas[:, ind]
    # compute standard errors
    stderr = sqrt_mse * error_terms[ind]

    # compute t values
    t_val = beta / stderr
    # compute cumulative empirical distribution for t-values
    cdf = stats.t.cdf(np.abs(t_val), dfs)
    # and compute p-values
    p_val = 2. * (1. - cdf)

    # project results back to channels x time points space
    beta = beta.reshape((n_channels, n_times))
    stderr = stderr.reshape((n_channels, n_times))
    t_val = t_val.reshape((n_channels, n_times))
    # replace p-values == 0 with asymptotic value `tiny`
    p_val = np.clip(p_val, tiny, 1.).reshape((n_channels, n_times))

    # create evoked object for plotting
    lm_betas[predictor] = EvokedArray(beta, epochs_info, tmin)
    stderrs[predictor] = EvokedArray(stderr, epochs_info, tmin)
    t_vals[predictor] = EvokedArray(t_val, epochs_info, tmin)
    p_vals[predictor] = p_val
    p_vals_evoked[predictor] = EvokedArray(p_val*1e-6, epochs_info, tmin)


###############################################################################
# --- plot inference results for predictor = 'phase-coherence' ---

predictor = 'phase-coherence'

# only show -250 to 500 ms
ts_args = dict(xlim=(-.25, 0.5),
               # use unit=False to avoid conversion to micro-volt
               unit=False)
topomap_args = dict(cmap='RdBu_r',
                    # keep values scale
                    scalings=dict(eeg=1),
                    average=0.05)
# plot t-values
fig = t_vals[predictor].plot_joint(ts_args=ts_args,
                                   topomap_args=topomap_args,
                                   title='T-values for predictor %s'
                                         % predictor,
                                   times=[.13, .23])
fig.axes[0].set_ylabel('T-value')

###############################################################################
# correct p-values for multiple testing and create a mask for non-significant
# time point dor each channel.
reject_H0, fdr_pvals = fdr_correction(p_vals[predictor],
                                      alpha=0.01)
# plot t-values, masking non-significant time points.
fig = t_vals['phase-coherence'].plot_image(time_unit='s',
                                           mask=reject_H0,
                                           unit=False,
                                           # keep values scale
                                           scalings=dict(eeg=1))
fig.axes[1].set_title('T-value')

###############################################################################
# plot p-values as "erp"
# only show electrode `B8`
pick = limo_epochs['2'].info['ch_names'].index('B8')
fig, ax = plt.subplots(figsize=(10, 7))
ax = plot_compare_evokeds(p_vals_evoked[predictor],
                          picks=pick,
                          ylim=dict(eeg=[tiny, 1.]),
                          axes=ax)
# use a logarithmic scaling
ax.axes[0].set_yscale('log')
ax.axes[0].axhline(y=1, color='k')
plt.plot()
