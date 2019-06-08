# -- import necessary packages and modules
from inspect import isgenerator
from collections import namedtuple

import numpy as np
from scipy import linalg
from sklearn.linear_model import LinearRegression

# import pandas as pd
# import matplotlib.pyplot as plt

# import mne
from mne.datasets import limo

# from mne.viz import plot_compare_evokeds
from mne.stats import linear_regression

from mne.source_estimate import SourceEstimate
from mne.epochs import BaseEpochs
from mne.evoked import Evoked, EvokedArray
from mne.utils import logger, _reject_data_segments, warn, fill_doc
from mne.io.pick import pick_types, pick_info, _picks_to_idx

# --- define new regression function


def linear_regression_reloaded(inst, design_matrix, names=None):
    """Fit Ordinary Least Squares regression (OLS).

    Parameters
    ----------
    inst : instance of Epochs | iterable of SourceEstimate
        The data to be regressed. Contains all the trials, sensors, and time
        points for the regression. For Source Estimates, accepts either a list
        or a generator object.
    design_matrix : ndarray, shape (n_observations, n_regressors)
        The regressors to be used. Must be a 2d array with as many rows as
        the first dimension of the data. The first column of this matrix will
        typically consist of ones (intercept column).
    names : array-like | None
        Optional parameter to name the regressors. If provided, the length must
        correspond to the number of columns present in regressors
        (including the intercept, if present).
        Otherwise the default names are x0, x1, x2...xn for n regressors.

    Returns
    -------
    results : dict of namedtuple
        For each regressor (key) a namedtuple is provided with the
        following attributes:

            beta : regression coefficients
            stderr : standard error of regression coefficients
            t_val : t statistics (beta / stderr)
            p_val : two-sided p-value of t statistic under the t distribution
            mlog10_p_val : -log10 transformed p-value.

        The tuple members are numpy arrays. The shape of each numpy array is
        the shape of the data minus the first dimension; e.g., if the shape of
        the original data was (n_observations, n_channels, n_timepoints),
        then the shape of each of the arrays will be
        (n_channels, n_timepoints).
    """
    if names is None:
        names = ['x%i' % i for i in range(design_matrix.shape[1])]

    if isinstance(inst, BaseEpochs):
        picks = pick_types(inst.info, meg=True, eeg=True, ref_meg=True,
                           stim=False, eog=False, ecg=False,
                           emg=False, exclude=['bads'])
        if [inst.ch_names[p] for p in picks] != inst.ch_names:
            warn('Fitting linear model to non-data or bad channels. '
                 'Check picking')
        msg = 'Fitting linear model to epochs'
        data = inst.get_data()
        out = EvokedArray(np.zeros(data.shape[1:]), inst.info, inst.tmin)
    elif isgenerator(inst):
        msg = 'Fitting linear model to source estimates (generator input)'
        out = next(inst)
        data = np.array([out.data] + [i.data for i in inst])
    elif isinstance(inst, list) and isinstance(inst[0], SourceEstimate):
        msg = 'Fitting linear model to source estimates (list input)'
        out = inst[0]
        data = np.array([i.data for i in inst])
    else:
        raise ValueError('Input must be epochs or iterable of source '
                         'estimates')
    logger.info(msg + ', (%s targets, %s regressors)' %
                (np.product(data.shape[1:]), len(names)))

    # calls refurbished function _fit_lm with sklearn estimator
    lm_params = _fit_lm_reloaded(data, design_matrix, names)
    lm = namedtuple('lm', 'beta stderr t_val p_val mlog10_p_val')
    lm_fits = {}

    for name in names:
        parameters = [p[name] for p in lm_params]
        for ii, value in enumerate(parameters):
            out_ = out.copy()
            if not isinstance(out_, (SourceEstimate, Evoked)):
                raise RuntimeError('Invalid container.')
            out_._data[:] = value
            parameters[ii] = out_
        lm_fits[name] = lm(*parameters)
    logger.info('Done')
    return lm_fits


# change mne _fit_lm function
def _fit_lm_reloaded(data, design_matrix, names):
    """Aux function."""
    from scipy import stats
    n_samples = len(data)
    n_features = np.product(data.shape[1:])
    if design_matrix.ndim != 2:
        raise ValueError('Design matrix must be a 2d array')
    n_rows, n_predictors = design_matrix.shape

    if n_samples != n_rows:
        raise ValueError('Number of rows in design matrix must be equal '
                         'to number of observations')
    if n_predictors != len(names):
        raise ValueError('Number of regressor names must be equal to '
                         'number of column in design matrix')

    y = np.reshape(data, (n_samples, n_features))

    # fit linear model with sklearn
    # use fit intercept=false because b0 already in design matrix
    linaer_model = LinearRegression(fit_intercept=False).fit(design_matrix, y)

    # extract coefficients (i.e., betas)
    betas = linaer_model.coef_.T

    # save predictions and calculate rss
    predictions = linaer_model.predict(design[predictors])
    resid_sum_squares = sum((y - predictions) ** 2)  # sum of squared residuals

    # dfs
    df = n_rows - n_predictors

    sqrt_noise_var = np.sqrt(resid_sum_squares / df).reshape(data.shape[1:])
    design_invcov = linalg.inv(np.dot(design_matrix.T, design_matrix))
    unscaled_stderrs = np.sqrt(np.diag(design_invcov))
    tiny = np.finfo(np.float64).tiny
    beta, stderr, t_val, p_val, mlog10_p_val = (dict() for _ in range(5))

    for x, unscaled_stderr, predictor in zip(betas, unscaled_stderrs, names):
        beta[predictor] = x.reshape(data.shape[1:])
        stderr[predictor] = sqrt_noise_var * unscaled_stderr
        p_val[predictor] = np.empty_like(stderr[predictor])
        t_val[predictor] = np.empty_like(stderr[predictor])

        stderr_pos = (stderr[predictor] > 0)
        beta_pos = (beta[predictor] > 0)
        t_val[predictor][stderr_pos] = (beta[predictor][stderr_pos] /
                                        stderr[predictor][stderr_pos])
        cdf = stats.t.cdf(np.abs(t_val[predictor][stderr_pos]), df)
        p_val[predictor][stderr_pos] = np.clip((1. - cdf) * 2., tiny, 1.)
        # degenerate cases
        mask = (~stderr_pos & beta_pos)
        t_val[predictor][mask] = np.inf * np.sign(beta[predictor][mask])
        p_val[predictor][mask] = tiny
        # could do NaN here, but hopefully this is safe enough
        mask = (~stderr_pos & ~beta_pos)
        t_val[predictor][mask] = 0
        p_val[predictor][mask] = 1.
        mlog10_p_val[predictor] = -np.log10(p_val[predictor])

    return beta, stderr, t_val, p_val, mlog10_p_val


###############################################################################
# list with subjects ids that should be imported
subjects = list(range(2, 3))
# create a dictionary containing participants data for easy slicing
limo_epochs = {str(subject): limo.load_data(subject=subject) for subject in subjects}  # noqa

# get key from subjects dict for easy slicing
subjects = list(limo_epochs.keys())

# drop EOGs and interpolate missing channels
for subject in subjects:
    limo_epochs[subject].drop_channels(['EXG1', 'EXG2', 'EXG3', 'EXG4'])
    limo_epochs[subject].interpolate_bads(reset_bads=True)

###############################################################################
# create design for linear regression
design = limo_epochs['2'].metadata.copy()
design = design.assign(intercept=1)  # add intercept
design['face a - face b'] = np.where(design['face'] == 'A', 1, -1)
predictors = ['intercept', 'face a - face b', 'phase-coherence']
design = design[predictors]

###############################################################################
# fit linear model with old lm function
reg = linear_regression(limo_epochs[subjects[0]], design,
                        names=predictors)

# Visualise effect of phase-coherence.
# only show -250 to 500 ms
ts_args = dict(xlim=(-.25, 0.5))
reg['phase-coherence'].beta.plot_joint(ts_args=ts_args,
                                       title='Effect of phase-coherence (old lm)',  # noqa
                                       times=[.23])

###############################################################################
# fit linear model with new lm function
reg_new = linear_regression_reloaded(limo_epochs[subjects[0]], design,
                                     names=predictors)

# Visualise effect of phase-coherence.
# only show -250 to 500 ms
ts_args = dict(xlim=(-.25, 0.5))
reg_new['phase-coherence'].beta.plot_joint(ts_args=ts_args,
                                           title='Effect of phase-coherence (new lm)',  # noqa
                                           times=[.23])
