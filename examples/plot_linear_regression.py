# -- import necessary packages and modules
from collections import namedtuple

import numpy as np
from scipy import linalg, stats
from sklearn.linear_model import LinearRegression

from mne.datasets import limo
from mne.stats import linear_regression
# from mne.source_estimate import SourceEstimate
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
predictors = ['intercept', 'face a - face b', 'phase-coherence']
design = design[predictors]

###############################################################################
# --- run linear regression analysis using scikit-learn ---

# design matrix
design_matrix = design.copy()
# name of predictors
names = predictors.copy()

# data to be analysed
data = limo_epochs['2'].get_data(picks_eeg)
# place holder for results
out = EvokedArray(np.zeros(data.shape[1:]), epochs_info, tmin)

# number of epochs in data set
n_samples = len(data)
# channels x time points
n_features = np.product(data.shape[1:])

# design matrix dimensions
n_rows, n_predictors = design_matrix.shape

# vectorised (channel) data for linear regression
y = np.reshape(data, (n_samples, n_features))

# fit linear model with sklearn
# use fit intercept=false because b0 already in design matrix
linaer_model = LinearRegression(fit_intercept=False).fit(design_matrix, y)

# extract coefficients (i.e., betas)
betas = linaer_model.coef_.T

# save predictions and calculate residual sum of squares
predictions = linaer_model.predict(design[predictors])
resid_sum_squares = sum((y - predictions) ** 2)  # sum of squared residuals

# degrees of freedom for inference
df = n_rows - n_predictors

# estimate and save linear regression results
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

# linear regression results
lm_params = beta, stderr, t_val, p_val, mlog10_p_val

# create epochs objects for storage of results
lm = namedtuple('lm', 'beta stderr t_val p_val mlog10_p_val')
lm_fits = {}

# loop through predictors and save results
for name in names:
    parameters = [p[name] for p in lm_params]
    for ii, value in enumerate(parameters):
        out_ = out.copy()
        # if not isinstance(out_, (SourceEstimate, Evoked)):
        #     raise RuntimeError('Invalid container.')
        out_._data[:] = value
        parameters[ii] = out_
    lm_fits[name] = lm(*parameters)

###############################################################################
# --- plot results of linear regression ---
# only show -250 to 500 ms
ts_args = dict(xlim=(-.25, 0.5))

# visualise effect of phase-coherence for sklearn estimation method.
lm_fits['phase-coherence'].beta.plot_joint(ts_args=ts_args,
                                           title='Phase-coherence (sklearn)',
                                           times=[.23])

###############################################################################
# replicate analysis with using mne.stats.linear_regression
reg = linear_regression(limo_epochs['2'], design, names=predictors)

# visualise effect of phase-coherence for mne.stats method.
reg['phase-coherence'].beta.plot_joint(ts_args=ts_args,
                                       title='Phase-coherence (mne.stats)',
                                       times=[.23])
