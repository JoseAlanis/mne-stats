.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_limo_examples_single_subject_plot_r_squared.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_limo_examples_single_subject_plot_r_squared.py:


=================================
Plot R-squared for a linear model
=================================


.. code-block:: default


    # Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
    #
    # License: BSD (3-clause)

    import numpy as np

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    from mne.decoding import Vectorizer, get_coef
    from mne.datasets import limo
    from mne.evoked import EvokedArray
    from mne.io.pick import pick_types







list with subjects ids that should be imported


.. code-block:: default

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





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1052 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    Computing interpolation matrix from 117 sensor positions
    Interpolating 11 sensors



use epochs metadata to create design matrix for linear regression analyses


.. code-block:: default


    # add intercept
    design = limo_epochs['2'].metadata.copy().assign(intercept=1)
    # effect code contrast for categorical variable (i.e., condition a vs. b)
    design['face a - face b'] = np.where(design['face'] == 'A', 1, -1)
    # create design matrix with named predictors
    predictors = ['intercept', 'face a - face b', 'phase-coherence']
    design = design[predictors]







--- run linear regression analysis using scikit-learn ---
data to be analysed


.. code-block:: default

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

    # set up model and fit linear model
    linear_model = LinearRegression(fit_intercept=False)
    linear_model.fit(design, Y)

    # extract the coefficients for linear model estimator
    betas = get_coef(linear_model, 'coef_')

    # calculate coefficient of determination (r-squared)
    r_squared = r2_score(Y, linear_model.predict(design), multioutput='raw_values')
    # project r-squared back to channels by times space
    r_squared = r_squared.reshape((n_channels, n_times))
    r_squared = EvokedArray(r_squared, epochs_info, tmin)







--- plot r-squared ---
only show -250 to 500 ms


.. code-block:: default

    ts_args = dict(xlim=(-.25, 0.5),
                   unit=False,
                   ylim=dict(eeg=[0, 0.8]))
    topomap_args = dict(cmap='Reds', scalings=dict(eeg=1),
                        vmin=0, vmax=0.8, average=0.05)
    # create plot
    fig = r_squared.plot_joint(ts_args=ts_args,
                               topomap_args=topomap_args,
                               title='Proportion of variance explained by '
                                     'predictors',
                               times=[.13, .23])
    fig.axes[0].set_ylabel('R-squared')



.. image:: /limo_examples/single_subject/images/sphx_glr_plot_r_squared_001.png
    :class: sphx-glr-single-img





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  6.542 seconds)


.. _sphx_glr_download_limo_examples_single_subject_plot_r_squared.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_r_squared.py <plot_r_squared.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_r_squared.ipynb <plot_r_squared.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
