.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_limo_examples_group_level_plot_group_analysis.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_limo_examples_group_level_plot_group_analysis.py:


=======================
Plot Grand-Average ERPs
=======================


.. code-block:: default


    # Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
    #
    # License: BSD (3-clause)

    import pandas as pd
    import matplotlib.pyplot as plt

    import mne
    from mne.datasets import limo
    from mne.viz import plot_compare_evokeds







list with subjects ids that should be imported


.. code-block:: default

    subjects = list(range(2, 19))
    # create a dictionary containing participants data for easy slicing
    limo_epochs = {str(subject): limo.load_data(subject=subject) for subject in subjects}  # noqa

    # get key from subjects dict for easy slicing
    subjects = list(limo_epochs.keys())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1052 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    1072 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    1050 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    1118 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    1108 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    1060 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    1030 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    1059 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    1038 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    1029 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    943 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    1108 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    998 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    1076 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    1061 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    1098 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped
    1103 matching events found
    No baseline correction applied
    Adding metadata with 2 columns
    0 projection items activated
    0 bad epochs dropped



drop EOGs and interpolate missing channels


.. code-block:: default

    for subject in subjects:
        limo_epochs[subject].drop_channels(['EXG1', 'EXG2', 'EXG3', 'EXG4'])
        limo_epochs[subject].interpolate_bads(reset_bads=True)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Computing interpolation matrix from 117 sensor positions
    Interpolating 11 sensors
    Computing interpolation matrix from 121 sensor positions
    Interpolating 7 sensors
    Computing interpolation matrix from 119 sensor positions
    Interpolating 9 sensors
    Computing interpolation matrix from 122 sensor positions
    Interpolating 6 sensors
    Computing interpolation matrix from 118 sensor positions
    Interpolating 10 sensors
    Computing interpolation matrix from 117 sensor positions
    Interpolating 11 sensors
    Computing interpolation matrix from 117 sensor positions
    Interpolating 11 sensors
    Computing interpolation matrix from 121 sensor positions
    Interpolating 7 sensors
    Computing interpolation matrix from 116 sensor positions
    Interpolating 12 sensors
    /Users/josealanis/Documents/github/mne-stats/examples/group_level/plot_group_analysis.py:32: RuntimeWarning: No bad channels to interpolate. Doing nothing...
      limo_epochs[subject].interpolate_bads(reset_bads=True)
    Computing interpolation matrix from 115 sensor positions
    Interpolating 13 sensors
    Computing interpolation matrix from 122 sensor positions
    Interpolating 6 sensors
    Computing interpolation matrix from 114 sensor positions
    Interpolating 14 sensors
    Computing interpolation matrix from 117 sensor positions
    Interpolating 11 sensors
    Computing interpolation matrix from 125 sensor positions
    Interpolating 3 sensors
    Computing interpolation matrix from 126 sensor positions
    Interpolating 2 sensors
    Computing interpolation matrix from 122 sensor positions
    Interpolating 6 sensors



check metadata


.. code-block:: default

    print(limo_epochs[subjects[0]].metadata.head())

    # create factor for phase-variable
    name = "phase-coherence"
    factor = 'factor-' + name
    for subject in subjects:
        df = limo_epochs[subject].metadata
        df[factor] = pd.cut(df[name], 11, labels=False) / 10
        # overwrite metadata
        limo_epochs[subject].metadata = df





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

      face  phase-coherence
    0    A        -1.456885
    1    A        -1.456885
    2    A        -0.108914
    3    A         1.624191
    4    A         0.276221
    Replacing existing metadata with 3 columns
    Replacing existing metadata with 3 columns
    Replacing existing metadata with 3 columns
    Replacing existing metadata with 3 columns
    Replacing existing metadata with 3 columns
    Replacing existing metadata with 3 columns
    Replacing existing metadata with 3 columns
    Replacing existing metadata with 3 columns
    Replacing existing metadata with 3 columns
    Replacing existing metadata with 3 columns
    Replacing existing metadata with 3 columns
    Replacing existing metadata with 3 columns
    Replacing existing metadata with 3 columns
    Replacing existing metadata with 3 columns
    Replacing existing metadata with 3 columns
    Replacing existing metadata with 3 columns
    Replacing existing metadata with 3 columns



--- compute and plot grand averages for phase-coherence factor
create dict of colors for plot


.. code-block:: default

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



.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /limo_examples/group_level/images/sphx_glr_plot_group_analysis_001.png
            :class: sphx-glr-multi-img

    *

      .. image:: /limo_examples/group_level/images/sphx_glr_plot_group_analysis_002.png
            :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Identifying common channels ...
    all channels are corresponding, nothing to do.
    Identifying common channels ...
    all channels are corresponding, nothing to do.
    Identifying common channels ...
    all channels are corresponding, nothing to do.
    Identifying common channels ...
    all channels are corresponding, nothing to do.
    Identifying common channels ...
    all channels are corresponding, nothing to do.
    Identifying common channels ...
    all channels are corresponding, nothing to do.
    Identifying common channels ...
    all channels are corresponding, nothing to do.
    Identifying common channels ...
    all channels are corresponding, nothing to do.
    Identifying common channels ...
    all channels are corresponding, nothing to do.
    Identifying common channels ...
    all channels are corresponding, nothing to do.
    Identifying common channels ...
    all channels are corresponding, nothing to do.
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    /Users/josealanis/anaconda3/lib/python3.7/site-packages/matplotlib/figure.py:445: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      % get_backend())
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...
    Float colors detected, mapping to percentiles ...




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  59.306 seconds)


.. _sphx_glr_download_limo_examples_group_level_plot_group_analysis.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_group_analysis.py <plot_group_analysis.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_group_analysis.ipynb <plot_group_analysis.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
