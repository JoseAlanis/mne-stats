.. MNE-LIMO documentation master file, updated by
   jose c. garcia alanis on Fri Jul 05 09:04:55 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MNE-LIMO
========

This site is dedicated to the statistical analysis of Electro-encephalography (EEG) and Magneto-encephalography (MEG) data using `MNE-Python <https://martinos.org/mne/stable/index.html>`_.

Particular emphasis is put on the statistical analysis of this dat using **LI**\ near regression **MO**\ dels (i.e., LIMO).

For this purpose, we have started to replicate and extend the analysis and tools integrated in `LIMO MEEG <https://github.com/LIMO-EEG-Toolbox/limo_eeg>`_, a Matlab toolbox originally designed to interface with EEGLAB.


Analyzing (M)EEG data with MNE-LIMO
===================================

Currently, we are implementing a series of examples to fit linear models on single subject data and derive inferential measures to evaluate the estimated effects.
Please visit the :ref:`single subject analysis gallery <sphx_glr_limo_examples_single_subject>` for more information on how to fit linear models to single subjects' data.

In addition, we have started to develop method to translate single subject analysis to group-levels analysis, i.e., estimating linear regression effects over a series of subjects.
Please visit the :ref:`group level analysis gallery <sphx_glr_limo_examples_group_level>` for more information on how to carry out linear regression analysis on a group level.


Acknowledgements
================

This project is currently supported by a `2019 Google Summer of Code project <https://summerofcode.withgoogle.com/projects/#5715889406607360>`_ grant issued to José C. García Alanis.

Special acknowledgements go to Denis A. Engemann, Jona Sassenhagen, and the MNE-Community for their support, guidance, and inputs through the project.
