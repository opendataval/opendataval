.. Data-Oob documentation master file, created by
   sphinx-quickstart on Thu Mar  9 01:42:29 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#################################
OpenDataVal Documentation
#################################

.. toctree::
   :maxdepth: 2
   :hidden:

   modules <modules>
   leaderboards <leaderboards>
   license <license>
   paper <paper>

**Version**: |version|

**Useful links**:
`Installation <https://github.com/opendataval/opendataval/#hourglass_flowing_sand-installation-options>`_ |
`Source Repository <https://github.com/opendataval/opendataval/>`_ |
`Issue Tracker <https://github.com/opendataval/opendataval/issues>`_ |
`Releases <https://github.com/opendataval/opendataval/releases>`_


**OpenDataVal** is an open-source initiative that with a diverse array of datasets/models
(image, NLP, and tabular), data valuation algorithims, and evaluation tasks using just a
few lines of code.

Abstract
==================

Assessing the quality and impact of individual data points is critical for improving model performance and mitigating undesirable biases within the training dataset. Several data valuation algorithms have been proposed to quantify data quality, however, there lacks a systemic and standardized benchmarking system for data valuation. In this paper, we introduce **OpenDataVal**, an easy-to-use and unified benchmark framework that empowers researchers and practitioners to apply and compare various data valuation algorithms. **OpenDataVal** provides an integrated environment that includes (i) a diverse collection of image, natural language, and tabular datasets, (ii) implementations of eleven different state-of-the-art data valuation algorithms, and (iii) a prediction model API that can import any models in scikit-learn. Furthermore, we propose four downstream machine learning tasks for evaluating the quality of data values. We perform benchmarking analysis using **OpenDataVal**, quantifying and comparing the efficacy of state-of-the-art data valuation approaches. We find that no single algorithm performs uniformly best across all tasks, and an appropriate algorithm should be employed for a user's downstream task. **OpenDataVal** is publicly available at https://opendataval.github.io with comprehensive documentation. Furthermore, we provide a leaderboard where researchers can evaluate the effectiveness of their own data valuation algorithms. 


Indices and tables
==================

* :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
