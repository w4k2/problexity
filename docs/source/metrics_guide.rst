#############
Metrics Guide
#############

lorem ipsum

Metric definitions
==================
.. :cite:`lorena2019complex`

lorem ipsum

**Example -- simple example**

.. code-block:: python

  from strlearn.evaluators import TestThenTrain
  from strlearn.ensembles import SEA
  from strlearn.utils.metrics import bac, f_score
  from strlearn.streams import StreamGenerator
  from sklearn.naive_bayes import GaussianNB

  stream = StreamGenerator(chunk_size=200, n_chunks=250)
  clf = SEA(base_estimator=GaussianNB())
  evaluator = TestThenTrain(metrics=(bac, f_score))

  evaluator.process(stream, clf)
  print(evaluator.scores)

References
----------
.. bibliography:: references.bib
  :list: enumerated
  :all:
