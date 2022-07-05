#################
Quick start guide
#################

Installation
------------

To use the `problexity` package, it will be absolutely useful to install it. Fortunately, it is available in the PyPI repository, so you may install it using `pip`::

  pip install -U problexity

To enable the possibility to modify the measures provided by `problexity` or in case of necessity to expand it with functions that it does not yet include, it is also possible to install the module directly from the source code. If any modifications are introduced, they propagate to the module currently available to the environment::

  git clone https://github.com/w4k2/problexity.git
  cd problexity
  make install

Minimal processing example
--------------------------

The `problexity` module is imported in the standard Python fashion. At the same time, for the convenience of implementation, the authors recommend importing it under the `px` alias::

.. code-block:: python

  # Importing problexity
  import problexity as px

The library is equipped with the `ComplexityCalculator` calculator, which serves as the basic tool for establishing metrics. The following code presents an example of the generation of a synthetic data set – typical for the `scikit-learn` module – and the determination of the value of measures by fitting the complexity model in accordance with the standard API adopted for `scikit-learn` estimators::

.. code-block:: python

  # Loading benchmark dataset from scikit-learn
  from sklearn.datasets import load_breast_cancer
  X, y = load_breast_cancer(return_X_y=True)

  # Initialize CoplexityCalculator with default parametrization
  cc = px.ComplexityCalculator()

  # Fit model with data
  cc.fit(X,y)

As the $L1$, $L2$ and $L3$ measures use the recommended `LinearSVC` implementation from the `svm` module of the `scikit-learn` package in their calculations, the warning "`ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.`" may occur. It is not a problem for the metric calculation -- only indicating the lack of linear problem separability.

The complexity calculator object stores a list of all estimated measures that can be read by the model's `complexity` attribute::

.. code-block:: python
  
  cc.complexity