# problexity

[![w4k2](https://circleci.com/gh/w4k2/problexity.svg?style=shield)](https://circleci.com/gh/w4k2/problexity)
[![codecov](https://codecov.io/gh/w4k2/problexity/branch/master/graph/badge.svg?token=KxuYRg7J8B)](https://codecov.io/gh/w4k2/problexity)
[![Documentation Status](https://readthedocs.org/projects/problexity/badge/?version=latest)](http://problexity.readthedocs.io)
[![PyPI version](https://badge.fury.io/py/problexity.svg)](https://badge.fury.io/py/problexity)

The `problexity` module is an open-source python library containing the implementation of measures describing the complexity of the classification and regression problems. The package contains a ComplexityCalculator model, allowing the calculation, analysis and visualization of problem complexity measures.

## Citation policy

If you use problexity in a scientific publication, We would appreciate citation to the following papers, including introduction of library and original introduction of used measures:

```
@article{komorniczak2023problexity,
  title={problexity—An open-source Python library for supervised learning problem complexity assessment},
  author={Komorniczak, Joanna and Ksieniewicz, Pawe{\l}},
  journal={Neurocomputing},
  volume={521},
  pages={126--136},
  year={2023},
  publisher={Elsevier}
}
```

```
@article{lorena2018complex,
  title={How complex is your classification problem},
  author={Lorena, A and Garcia, L and Lehmann, Jens and Souto, M and Ho, T},
  journal={A survey on measuring classification complexity. arXiv},
  year={2018}
}
```

## Quick start guide

### Installation

To use the `problexity` package, it will be absolutely useful to install it. Fortunately, it is available in the *PyPI* repository, so you may install it using `pip`:

```shell
pip3 install -U problexity
```

The package is also available through `conda`:
```
conda install -c w4k2 problexity
```

To enable the possibility to modify the measures provided by `problexity` or in case of necessity to expand it with functions that it does not yet include, it is also possible to install the module directly from the source code. If any modifications are introduced, they propagate to the module currently available to the environment.

```shell
git clone https://github.com/w4k2/problexity.git
cd problexity
make install
```

### Minimal processing example

The `problexity` module is imported in the standard Python fashion. At the same time, for the convenience of implementation, the authors recommend importing it under the `px` alias:

```python
# Importing problexity
import problexity as px
```

The library is equipped with the `ComplexityCalculator` calculator, which serves as the basic tool for establishing metrics. The following code presents an example of the generation of a synthetic data set – typical for the `scikit-learn` module – and the determination of the value of measures by fitting the complexity model in accordance with the standard API adopted for `scikit-learn` estimators:

```python
# Loading benchmark dataset from scikit-learn
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

# Initialize CoplexityCalculator with default parametrization
cc = px.ComplexityCalculator()

# Fit model with data
cc.fit(X,y)
```

As the $L1$, $L2$ and $L3$ measures use the recommended `LinearSVC` implementation from the `svm` module of the `scikit-learn` package in their calculations, the warning "`ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.`" may occur. It is not a problem for the metric calculation -- only indicating the lack of linear problem separability.

The complexity calculator object stores a list of all estimated measures that can be read by the model's `complexity` attribute:

```python
cc.complexity
```
```
[0.227 0.064 0.000 0.478 0.012 0.225 0.070 0.042 0.043 0.296 0.084
 0.025 0.178 0.912 0.741 0.268 0.569 0.053 0.002 0.033 0.047 0.122]
```

They appear in the list in the same order as the declarations of the used metrics, which can also be obtained from the hidden method `_metrics()`:

```python
cc._metrics()
```
```
['f1', 'f1v', 'f2', 'f3', 'f4', 'l1', 'l2', 'l3', 'n1', 'n2', 'n3', 
 'n4', 't1', 'lsc', 'density', 'clsCoef', 'hubs', 't2', 't3', 't4', 
 'c1', 'c2']
```

The problem difficulty score can also be obtained as a single scalar measure, which is the arithmetic mean of all measures used in the calculation:

```python
cc.score()
```
```
0.203
```

The problexity module, in addition to raw data output, also provides two standard representations of problem analysis. The first is a report in the form of a dictionary presenting the number of patterns (`n_samples`), attributes (`n_features`), classes (`classes`), their prior distribution (`prior_probability`), average metric (`score`) and all member metrics (`complexities`), which can be obtained using the model's `report()` method:

```python
cc.report()
```
```
{
    'n_samples': 569, 
    'n_features': 30, 
    'n_classes': 2, 
    'classes': array([0, 1]), 
    'prior_probability': array([0.373, 0.627]), 
    'score': 0.214, 
    'complexities': 
    {
        'f1': 0.227, 'f1v': 0.064, 'f2': 0.001, 'f3': 0.478, 'f4': 0.012, 
        'l1': 0.433, 'l2' : 0.069, 'l3': 0.049, 'n1': 0.043, 'n2': 0.296, 
        'n3': 0.084, 'n4' : 0.039, 't1': 0.178, 't2': 0.053, 't3': 0.002, 
        't4': 0.033, 'c1' : 0.047, 'c2': 0.122,
        'lsc': 0.912, 'density': 0.741, 'clsCoef': 0.268, 'hubs': 0.569
    }
}
```

The second form of reporting is a graph which, in the polar projection, collates all metrics, grouped into categories using color codes:

- `red` – feature based measures,
- `orange` – linearity measures,
- `yellow` – neighborhood measures,
- `green` – network measures,
- `teal` – dimensionality measures,
- `blue` – class imbalance measures.

Each problem difficulty category occupies the same graph area, meaning that contexts that are less numerous in metrics (class imbalance) are not dominated in this presentation by categories described by many metrics (neighborhood). The illustration is built with the standard tools of the `matplotlib` module as a subplot of a figure and can be generated with the following source code:

```python
# Import matplotlib
import matplotlib.pyplot as plt

# Prepare figure
fig = plt.figure(figsize=(7,7))

# Generate plot describing the dataset
cc.plot(fig, (1,1,1))
```

An example of a complexity graph is shown below.

![Example graph](example_graph.png)
