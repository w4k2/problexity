"""Regression tests."""
import problexity as px
from sklearn.datasets import make_regression
import numpy as np

np.random.seed(123)
    
def _get_comparison(metric, args=[]):
    reps = 10
    res = np.zeros((reps, 2))
    rs = np.random.randint(100,1000,reps)

    for r in range(reps):
        X_simple, y_simple = make_regression(n_samples=300, n_features=50,
                        n_informative=50, noise=0, random_state=rs[r])

        X_complex, y_complex = make_regression(n_samples=300, n_features=100,
                        n_informative=10, noise=1200, random_state=rs[r])

        r_s = metric(X_simple, y_simple, *args)
        r_c = metric(X_complex, y_complex, *args)

        res[r, 0] = r_s
        res[r, 1] = r_c

    compare = np.mean(res[:,0]<res[:,1])
    return compare
 
### C1 tests ###

def test_C1():
    metric = px.regression.c1
    assert _get_comparison(metric)<0.5

### C2 tests ###

def test_C2():
    metric = px.regression.c2
    assert _get_comparison(metric)<0.5

## C3 tests ###

def test_C3():
    metric = px.regression.c3
    assert _get_comparison(metric)>0.5

def test_C3_l():
    metric = px.regression.c3
    assert _get_comparison(metric, [False])>0.5

## C4 tests ###

def test_C4():
    metric = px.regression.c4
    assert _get_comparison(metric)>0.5

## L1 tests ###

def test_L1():
    metric = px.regression.l1
    assert _get_comparison(metric)>0.5

## L2 tests ###

def test_L2():
    metric = px.regression.l2
    assert _get_comparison(metric)>0.5

### S1 tests ###

def test_S1():
    metric = px.regression.s1
    assert _get_comparison(metric)>0.5

### S2 tests ###

def test_S2():
    metric = px.regression.s2
    assert _get_comparison(metric)>0.5

### S3 tests ###

def test_S3():
    metric = px.regression.s3
    assert _get_comparison(metric)>0.5

### L3 tests ###

def test_L3():
    metric = px.regression.l3
    assert _get_comparison(metric)>0.5

### S4 tests ###

def test_S4():
    metric = px.regression.s4
    assert _get_comparison(metric)>0.5

### T2 tests ###

def test_T2():
    metric = px.regression.t2
    assert _get_comparison(metric)<0.5

### Test ComplexityCalculator ###
def test_ComplexityCalculator():
    c = px.ComplexityCalculator(mode='regression')
    reps = 5
    rs = np.random.randint(100,1000,reps)

    for r in range(reps):
        X_simple, y_simple = make_regression(n_samples=50, n_features=50,
                        n_informative=50, noise=0, random_state=rs[r])

        X_complex, y_complex = make_regression(n_samples=50, n_features=100,
                        n_informative=10, noise=1200, random_state=rs[r])

        c.fit(X_simple, y_simple).score()
        c.fit(X_complex, y_complex).score()
        
def test_ComplexityCalculator_report():
    c = px.ComplexityCalculator(mode = 'regression')

    X_simple, y_simple = make_regression(n_samples=50, n_features=50,
                        n_informative=50, noise=0, random_state = 123)

    report = c.fit(X_simple, y_simple).report()
    assert isinstance(report, dict)