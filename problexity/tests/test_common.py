"""Basic tests."""
import problexity as px
from sklearn.datasets import make_classification
import numpy as np
    
def _get_comparison(metric):
    reps = 20
    res = np.zeros((reps, 2))

    for r in range(reps):
        X_simple, y_simple = make_classification(n_samples=500, n_features=5, n_redundant=0, 
                        n_informative=5, n_clusters_per_class=2, n_classes=2, flip_y=0, 
                        class_sep=10, weights=[0.5, 0.5])

        X_complex, y_complex = make_classification(n_samples=500, n_features=10, n_redundant=0, 
                        n_informative=10, n_clusters_per_class=2, n_classes=2, flip_y=0.2, 
                        class_sep=0.01, weights=[0.9, 0.1])

        r_s = metric(X_simple, y_simple)
        r_c = metric(X_complex, y_complex)

        res[r, 0] = r_s
        res[r, 1] = r_c

    compare = np.mean(res[:,0]<res[:,1])
    return compare

def test_F1():
    metric = px.f1
    assert _get_comparison(metric)>0.5

def test_F1v():
    metric = px.f1v
    assert _get_comparison(metric)>0.5

def test_F2():
    metric = px.f2
    assert _get_comparison(metric)>0.5

def test_F3():
    metric = px.f3
    assert _get_comparison(metric)>0.5

def test_F4():
    metric = px.f4
    assert _get_comparison(metric)>0.5

def test_L1():
    metric = px.l1
    assert _get_comparison(metric)>0.5

def test_L2():
    metric = px.l2
    assert _get_comparison(metric)>0.5

def test_L3():
    metric = px.l3
    assert _get_comparison(metric)>0.5

def test_N1():
    metric = px.n1
    assert _get_comparison(metric)>0.5

def test_N2():
    metric = px.n2
    assert _get_comparison(metric)>0.5

def test_N3():
    metric = px.n3
    assert _get_comparison(metric)>0.5

def test_N4():
    metric = px.n4
    assert _get_comparison(metric)>0.5

def test_T1():
    metric = px.t1
    assert _get_comparison(metric)>0.5

def test_LSC():
    metric = px.lsc
    assert _get_comparison(metric)>0.5

def test_density():
    metric = px.density
    assert _get_comparison(metric)>0.5

def test_clsCoef():
    metric = px.clsCoef
    assert _get_comparison(metric)>0.5

def test_hubs():
    metric = px.hubs
    assert _get_comparison(metric)>0.5

def test_T2():
    metric = px.t2
    assert _get_comparison(metric)>0.5

def test_T3():
    metric = px.t3
    assert _get_comparison(metric)>0.5

def test_T4():
    metric = px.t4
    assert _get_comparison(metric)>0.5

def test_C1():
    metric = px.c1
    assert _get_comparison(metric)>0.5

def test_C2():
    metric = px.c2
    assert _get_comparison(metric)>0.5

def test_ComplexityCalculator():
    c = px.ComplexityCalculator()
    reps = 10
    res = np.zeros((reps, 2))

    for r in range(reps):
        X_simple, y_simple = make_classification(n_samples=500, n_features=5, n_redundant=0, 
                        n_informative=5, n_clusters_per_class=2, n_classes=2, flip_y=0, 
                        class_sep=3.5, weights=[0.5, 0.5])

        X_complex, y_complex = make_classification(n_samples=500, n_features=10, n_redundant=0, 
                        n_informative=10, n_clusters_per_class=2, n_classes=2, flip_y=0.2, 
                        class_sep=0.01, weights=[0.9, 0.1])

        res[r, 0] = c.fit(X_simple, y_simple).score()
        res[r, 1] = c.fit(X_complex, y_complex).score()

    comparison = np.mean(res, axis=0)
    assert comparison[0]<comparison[1]