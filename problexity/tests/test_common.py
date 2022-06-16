"""Basic tests."""
import problexity as px
from sklearn.datasets import make_classification
import numpy as np

def test_something():
    result = px.foo()
    assert result == 'bar'


def test_F1():
    metrics = [px.F1]

    reps = 10
    res = np.zeros((reps, len(metrics), 2))

    for r in range(reps):
        X_simple, y_simple = make_classification(n_samples=500, n_features=5, n_redundant=0, 
                        n_informative=5, n_clusters_per_class=2, n_classes=2, flip_y=0, 
                        class_sep=3.5, weights=[0.5, 0.5])

        X_complex, y_complex = make_classification(n_samples=500, n_features=10, n_redundant=0, 
                        n_informative=10, n_clusters_per_class=2, n_classes=2, flip_y=0.2, 
                        class_sep=0.01, weights=[0.9, 0.1])
        for m_id, m in enumerate(metrics):

            r_s = m(X_simple, y_simple)
            r_c = m(X_complex, y_complex)

            res[r, m_id, 0] = r_s
            res[r, m_id, 1] = r_c

    compare = res[:,:,0]<res[:,:,1]
    compare_reps = np.mean(compare, axis=0)

    for i in range(len(metrics)):
        print(metrics[i].__name__, compare_reps[i])

    assert compare_reps[0]>0.5