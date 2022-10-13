"""Classification tests."""
import problexity as px
from sklearn.datasets import make_classification
import numpy as np

np.random.seed(123)

def _get_comparison(metric):
    reps = 20
    res = np.zeros((reps, 2))
    rs = np.random.randint(100,10000,reps)

    for r in range(reps):
        X_simple, y_simple = make_classification(n_samples=500, n_features=5, n_redundant=0, 
                        n_informative=5, n_clusters_per_class=2, n_classes=2, flip_y=0, 
                        class_sep=10, weights=[0.5, 0.5], random_state=rs[r])

        X_complex, y_complex = make_classification(n_samples=500, n_features=10, n_redundant=0, 
                        n_informative=10, n_clusters_per_class=2, n_classes=2, flip_y=0.2, 
                        class_sep=0.01, weights=[0.9, 0.1], random_state=rs[r])

        r_s = metric(X_simple, y_simple)
        r_c = metric(X_complex, y_complex)

        res[r, 0] = r_s
        res[r, 1] = r_c

    compare = np.mean(res[:,0]<res[:,1])
    return compare
 
### F1 Tests ###

def test_F1():
    metric = px.classification.f1
    assert _get_comparison(metric)>0.5

def test_F1_close_to_zero():
    X, y = make_classification(random_state=115)
    X[y==0] += 100
    value = px.classification.f1(X,y)
    assert value<0.0001

def test_F1_equal_to_one():
    X, y = make_classification(random_state=115)
    X = np.concatenate((X[y==0],X[y==0]), axis=0)
    y = np.concatenate((y[y==0],np.ones((len(y[y==0])))))
    value = px.classification.f1(X,y)
    assert value==1

def test_F1_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.f1(X,y)
    assert 0 <= value <= 1 

def test_F1_single_class():
    X, y = make_classification(random_state=115, weights=[1., .0], flip_y=0.)
    value = px.classification.f1(X,y)
    assert np.isnan(value)

def test_F1_emptyset():
    X, y = [], []
    value = px.classification.f1(X,y)
    assert np.isnan(value)

### F1v Tests ###

def test_F1v():
    metric = px.classification.f1v
    assert _get_comparison(metric)>0.5

def test_F1v_close_to_zero():
    X, y = make_classification(random_state=115)
    X[y==0] += 100
    value = px.classification.f1v(X,y)
    assert value<0.0001

def test_F1v_equal_to_one():
    X, y = make_classification(random_state=115)
    X = np.concatenate((X[y==0],X[y==0]), axis=0)
    y = np.concatenate((y[y==0],np.ones((len(y[y==0])))))
    value = px.classification.f1v(X,y)
    assert value==1

def test_F1v_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.f1v(X,y)
    assert 0 <= value <= 1 
    

def test_F1v_one_class():
    X, y = make_classification(random_state=115, weights=[1, 0], flip_y=0)
    value = px.classification.f1v(X,y)
    assert np.isnan(value)

### F2 tests ###

def test_F2():
    metric = px.classification.f2
    assert _get_comparison(metric)>0.5

def test_F2_equal_to_zero():
    X, y = make_classification(random_state=115)
    X[y==0] += 100
    value = px.classification.f2(X,y)
    assert value==0

def test_F2_equal_to_one():
    X, y = make_classification(random_state=115)
    X = np.concatenate((X[y==0],X[y==0]), axis=0)
    y = np.concatenate((y[y==0],np.ones((len(y[y==0])))))
    value = px.classification.f2(X,y)
    assert value==1

def test_F2_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.f2(X,y)
    assert 0 <= value <= 1 

### F3 tests ###

def test_F3():
    metric = px.classification.f3
    assert _get_comparison(metric)>0.5

def test_F3_equal_to_zero():
    X, y = make_classification(random_state=115)
    X[y==0] += 100
    value = px.classification.f3(X,y)
    assert value==0

def test_F3_equal_to_one():
    X, y = make_classification(random_state=115)
    X = np.concatenate((X[y==0],X[y==0]), axis=0)
    y = np.concatenate((y[y==0],np.ones((len(y[y==0])))))
    value = px.classification.f3(X,y)
    assert value==1

def test_F3_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.f3(X,y)
    assert 0 <= value <= 1 

### F4 tests ###

def test_F4():
    metric = px.classification.f4
    assert _get_comparison(metric)>0.5

def test_F4_equal_to_zero():
    X, y = make_classification(random_state=115)
    X[y==0] += 100
    value = px.classification.f4(X,y)
    assert value==0

def test_F4_equal_to_one():
    X, y = make_classification(random_state=115)
    X = np.concatenate((X[y==0],X[y==0]), axis=0)
    y = np.concatenate((y[y==0],np.ones((len(y[y==0])))))
    value = px.classification.f4(X,y)
    assert value==1

def test_F4_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.f4(X,y)
    assert 0 <= value <= 1 

### Test L1 ###

def test_L1():
    metric = px.classification.l1
    assert _get_comparison(metric)>0.5

def test_L1_equal_to_zero():
    X, y = make_classification(random_state=115)
    X[y==0] += 100
    value = px.classification.l1(X,y)
    assert value==0

def test_L1_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.l1(X,y)
    assert 0 <= value <= 1 

### Test L2 ###

def test_L2():
    metric = px.classification.l2
    assert _get_comparison(metric)>0.5

def test_L2_equal_to_zero():
    X, y = make_classification(random_state=115)
    X[y==0] += 100
    value = px.classification.l2(X,y)
    assert value==0

def test_L2_random_classifier_accuracy():
    X, y = make_classification(random_state=115)
    X = np.concatenate((X[y==0],X[y==0]), axis=0)
    y = np.concatenate((y[y==0],np.ones((len(y[y==0])))))
    value = px.classification.l2(X,y)
    assert value==0.5

def test_L2_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.l2(X,y)
    assert 0 <= value <= 1 

### Test L3 ###

def test_L3():
    metric = px.classification.l3
    assert _get_comparison(metric)>0.5

def test_L3_equal_to_zero():
    X, y = make_classification(random_state=115)
    X[y==0] += 100
    value = px.classification.l3(X,y)
    assert value==0

def test_L3_random_classifier_accuracy():
    X, y = make_classification(random_state=115)
    X = np.concatenate((X[y==0],X[y==0]), axis=0)
    y = np.concatenate((y[y==0],np.ones((len(y[y==0])))))
    value = px.classification.l3(X,y)
    assert 0.4 <= value <= 0.6

def test_L3_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.l3(X,y)
    assert 0 <= value <= 1

### Test N1 ###

def test_N1():
    metric = px.classification.n1
    assert _get_comparison(metric)>0.5

def test_N1_close_to_zero():
    X, y = make_classification(random_state=115)
    X[y==0] += 100
    value = px.classification.n1(X,y)
    assert value<0.01

def test_N1_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.n1(X,y)
    assert 0 <= value <= 1 

### Test N2 ###

def test_N2():
    metric = px.classification.n2
    assert _get_comparison(metric)>0.5

def test_N2_close_to_zero():
    X, y = make_classification(random_state=115)
    X[y==0] += 1000
    value = px.classification.n2(X,y)
    assert value<0.01

def test_N2_equal_to_half():
    X, y = make_classification(random_state=115)
    X = np.concatenate((X[y==0],X[y==0]), axis=0)
    y = np.concatenate((y[y==0],np.ones((len(y[y==0])))))
    value = px.classification.n2(X,y)
    assert value==0.5

def test_N2_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.n2(X,y)
    assert 0 <= value <= 1 

def test_N2_single_class():
    X, y = make_classification(random_state=115, weights=[1., .0], flip_y=0.)
    value = px.classification.n2(X,y)
    assert np.isnan(value)

def test_N2_emptyset():
    X, y = [], []
    value = px.classification.n2(X,y)
    assert np.isnan(value)

### Test N3 ###

def test_N3():
    metric = px.classification.n3
    assert _get_comparison(metric)>0.5

def test_N3_equal_to_zero():
    X, y = make_classification(random_state=115)
    X[y==0] += 100
    value = px.classification.n3(X,y)
    assert value==0

def test_N3_equal_to_one():
    X, y = make_classification(random_state=115)
    X = np.concatenate((X[y==0],X[y==0]), axis=0)
    y = np.concatenate((y[y==0],np.ones((len(y[y==0])))))
    value = px.classification.n3(X,y)
    assert value==1

def test_N3_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.n3(X,y)
    assert 0 <= value <= 1

### Test N4 ###

def test_N4():
    metric = px.classification.n4
    assert _get_comparison(metric)>0.5

def test_N4_equal_to_zero():
    X, y = make_classification(random_state=115)
    X[y==0] += 100
    value = px.classification.n4(X,y)
    assert value==0

def test_N4_random_classifier_accuracy():
    X, y = make_classification(random_state=115)
    X = np.concatenate((X[y==0],X[y==0]), axis=0)
    y = np.concatenate((y[y==0],np.ones((len(y[y==0])))))
    value = px.classification.n4(X,y)
    assert 0.4 <= value <= 0.6

def test_N4_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.n4(X,y)
    assert 0 <= value <= 1

### Test T1 ###

def test_T1():
    metric = px.classification.t1
    assert _get_comparison(metric)>0.5

def test_T1_close_to_zero():
    X, y = make_classification(random_state=115)
    X[y==0] += 100
    value = px.classification.t1(X,y)
    assert value<=0.05

def test_T1_equal_to_one():
    X, y = make_classification(random_state=115)
    X = np.concatenate((X[y==0],X[y==0]), axis=0)
    y = np.concatenate((y[y==0],np.ones((len(y[y==0])))))    
    value = px.classification.t1(X,y)
    assert value==1

def test_T1_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.t1(X,y)
    assert 0 <= value <= 1

### Test LSC ###

def test_LSC():
    metric = px.classification.lsc
    assert _get_comparison(metric)>0.5

def test_LSC_close_to_zero():
    # depends on imbalance -- for balanced problems minimum is 0.5
    X, y = make_classification(random_state=115, weights=[0.99, 0.01])
    X[y==0] += 100
    value = px.classification.lsc(X,y)
    assert value<0.05

def test_LSC_equal_to_half():
    # depends on imbalance -- for balanced problems minimum is 0.5
    X, y = make_classification(random_state=115)
    X[y==0] += 100
    value = px.classification.lsc(X,y)
    assert value==0.5

def test_LSC_equal_to_one():
    X, y = make_classification(random_state=115)
    X = np.concatenate((X[y==0],X[y==0]), axis=0)
    y = np.concatenate((y[y==0],np.ones((len(y[y==0])))))
    value = px.classification.lsc(X,y)
    assert value==1

def test_LSC_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.lsc(X,y)
    assert 0 <= value <= 1

### Test Density ###

def test_density():
    metric = px.classification.density
    assert _get_comparison(metric)>0.5

def test_density_close_to_zero():
    # depends on imbalance -- for balanced problems minimum is 0.5
    X, y = make_classification(random_state=115, weights=[0.99, 0.01])
    X[y==0] += 100
    value = px.classification.density(X,y)
    assert value<0.05

def test_density_close_to_half():
    # depends on imbalance -- for balanced problems minimum is 0.5
    X, y = make_classification(random_state=115)
    X[y==0] += 100
    value = px.classification.density(X,y)
    assert 0.4 <= value <= 0.6

def test_density_equal_to_one():
    X, y = make_classification(random_state=115)
    X = np.concatenate((X[y==0],X[y==0]), axis=0)
    y = np.concatenate((y[y==0],np.ones((len(y[y==0])))))
    value = px.classification.density(X,y)
    assert value==1

def test_density_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.density(X,y)
    assert 0 <= value <= 1

### Test ClsCoef ###

def test_clsCoef():
    metric = px.classification.clsCoef
    assert _get_comparison(metric)>0.5

def test_clsCoef_equal_to_zero():
    X, y = make_classification(random_state=115)
    X[y==0] += 100
    value = px.classification.clsCoef(X,y)
    assert value==0

def test_clsCoef_equal_to_one():
    X, y = make_classification(random_state=115)
    X = np.concatenate((X[y==0],X[y==0]), axis=0)
    y = np.concatenate((y[y==0],np.ones((len(y[y==0])))))
    value = px.classification.clsCoef(X,y)
    assert value==1

def test_clsCoef_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.clsCoef(X,y)
    assert 0 <= value <= 1

### Test Hubs ###

def test_hubs():
    metric = px.classification.hubs
    assert _get_comparison(metric)>0.5

def test_hubs_equal_to_zero():
    X, y = make_classification(random_state=115)
    X[y==0] += 100
    value = px.classification.hubs(X,y)
    assert value==0

def test_hubs_close_to_one():
    X, y = make_classification(random_state=115, flip_y=0.5)
    value = px.classification.hubs(X,y)
    assert value>=0.95

def test_hubs_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.hubs(X,y)
    assert 0 <= value <= 1

### Test T2 ###

def test_T2():
    metric = px.classification.t2
    assert _get_comparison(metric)>0.5

def test_T2_close_to_zero():
    X, y = make_classification(random_state=115, n_features=2, n_informative=2, n_redundant=0, n_samples=1000)
    value = px.classification.t2(X,y)
    assert value==0.002

def test_T2_equal_to_one():
    X, y = make_classification(random_state=115, n_samples=20)
    value = px.classification.t2(X,y)
    assert value==1

def test_T2_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.t2(X,y)
    assert 0 <= value <= 1

### Test T3 ###

def test_T3():
    metric = px.classification.t3
    assert _get_comparison(metric)>0.5

def test_T3_close_to_zero():
    X, y = make_classification(random_state=115, n_features=20, n_informative=2, n_redundant=0, n_repeated=18, n_samples=1000)
    value = px.classification.t3(X,y)
    assert value==0.002

def test_T3_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.t3(X,y)
    assert 0 <= value <= 1

### Test T4 ###

def test_T4():
    metric = px.classification.t4
    assert _get_comparison(metric)>0.5

def test_T4_close_to_zero():
    X, y = make_classification(random_state=115, n_features=100, n_informative=2, n_redundant=0, n_repeated=98)
    value = px.classification.t4(X,y)
    assert value==0.02

def test_T4_imbalance():
    X, y = make_classification(random_state=115, weights=[.99, .01], flip_y=0.)
    value = px.classification.t4(X,y)
    assert 0 <= value <= 1

### Test C1 ###

def test_C1():
    metric = px.classification.c1
    assert _get_comparison(metric)>0.5

def test_C1_equal_to_zero():
    X, y = make_classification(random_state=115)
    value = px.classification.c1(X,y)
    assert value==0

def test_C1_close_to_one():
    X, y = make_classification(random_state=115, weights=[.999, .001], flip_y=0.)
    value = px.classification.c1(X,y)
    assert value>=0.9

### Test C2 ###

def test_C2():
    metric = px.classification.c2
    assert _get_comparison(metric)>0.5

def test_C2_equal_to_zero():
    X, y = make_classification(random_state=115)
    value = px.classification.c2(X,y)
    assert value==0

def test_C2_close_to_one():
    X, y = make_classification(random_state=115, weights=[.999, .001], flip_y=0.)
    value = px.classification.c2(X,y)
    assert value>=0.95

### Test ComplexityCalculator ###

def test_ComplexityCalculator():
    c = px.ComplexityCalculator()
    reps = 10
    res = np.zeros((reps, 2))
    rs = np.random.randint(100,10000,reps)    

    for r in range(reps):
        X_simple, y_simple = make_classification(n_samples=500, n_features=5, n_redundant=0, 
                        n_informative=5, n_clusters_per_class=2, n_classes=2, flip_y=0, 
                        class_sep=3.5, weights=[0.5, 0.5], random_state=rs[r])

        X_complex, y_complex = make_classification(n_samples=500, n_features=10, n_redundant=0, 
                        n_informative=10, n_clusters_per_class=2, n_classes=2, flip_y=0.2, 
                        class_sep=0.01, weights=[0.9, 0.1], random_state=rs[r])

        res[r, 0] = c.fit(X_simple, y_simple).score()
        res[r, 1] = c.fit(X_complex, y_complex).score()

    comparison = np.mean(res, axis=0)
    assert comparison[0]<comparison[1]
    
def test_ComplexityCalculator_report():
    c = px.ComplexityCalculator()

    X_simple, y_simple = make_classification(n_samples=500, n_features=5, n_redundant=0, 
                        n_informative=5, n_clusters_per_class=2, n_classes=2, flip_y=0, 
                        class_sep=3.5, weights=[0.5, 0.5], random_state=123)

    report = c.fit(X_simple, y_simple).report()
    assert isinstance(report, dict)
    
def test_OVA():
    c = px.ComplexityCalculator(multiclass_strategy='ova')
    reps = 10
    res = np.zeros((reps, 2))
    rs = np.random.randint(100,10000,reps)

    for r in range(reps):
        X_simple, y_simple = make_classification(n_samples=500, n_features=5, n_redundant=0, 
                        n_informative=5, n_clusters_per_class=2, n_classes=3, flip_y=0, 
                        class_sep=3.5, weights=[0.33, 0.33, 0.34], random_state=rs[r])

        X_complex, y_complex = make_classification(n_samples=500, n_features=10, n_redundant=0, 
                        n_informative=10, n_clusters_per_class=2, n_classes=3, flip_y=0.2, 
                        class_sep=0.01, weights=[0.8, 0.1, 0.1], random_state=rs[r])

        res[r, 0] = c.fit(X_simple, y_simple).score()
        res[r, 1] = c.fit(X_complex, y_complex).score()

    comparison = np.mean(res, axis=0)
    assert comparison[0]<comparison[1]
    
def test_OVO():
    c = px.ComplexityCalculator(multiclass_strategy='ovo')
    reps = 10
    res = np.zeros((reps, 2))
    rs = np.random.randint(100,10000,reps)

    for r in range(reps):
        X_simple, y_simple = make_classification(n_samples=500, n_features=5, n_redundant=0, 
                        n_informative=5, n_clusters_per_class=2, n_classes=3, flip_y=0, 
                        class_sep=3.5, weights=[0.33, 0.33, 0.34], random_state=rs[r])

        X_complex, y_complex = make_classification(n_samples=500, n_features=10, n_redundant=0, 
                        n_informative=10, n_clusters_per_class=2, n_classes=3, flip_y=0.2, 
                        class_sep=0.01, weights=[0.8, 0.1, 0.1], random_state=rs[r])

        res[r, 0] = c.fit(X_simple, y_simple).score()
        res[r, 1] = c.fit(X_complex, y_complex).score()

    comparison = np.mean(res, axis=0)
    assert comparison[0]<comparison[1]