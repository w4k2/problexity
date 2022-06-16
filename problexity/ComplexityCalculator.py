from .feature_based import F1, F1v, F2, F3, F4
from .linearity import L1, L2, L3
from .neighborhood import N1, N2, N3, N4, T1, LSC
from .network import density, clsCoef, hubs
from .dimensionality import  T2, T3, T4
from .class_imbalance import C1, C2
import numpy as np

class ComplexityCalculator:
    def __init__(self, metrics=[F1, F1v, F2, F3, F4, L1, L2, L3, N1, N2, N3, N4, T1, LSC, density, clsCoef, hubs, T2, T3, T4, C1, C2]):
        self.metrics=metrics

    def calculate(self, X, y):
        complexity = []
        for m in self.metrics:
            complexity.append(m(X, y))
        return complexity

    def score(self, X, y):
        return np.mean(self.calculate(X, y))