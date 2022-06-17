from .feature_based import f1, f1v, f2, f3, f4
from .linearity import l1, l2, l3
from .neighborhood import n1, n2, n3, n4, t1, lsc
from .network import density, clsCoef, hubs
from .dimensionality import t2, t3, t4
from .class_imbalance import c1, c2
import numpy as np

C_METRICS = [f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4,
             t1, lsc, density, clsCoef, hubs, t2, t3, t4, c1, c2]
C_COLORS = ['#FD0100', '#F76915', '#EEDE04', '#A0D636', '#2FA236', '#333ED4']
C_RANGES = {'FB': 5, 'LR': 3, 'NB': 6,
            'NE': 3, 'DM': 3, 'CI': 2}
class ComplexityCalculator:
    """
    Complexity Calculator Class.

    The method was inspired by Accuracy Weighted Ensemble (AWE) algorithm to which it introduces two main modifications: (I) classifier weights depend on the individual classifier accuracies and time they have been spending in the ensemble, (II) individual classifier are chosen on the basis on the non-pairwise diversity measure.

    :type base_estimator: ClassifierMixin class object
    :param base_estimator: Classification algorithm used as a base estimator.
    :type n_estimators: integer, optional (default=10)
    :param  n_estimators: The maximum number of estimators trained using consecutive data chunks and maintained in the ensemble.
    :type theta: float, optional (default=0.1)
    :param theta: Threshold for weight calculation method and aging procedure control.
    :type post_pruning: boolean, optional (default=False)
    :param post_pruning: Whether the pruning is conducted before or after adding the classifier.
    :type pruning_criterion: string, optional (default='accuracy')
    :param pruning_criterion: Selection of pruning criterion.
    :type weight_calculation_method: string, optional (default='kuncheva')
    :param weight_calculation_method: same_for_each, proportional_to_accuracy, kuncheva, pta_related_to_whole, bell_curve,
    :type aging_method: string, optional (default='weights_proportional')
    :param aging_method: weights_proportional, constant, gaussian.
    :type rejuvenation_power: float, optional (default=0.0)
    :param rejuvenation_power: Rejuvenation dynamics control of classifiers with high prediction accuracy.

    :vartype ensemble_: list of classifiers
    :var ensemble_: The collection of fitted sub-estimators.
    :vartype classes_: array-like, shape (n_classes, )
    :var classes_: The class labels.
    :vartype weights_: array-like, shape (n_estimators, )
    :var weights_: Classifier weights.

    :Examples:

    >>> import strlearn as sl
    >>> from sklearn.naive_bayes import GaussianNB
    >>> stream = sl.streams.StreamGenerator()
    >>> clf = sl.ensembles.WAE(GaussianNB())
    >>> ttt = sl.evaluators.TestThenTrain(
    >>> metrics=(sl.metrics.balanced_accuracy_score))
    >>> ttt.process(stream, clf)
    >>> print(ttt.scores)
    [[[0.91386218]
      [0.93032581]
      [0.90907219]
      [0.90544872]
      [0.90466186]
      [0.91956783]
      [0.90776942]
      [0.92685422]
      [0.92895186]
      ...
    """
    def __init__(self, metrics=C_METRICS, colors=C_COLORS, ranges=C_RANGES):
        # Initlialize configuration
        self.metrics = metrics
        self.colors = colors
        self.ranges = ranges
        
        # Validate test configuration
        rsum = np.sum([self.ranges[k] for k in ranges])
        if len(self.ranges) != len(self.colors):
            raise Exception('Number of ranges and colors does not match.')
        if rsum != len(metrics):
            raise Exception('Ranges does not sum with number of metrics.')
        
        # Get all the metric names
        self.cnames = [m.__name__ for m in self.metrics]   
        
    def fit(self, X, y):
        """
        Predict classes for X.

        :type X: array-like, shape (n_samples, n_features)
        :param X: The training input samples.

        :rtype: array-like, shape (n_samples, )
        :returns: The predicted classes.
        """

        # Check is fit had been called
        # Do basic calculations
        self.n_samples, self.n_features = X.shape
        self.classes, self.prior_probability = np.unique(y, return_counts=True)
        self.prior_probability = self.prior_probability / self.n_samples
        self.n_classes = len(self.classes)
        
        # Calculate complexity
        self.complexity = [m(X, y) for m in self.metrics]
        return self
    
    def report(self, precision=3):
        """Returns minority and majority data

        :type X: array-like, shape (n_samples, n_features)
        :param X: The training input samples.
        :type y: array-like, shape  (n_samples)
        :param y: The target values.

        :rtype: tuple (array-like, shape = [n_samples, n_features], array-like, shape = [n_samples, n_features])
        :returns: Tuple of minority and majority class samples
        """
        self._check_is_fitted()
        report = {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'classes': self.classes,
            'prior_probability': self.prior_probability,
            'score': np.around(self.score(), precision),
            'complexities': {}
        }
        for metric, score in zip(self.metrics, self.complexity):
            report['complexities'].update({
                metric.__name__: np.around(score,precision)
            })
        return report

    def score(self, weights=None):
        self._check_is_fitted()  
        if weights is not None:
            # Check length of weights vector
            if len(weights) != len(self.metrics):
                raise Exception('Mismatch between number of metrics and number of weights.')

            # Normalize weights
            self.weights = np.array(weights) / np.sum(weights)
            
            # Calculate weighted score
            return np.sum(self.weights * self.complexity)
        else:            
            return np.mean(self.complexity)
       
    def plot(self, figure, spec=(1,1,1)):
        # Establish location
        ax = figure.add_subplot(*spec, projection='polar') 
        
        # Prepare calculation values
        count = 0
        index = 0
        aid = 0
        
        # Group angles
        g_angles = np.linspace(0, np.pi*2, len(self.ranges)+1)

        # Major and minor ticks
        b_ticks = np.linspace(0,1,5)
        s_ticks = np.linspace(0,1,9)
        
        # Select scale
        scale = .75
        
        # Iterate groups
        for ridx, rname in enumerate(self.ranges):
            l = self.ranges[rname]
            c = self.colors[ridx]
            
            v = self.complexity[count:count+l]
            
            # Iterate cv up
            count += l
            index += (l-1)
            
            # Iterate leading angle
            g_a, g_b = g_angles[ridx], g_angles[ridx+1]
            g_r = np.linspace(g_a, g_b, l+1)
                        
            # Iterate metrics
            for vidx, val in enumerate(v):
                # Clip value to range
                cval = np.clip(val, 0, 1)
                grv = g_r[vidx:vidx+2]
                
                # Plot metric region
                ax.fill_between(grv*scale, [-.0,-.0], [cval,cval],
                                color=self.colors[ridx],
                                alpha=1, lw=0)
                
                # Plot baseline
                ax.plot([g_r[vidx]*scale,
                        g_r[vidx]*scale], 
                        [0,1], 
                        c='white')
                
                # Iterate major y ticks
                for tick in b_ticks:
                    ax.plot(grv*scale, [tick, tick], 
                            color='white',
                            lw=1)
                    
                # Iterate minor y ticks
                for tick in s_ticks:
                    ax.plot(grv*scale, [tick, tick], 
                            color='white',
                            lw=.5)
                        
                # Leading metric lines                
                ax.plot(np.ones(b_ticks.shape)*np.mean(grv*scale),
                        b_ticks,
                        color=c,
                        zorder=1, 
                        alpha=.25)
                
                # Pinhead
                ax.scatter([np.mean(grv*scale)],[1], color=c, zorder=10)
                
                # Value label
                ax.text(np.mean(grv*scale),1.1,
                    '%s\n%.2f' % (self.cnames[aid], val),
                    color=c,
                    ha='center',
                    va='center')
                
                # Iterate metric pointer
                aid += 1

        # Plot cleanup
        ax.spines['polar'].set_visible(False)
        ax.spines['start'].set_visible(False)
        ax.spines['end'].set_visible(False)
        ax.spines['inner'].set_visible(False)
        ax.grid(lw=0)
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.set_ylim(-.15,1.25)
        ax.text(-.15, -.15, '%.1f' % (self.score()*100),
                ha='center',
                va='center',
                fontsize=18,
                color='#333')
    
    def _check_is_fitted(self):
        if not hasattr(self, 'complexity'):
            raise Exception('Measures not calculated, please call fit() first.')
