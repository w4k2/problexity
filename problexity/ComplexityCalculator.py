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
    
    A class that allows to determine all or selected metrics for a given data set. The report can be returned both as a simple vector of metrics, as well as a dictionary containing all set parameters and visualization in the form of a radar.

    :type metrics: list, optional (default=all the metrics avalable in problexity)
    :param metrics: List of classification complexity measures used to validate a given set.
    :type ranges: dict, optional (default=all the default six groups of metrics)
    :param ranges: Configuration of radar visualisation, allowing to group metrics by color.
    :type colors: list, optional (default=six-color palette)
    :param colors: List of colors assigned to groups on radar visualisation.

    :vartype complexity: list
    :var complexity: The list of all the scores acquired with metrics defined by metrics list.
    :vartype n_samples: int
    :var n_samples: The number of samples in the fitted dataset.
    :vartype n_features: int
    :var n_features: The number of features of the fitted dataset.
    :vartype n_classes: int
    :var n_classes: The number of classes in the fitted dataset.
    :vartype classes: array-like, shape (n_classes, )
    :var classes: The class labels.
    :vartype prior_probability: array-like, shape (n_classes, )
    :var prior_probability: The prior probability of classes.

    :Examples:

    >>> from problexity import ComplexityCalculator
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification()
    >>> cc = ComplexityCalculator().fit(X, y)
    >>> print(cc.complexity)
    [0.3158144010174404, 0.1508882806154997, 0.005974480517635054, 0.57, 0.0, 
     0.10518058962953956, 0.1, 0.07, 0.135, 0.48305940839428635, 0.27, 0.11, 
     1.0, 0.9642, 0.9892929292929293, 0.9321428571428572, 0.9297111755529109, 
     0.2, 0.16, 0.8, 0.0, 0.0]
    >>> report = cc.report()
    >>> print(report)
    {
        'n_samples': 100, 'n_features': 20, 'n_classes': 2, 
        'classes': array([0, 1]), 
        'prior_probability': array([0.5, 0.5]), 
        'score': 0.377, 
        'complexities': 
        {
            'f1': 0.316, 'f1v': 0.151, 'f2': 0.006, 'f3': 0.57, 'f4': 0.0, 
            'l1': 0.105, 'l2': 0.1, 'l3': 0.07, 
            'n1': 0.135, 'n2': 0.483, 'n3': 0.27, 'n4': 0.11, 't1': 1.0, 'lsc': 0.964, 
            'density': 0.989, 'clsCoef': 0.932, 'hubs': 0.93, 
            't2': 0.2, 't3': 0.16, 't4': 0.8, 'c1': 0.0, 'c2': 0.0
        }
    }
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
        Calculates metrics for given dataset.

        :type X: array-like, shape (n_samples, n_features)
        :param X: The training input samples.
        :type y: array-like, shape (n_samples, )
        :param y: The training input labels.

        :rtype: ComplexityCalculator class object
        :returns: ComplexityCalculator class object.
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
    
    def _metrics(self):
        return [metric.__name__ for metric in self.metrics]
    
    def report(self, precision=3):
        """Returns report of problem complexity

        :type precision: int, optional (default=3)
        :param precision: The rounding precision.

        :rtype: dict
        :returns: Dictionary with complexity report
        
        :Examples:

        >>> from problexity import ComplexityCalculator
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification()
        >>> cc = ComplexityCalculator().fit(X, y)
        >>> report = cc.report()
        >>> print(report)
        {
            'n_samples': 100, 'n_features': 20, 'n_classes': 2, 
            'classes': array([0, 1]), 
            'prior_probability': array([0.5, 0.5]), 
            'score': 0.377, 
            'complexities': 
            {
                'f1': 0.316, 'f1v': 0.151, 'f2': 0.006, 'f3': 0.57, 'f4': 0.0, 
                'l1': 0.105, 'l2': 0.1, 'l3': 0.07, 
                'n1': 0.135, 'n2': 0.483, 'n3': 0.27, 'n4': 0.11, 't1': 1.0, 'lsc': 0.964, 
                'density': 0.989, 'clsCoef': 0.932, 'hubs': 0.93, 
                't2': 0.2, 't3': 0.16, 't4': 0.8, 'c1': 0.0, 'c2': 0.0
            }
        }
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
        """Returns integrated score of problem complexity

        :type weights: array-like, optional (default=None), shape (n_metrics) 
        :param weights: Optional weights of metrics.

        :rtype: float
        :returns: Single score for integrated metrics
        """
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
        """Returns integrated score of problem complexity

        :type weights: matplotlib figure object 
        :param weights: Figure to draw radar on.
        :type spec: tuple, optional (default=(1,1,1)) 
        :param spec: Matplotlib subplot location.
        
        :rtype: object
        :returns: Matplotlib axis object.

        .. image:: plots/radar.png

        """
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
        scale = 1.
        
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
                ax.text(np.mean(grv*scale),1.2,
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
                fontsize=12,
                color='#333')
        
        return ax
    
    def _check_is_fitted(self):
        if not hasattr(self, 'complexity'):
            raise Exception('Measures not calculated, please call fit() first.')
