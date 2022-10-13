from . import classification as c
from . import regression as r
import numpy as np

C_METRICS = [c.f1, c.f1v, c.f2, c.f3, c.f4, c.l1, c.l2, c.l3, c.n1, c.n2, c.n3, c.n4,
             c.t1, c.lsc, c.density, c.clsCoef, c.hubs, c.t2, c.t3, c.t4, c.c1, c.c2]
C_COLORS = ['#FD0100', '#F76915', '#EEDE04', '#A0D636', '#2FA236', '#333ED4']
C_RANGES = {'FB': 5, 'LR': 3, 'NB': 6,
            'NE': 3, 'DM': 3, 'CI': 2}
C_WEIGHTS = np.ones((22))

R_METRICS = [r.c1, r.c2, r.c3, r.c4, r.l1, r.l2, r.s1, r.s2, r.s3, r.l3, r.s4, r.t2]
R_COLORS = ['#FD0100', '#F76915', '#EEDE04', '#A0D636']
R_RANGES = {'FC': 4, 'LR': 2, 'SM': 3, 'GT': 3} 
R_WEIGHTS = np.ones((12))
R_WEIGHTS[[0,1,-1]]=-1

MULTICLASS_STRATEGIES = ['ova', 'ovo'] 
MODES = ['classification', 'regression']

class ComplexityCalculator:
    """
    Complexity Calculator Class.
    
    A class that allows to determine all or selected metrics for a given data set. The report can be returned both as a simple vector of metrics, as well as a dictionary containing all set parameters and visualization in the form of a radar.

    :type metrics: list, optional (default=all the metrics avalable in problexity)
    :param metrics: List of classification complexity measures used to validate a given set.
    :type mode: string, optional (default=classification)
    :param mode: Recognition task for which metrics should be calculated. Might be selected between `classification` and `regression`.
    :type multiclass_strategy: string, optional (default=ova)
    :param multiclass_strategy: Strategy used for multiclass metric integration. Might be selected between `ova` and `ovo`.
    :type ranges: dict, optional (default=all the default six groups of metrics)
    :param ranges: Configuration of radar visualisation, allowing to group metrics by color.
    :type colors: list, optional (default=six-color palette)
    :param colors: List of colors assigned to groups on radar visualisation.
    :type weights: list, optional (default=list of weights, where weight are equal to 1 for all measures where simpler problems have smaller value, otherwise -1)
    :param weights: List of weights taken into account in score() procedure.

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
    def __init__(self, metrics=None, colors=None, ranges=None, weights=None, mode='classification', multiclass_strategy='ovo'):
        # Initlialize configuration
        if mode not in MODES:
            raise Exception('Unsupported mode %s, must be from list %s.' % (
                mode,
                ', '.join(MODES)
            ))
        else:
            self.mode = mode
        
        # Set default metrics, colors and ranges based on mode, if not provided
        if None not in [metrics, colors, ranges, weights]:
            self.metrics = metrics
            self.colors = colors
            self.ranges = ranges
            self.weights = weights
        else:
            if self.mode == 'regression':
                self.metrics = R_METRICS
                self.colors = R_COLORS
                self.ranges = R_RANGES
                self.weights = R_WEIGHTS
            else:
                self.metrics = C_METRICS
                self.colors = C_COLORS
                self.ranges = C_RANGES
                self.weights = C_WEIGHTS
        
        # Validate test configuration
        rsum = np.sum([self.ranges[k] for k in self.ranges])
        if len(self.ranges) != len(self.colors):
            raise Exception('Number of ranges and colors does not match.')
        if rsum != len(self.metrics):
            raise Exception('Ranges does not sum with number of metrics.')
        if multiclass_strategy not in MULTICLASS_STRATEGIES:
            raise Exception('Unsupported multiclass_strategy %s, must be from list %s.' % (
                multiclass_strategy,
                ', '.join(MULTICLASS_STRATEGIES)
            ))
        else:
            self.multiclass_strategy = multiclass_strategy
        
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
        if self.mode == 'classification':
            self.classes, self.prior_probability = np.unique(y, return_counts=True)
            self.prior_probability = self.prior_probability / self.n_samples
            self.n_classes = len(self.classes)
        
        # Calculate complexity
        # Binary case
        if (self.mode == 'regression') or (self.n_classes == 2):
            self.complexity = [m(X, y) for m in self.metrics]
        else:
            # Prepare container for complexities
            complexities = []
            
            
            if self.multiclass_strategy == 'ova':
                # OVA
                for i in self.classes:
                    complexities.append([m(X, (y == i).astype(int)) for m in self.metrics])
            elif self.multiclass_strategy == 'ovo':
                # OVO
                for i in self.classes:
                    for j in self.classes:
                        if i!= j:
                            mask = (y==i) + (y==j)
                            _y = (y[mask] == j).astype(int)
                            complexities.append([m(X[mask], _y) for m in self.metrics])
              
            # Calculate mean complexity                  
            self.complexity = np.mean(np.array(complexities), axis=0).tolist()
            
            # Establish complexity deviation
            self.complexity_std = np.std(np.array(complexities), axis=0).tolist()            
            
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
            'score': np.around(self.score(), precision),
            'complexities': {}
        }
        if self.mode == 'classification':
            report.update({'n_classes': self.n_classes,
                           'classes': self.n_classes,
                           'prior_probability': self.prior_probability})

            if self.n_classes > 2:
                report.update({'complexities_std': {}})

        if (self.mode == 'regression') or (self.n_classes == 2):
            for metric, score in zip(self.metrics, self.complexity):
                report['complexities'].update({
                    metric.__name__: np.around(score,precision)
                })
        else:
            for metric, score, std in zip(self.metrics, self.complexity, self.complexity_std):
                report['complexities'].update({
                    metric.__name__: np.around(score,precision)
                })
                report['complexities_std'].update({
                    metric.__name__: np.around(std,precision)
                })

        return report

    def score(self):
        """Returns integrated score of problem complexity

        :type weights: array-like, optional (default=None), shape (n_metrics) 
        :param weights: Optional weights of metrics.

        :rtype: float
        :returns: Single score for integrated metrics
        """
        self._check_is_fitted()  

        # Check length of weights vector
        if len(self.weights) != len(self.metrics):
            raise Exception('Mismatch between number of metrics and number of weights.')

        # Normalize weights
        self.weights = np.array(self.weights) / np.sum(self.weights)
        
        # Calculate weighted score
        return np.sum(self.weights * self.complexity)
       
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
        ax.text(-.15, -.15, ('%.1f' if self.mode=='classification' else '%.2f') % (self.score()*(100 if self.mode=='classification' else 1)),
                ha='center',
                va='center',
                fontsize=12,
                color='#333')
        
        return ax
    
    def _check_is_fitted(self):
        if not hasattr(self, 'complexity'):
            raise Exception('Measures not calculated, please call fit() first.')
