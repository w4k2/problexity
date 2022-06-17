from .feature_based import F1, F1v, F2, F3, F4
from .linearity import L1, L2, L3
from .neighborhood import N1, N2, N3, N4, T1, LSC
from .network import density, clsCoef, hubs
from .dimensionality import T2, T3, T4
from .class_imbalance import C1, C2
import numpy as np

C_METRICS = [F1, F1v, F2, F3, F4, L1, L2, L3, N1, N2, N3, N4,
             T1, LSC, density, clsCoef, hubs, T2, T3, T4, C1, C2]
C_COLORS = ['#FD0100', '#F76915', '#EEDE04', '#A0D636', '#2FA236', '#333ED4']
C_RANGES = {'FB': 5, 'LR': 3, 'NB': 6,
            'NE': 3, 'DM': 3, 'CI': 2}
class ComplexityCalculator:
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
        # Do basic calculations
        self.n_samples, self.n_features = X.shape
        self.classes, self.prior_probability = np.unique(y, return_counts=True)
        self.prior_probability = self.prior_probability / self.n_samples
        self.n_classes = len(self.classes)
        
        # Calculate complexity
        self.complexity = [m(X, y) for m in self.metrics]
        return self
    
    def report(self, precision=3):
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
