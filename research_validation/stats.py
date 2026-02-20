"""
STATISTICAL ANALYSIS MODULE

Paired t-tests and statistical validation
"""

import numpy as np
from scipy import stats
import json
from pathlib import Path


class StatisticalAnalysis:
    """
    Compute and format statistical test results
    """
    
    def __init__(self, results_dict):
        """
        Args:
            results_dict: {model_name: [list of test F1 values]}
        """
        self.results = results_dict
        self.test_results = {}
    
    def compute_stats(self, model_name):
        """Compute mean, std, min, max for a model"""
        values = self.results[model_name]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    
    def paired_ttest(self, model_a, model_b):
        """
        Paired t-test: does model_a significantly differ from model_b?
        
        Returns:
            dict with t-statistic, p-value, significance
        """
        
        values_a = self.results[model_a]
        values_b = self.results[model_b]
        
        t_stat, p_value = stats.ttest_rel(values_a, values_b)
        
        return {
            'model_a': model_a,
            'model_b': model_b,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'mean_diff': float(np.mean(values_a) - np.mean(values_b))
        }
    
    def run_all_tests(self):
        """Run all key comparisons"""
        
        comparisons = [
            ('qigat', 'baseline_128'),
            ('qigat', 'residual_expansion'),
            ('qigat', 'mlp_control')
        ]
        
        for model_a, model_b in comparisons:
            key = f"{model_a}_vs_{model_b}"
            self.test_results[key] = self.paired_ttest(model_a, model_b)
    
    def print_summary(self):
        """Print formatted statistical summary"""
        
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS - 5 SEED VALIDATION")
        print("="*80)
        
        # Summary statistics
        print("\nMEAN ± STD (Test F1):")
        print("-" * 80)
        
        for model_name in sorted(self.results.keys()):
            stats_dict = self.compute_stats(model_name)
            print(f"{model_name:25s}:  {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}  "
                  f"(range: {stats_dict['min']:.4f} - {stats_dict['max']:.4f})")
        
        # Paired t-tests
        print("\n" + "="*80)
        print("PAIRED T-TESTS (α = 0.05)")
        print("="*80)
        
        for key, result in self.test_results.items():
            sig = "SIGNIFICANT **" if result['significant'] else "NOT significant"
            print(f"\n{result['model_a'].upper()} vs {result['model_b'].upper()}")
            print(f"  t-statistic: {result['t_statistic']:8.4f}")
            print(f"  p-value:     {result['p_value']:8.6f}")
            print(f"  Mean diff:   {result['mean_diff']:8.4f}")
            print(f"  Result:      {sig}")
    
    def to_dict(self):
        """Convert all results to dict for JSON serialization"""
        
        output = {
            'statistics': {},
            'paired_tests': self.test_results
        }
        
        for model_name in self.results.keys():
            stats_dict = self.compute_stats(model_name)
            output['statistics'][model_name] = {
                'mean': float(stats_dict['mean']),
                'std': float(stats_dict['std']),
                'min': float(stats_dict['min']),
                'max': float(stats_dict['max']),
                'values': [float(v) for v in stats_dict['values']]
            }
        
        return output
    
    def save_json(self, filepath):
        """Save results to JSON"""
        
        data = self.to_dict()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n[Saved] Statistical results to {filepath}")


def interpret_significance(p_value):
    """Interpret p-value"""
    if p_value < 0.001:
        return "Highly significant (p < 0.001)"
    elif p_value < 0.01:
        return "Very significant (p < 0.01)"
    elif p_value < 0.05:
        return "Significant (p < 0.05)"
    else:
        return "Not significant (p >= 0.05)"
