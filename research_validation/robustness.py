"""
ROBUSTNESS TESTING MODULE

Test model performance under adversarial noise conditions
"""

import torch
import numpy as np
from evaluate import evaluate


class RobustnessTest:
    """
    Evaluate model robustness to Gaussian noise
    """
    
    def __init__(self, model, data, test_mask, device):
        self.model = model
        self.data = data
        self.test_mask = test_mask
        self.device = device
        
        # Store original features
        self.original_features = data.x.clone()
        
        self.results = {}
    
    def add_gaussian_noise(self, noise_level):
        """
        Add Gaussian noise to features
        
        Args:
            noise_level: standard deviation of noise
        """
        
        noise = torch.randn_like(self.data.x) * noise_level
        self.data.x = self.original_features + noise
        self.data.x = torch.clamp(self.data.x, min=-10, max=10)
    
    def reset_features(self):
        """Restore original features"""
        self.data.x = self.original_features.clone()
    
    def test_noise_robustness(self, evaluate_fn, noise_levels=[0.01, 0.05]):
        """
        Test performance at different noise levels
        
        Args:
            evaluate_fn: function to compute metrics
            noise_levels: list of noise standard deviations
        """
        
        print("\nRobustness Testing (Gaussian Noise):")
        print("-" * 60)
        
        # Baseline (no noise)
        baseline_metrics = evaluate_fn(self.model, self.data, self.test_mask)
        baseline_f1 = baseline_metrics['f1']
        
        print(f"Baseline (σ=0.00):    F1 = {baseline_f1:.4f}")
        self.results['baseline'] = {
            'noise_level': 0.0,
            'f1': baseline_f1
        }
        
        # Test with noise
        for noise_level in noise_levels:
            self.add_gaussian_noise(noise_level)
            
            metrics = evaluate_fn(self.model, self.data, self.test_mask)
            f1_degraded = metrics['f1']
            
            degradation = baseline_f1 - f1_degraded
            degradation_pct = (degradation / baseline_f1) * 100
            
            print(f"Noise (σ={noise_level:.2f}):      F1 = {f1_degraded:.4f} "
                  f"(degradation: {degradation_pct:+.2f}%)")
            
            self.results[f'noise_{noise_level}'] = {
                'noise_level': noise_level,
                'f1': f1_degraded,
                'degradation': degradation,
                'degradation_pct': degradation_pct
            }
        
        # Restore
        self.reset_features()
        
        return self.results
    
    def get_summary(self):
        """Summarize robustness"""
        
        baseline_f1 = self.results['baseline']['f1']
        
        summary = {
            'baseline_f1': baseline_f1,
            'robustness_results': self.results
        }
        
        # Average degradation
        degradation_list = [
            v['degradation_pct'] for k, v in self.results.items() 
            if k != 'baseline'
        ]
        
        if degradation_list:
            summary['avg_degradation_pct'] = np.mean(degradation_list)
        
        return summary
