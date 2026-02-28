"""Variance-driven adaptive quantization for water quality monitoring.

Implements the adaptive quantization mechanism described in the paper,
which dynamically adjusts bit-width based on data variance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
import time


class VarianceEstimator:
    """Estimates data variance over sliding windows for quantization policy."""
    
    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        self.buffer = []
        
    def update(self, data: np.ndarray) -> float:
        """Update buffer and compute variance.
        
        Args:
            data: New data point(s) of shape (features,)
            
        Returns:
            Current variance estimate
        """
        self.buffer.append(data)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
            
        if len(self.buffer) < 2:
            return 0.0
            
        # Compute variance across features and time
        arr = np.array(self.buffer)
        return float(np.std(arr))


class AdaptiveQuantizer:
    """Adaptive quantization with variance-driven bit-width selection.
    
    Quantization policy:
    - σ < 0.05: 4-bit (stable conditions)
    - 0.05 ≤ σ < 0.15: 6-bit (moderate variance)
    - σ ≥ 0.15: 8-bit (high variance/pollution events)
    """
    
    def __init__(self, window_size: int = 24, 
                 thresholds: Tuple[float, float] = (0.05, 0.15)):
        self.variance_estimator = VarianceEstimator(window_size)
        self.low_threshold, self.high_threshold = thresholds
        
        # Statistics
        self.bit_width_history = []
        self.variance_history = []
        self.power_consumption_history = []
        
    def select_bit_width(self, data: np.ndarray) -> int:
        """Select quantization bit-width based on current variance.
        
        Args:
            data: Current data window
            
        Returns:
            Selected bit-width (4, 6, or 8)
        """
        variance = self.variance_estimator.update(data)
        self.variance_history.append(variance)
        
        if variance < self.low_threshold:
            bit_width = 4
        elif variance < self.high_threshold:
            bit_width = 6
        else:
            bit_width = 8
            
        self.bit_width_history.append(bit_width)
        return bit_width
    
    def quantize_tensor(self, tensor: torch.Tensor, bit_width: int) -> torch.Tensor:
        """Quantize tensor to specified bit-width using symmetric quantization.
        
        Args:
            tensor: Float tensor to quantize
            bit_width: Target bit-width (4, 6, or 8)
            
        Returns:
            Quantized tensor (still in float format for compatibility)
        """
        # Symmetric quantization
        n_levels = 2 ** bit_width
        max_val = torch.max(torch.abs(tensor))
        
        if max_val == 0:
            return tensor
            
        scale = max_val / (n_levels / 2 - 1)
        
        # Quantize
        quantized = torch.round(tensor / scale)
        quantized = torch.clamp(quantized, -(n_levels / 2), (n_levels / 2) - 1)
        
        # Dequantize (for inference in float)
        dequantized = quantized * scale
        
        return dequantized
    
    def quantize_model_weights(self, model: nn.Module, bit_width: int) -> None:
        """Apply quantization to model weights in-place.
        
        Args:
            model: PyTorch model
            bit_width: Target bit-width
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name:
                    param.data = self.quantize_tensor(param.data, bit_width)
    
    def estimate_power_consumption(self, bit_width: int, base_power: float = 0.45) -> float:
        """Estimate power consumption for given bit-width.
        
        Args:
            bit_width: Current bit-width
            base_power: Baseline power consumption (W) for FP32
            
        Returns:
            Estimated power consumption (W)
        """
        # Power scaling based on bit-width (empirical model)
        scaling_factors = {4: 0.38, 6: 0.47, 8: 0.62}
        power = base_power * scaling_factors.get(bit_width, 1.0)
        
        self.power_consumption_history.append(power)
        return power
    
    def get_statistics(self) -> Dict:
        """Return quantization statistics."""
        if not self.bit_width_history:
            return {}
            
        bit_widths = np.array(self.bit_width_history)
        return {
            'avg_bit_width': float(np.mean(bit_widths)),
            'bit_width_distribution': {
                '4-bit': float(np.mean(bit_widths == 4) * 100),
                '6-bit': float(np.mean(bit_widths == 6) * 100),
                '8-bit': float(np.mean(bit_widths == 8) * 100)
            },
            'avg_variance': float(np.mean(self.variance_history)),
            'avg_power': float(np.mean(self.power_consumption_history)),
            'total_samples': len(self.bit_width_history)
        }


class MixedPrecisionQuantizer:
    """Layer-wise mixed-precision quantization.
    
    Assigns different bit-widths to different layers based on sensitivity.
    Early CNN layers get higher precision, deeper TCN layers get lower precision.
    """
    
    def __init__(self, base_bit_width: int = 8):
        self.base_bit_width = base_bit_width
        self.quantizer = AdaptiveQuantizer()
        
    def get_layer_bit_width(self, layer_name: str, base_bits: int) -> int:
        """Determine bit-width for specific layer.
        
        Args:
            layer_name: Name of the layer
            base_bits: Base bit-width from adaptive policy
            
        Returns:
            Adjusted bit-width for this layer
        """
        # CNN layers: full adaptive precision
        if 'cnn' in layer_name:
            return base_bits
        
        # TCN blocks 1-2: one bit lower
        if 'tcn_encoder.network.0' in layer_name or 'tcn_encoder.network.1' in layer_name:
            return max(4, base_bits - 1)
        
        # TCN blocks 3-4: two bits lower
        if 'tcn_encoder.network.2' in layer_name or 'tcn_encoder.network.3' in layer_name:
            return max(4, base_bits - 2)
        
        # Output layer: fixed 8-bit for numerical stability
        if 'fc' in layer_name:
            return 8
            
        return base_bits
    
    def quantize_model_mixed_precision(self, model: nn.Module, 
                                       data_variance: float) -> Dict[str, int]:
        """Apply mixed-precision quantization to model.
        
        Args:
            model: PyTorch model
            data_variance: Current data variance
            
        Returns:
            Dictionary mapping layer names to bit-widths used
        """
        # Determine base bit-width from variance
        dummy_data = np.random.randn(10) * data_variance
        base_bits = self.quantizer.select_bit_width(dummy_data)
        
        layer_bits = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name:
                    bits = self.get_layer_bit_width(name, base_bits)
                    param.data = self.quantizer.quantize_tensor(param.data, bits)
                    layer_bits[name] = bits
                    
        return layer_bits


if __name__ == "__main__":
    # Test adaptive quantization
    print("Testing Adaptive Quantization...\n")
    
    quantizer = AdaptiveQuantizer()
    
    # Simulate different variance regimes
    scenarios = [
        ("Stable conditions", 0.03, 500),
        ("Moderate variance", 0.10, 300),
        ("High variance (pollution event)", 0.20, 200)
    ]
    
    for scenario, variance, samples in scenarios:
        print(f"{scenario} (σ = {variance})")
        for _ in range(samples):
            data = np.random.randn(10) * variance
            bit_width = quantizer.select_bit_width(data)
            power = quantizer.estimate_power_consumption(bit_width)
        
        stats = quantizer.get_statistics()
        print(f"  Average bit-width: {stats['avg_bit_width']:.2f}")
        print(f"  Average power: {stats['avg_power']:.3f}W")
        print(f"  Distribution: 4-bit={stats['bit_width_distribution']['4-bit']:.1f}%, "
              f"6-bit={stats['bit_width_distribution']['6-bit']:.1f}%, "
              f"8-bit={stats['bit_width_distribution']['8-bit']:.1f}%\n")
