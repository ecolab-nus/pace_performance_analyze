# hw_config.py

from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
class HardwareConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        # Memory hierarchy configuration
        self.dram_size = config_dict.get('dram_size', 32) * 1024 * 1024  # Convert MB to bytes
        self.central_sram_size = config_dict.get('central_sram_size', 512) * 1024  # Convert KB to bytes
        self.cgra_sram_size = config_dict.get('cgra_sram_size', 32) * 1024  # Convert KB to bytes
        
        # Latency configuration
        self.dram_latency = config_dict.get('dram_latency', 100)
        self.central_sram_latency = config_dict.get('central_sram_latency', 10)
        self.cgra_sram_latency = config_dict.get('cgra_sram_latency', 1)
        self.dma_transfer_rate = config_dict.get('dma_transfer_rate', 16)

        self.bus_frequency = config_dict.get('bus_frequency', 100)
        self.cgra_frequency = config_dict.get('cgra_frequency', 100)

class BaseAnalyzer:
    def __init__(self, hw_config: HardwareConfig):
        self.hw_config = hw_config

    def _calculate_memory_utilization(self, total_memory: int) -> Dict[str, float]:
        return {
            'cgra_sram': min(1.0, total_memory / self.hw_config.cgra_sram_size),
            'central_sram': min(1.0, total_memory / self.hw_config.central_sram_size),
            'dram': min(1.0, total_memory / self.hw_config.dram_size)
        }

    def _calculate_transfer_latency(self, size_bytes: int) -> int:
        return int(np.ceil(size_bytes / self.hw_config.dma_transfer_rate))

def plot_conv_results(results, operation: str, output_prefix: str, hw_config):
    """Generate plots based on convolution analysis results with time in seconds"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Latency Breakdown in Seconds
    plt.subplot(2, 2, 1)
    
    # Get dimensions and latencies
    dims = [r['input_dimension'] for r in results]
    
    # Convert cycles to seconds
    bus_freq = hw_config.bus_frequency * 1e6  # MHz to Hz
    core_freq = hw_config.core_frequency * 1e6  # MHz to Hz
    
    dram_latencies = np.array([r['latencies']['dram_to_central'] / bus_freq for r in results])
    cgra_latencies = np.array([r['latencies']['central_to_cgra'] / bus_freq for r in results])
    comp_latencies = np.array([r['latencies']['computation'] / core_freq for r in results])
    
    # Create x-axis positions
    x_pos = np.arange(len(dims))
    width = 0.35
    
    # Create stacked bar plot
    plt.bar(x_pos, dram_latencies, width, label='DRAM-Central', color='#FF9E4A')
    plt.bar(x_pos, cgra_latencies, width, bottom=dram_latencies, label='Central-CGRA', color='#4A90E2')
    plt.bar(x_pos, comp_latencies, width, 
           bottom=dram_latencies + cgra_latencies, label='Computation', color='#50C878')
    
    # Set log scale for y-axis
    plt.yscale('log')
    
    # Set x-axis ticks and labels
    plt.xticks(x_pos, [f"{d}x{d}" for d in dims], rotation=0)
    
    plt.xlabel('Input Dimension')
    plt.ylabel('Time (seconds) - Log Scale')
    plt.title(f'{operation} Time Breakdown')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, which='both')
    
    # Ensure minimum y-axis range
    plt.ylim(bottom=1e-4)
    
    # Print the actual values
    print(f"\n{operation} Time Breakdown:")
    print("-" * 70)
    print(f"{'Input Size':<15} {'DRAM-Central':>12} {'Central-CGRA':>12} {'Computation':>12} {'Total':>12}")
    print("-" * 70)
    for i, dim in enumerate(dims):
        total = dram_latencies[i] + cgra_latencies[i] + comp_latencies[i]
        dim_str = f"{dim}x{dim}"
        print(f"{dim_str:<15} {dram_latencies[i]:>12.6f} {cgra_latencies[i]:>12.6f} "
              f"{comp_latencies[i]:>12.6f} {total:>12.6f}")
    
    # Plot 2: Memory Utilization
    plt.subplot(2, 2, 2)
    cgra_util = [r['utilization']['cgra_sram'] * 100 for r in results]
    central_util = [r['utilization']['central_sram'] * 100 for r in results]
    dram_util = [r['utilization']['dram'] * 100 for r in results]
    
    plt.plot(dims, cgra_util, 'o-', label='CGRA SRAM')
    plt.plot(dims, central_util, 's-', label='Central SRAM')
    plt.plot(dims, dram_util, '^-', label='DRAM')
    plt.xlabel('Input Dimension')
    plt.ylabel('Memory Utilization (%)')
    plt.title(f'{operation} Memory Utilization')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Kernel Size vs Input Size (if kernel sizes vary)
    plt.subplot(2, 2, 3)
    kernel_sizes = [r.get('kernel_size', 3) for r in results]
    plt.plot(dims, kernel_sizes, 'o-', color='#FF9E4A')
    plt.xlabel('Input Dimension')
    plt.ylabel('Kernel Size')
    plt.title('Convolution Kernel Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_{operation.lower()}_analysis.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()


def plot_results(results, operation: str, output_prefix: str, hw_config: HardwareConfig):
    """Generate plots based on analysis results with time in seconds"""

    if operation == "Conv":
        plot_conv_results(results, operation, output_prefix, hw_config)
        return
    
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Latency Breakdown in Seconds
    plt.subplot(2, 2, 1)
    
    # Get dimensions and latencies
    dims = [r['dimension' if 'dimension' in r else 'input_dimension'] for r in results]
    
    # Convert cycles to seconds
    bus_freq = hw_config.bus_frequency * 1e6  # MHz to Hz
    cgre_freq = hw_config.cgra_frequency * 1e6  # MHz to Hz
    
    dram_latencies = np.array([r['latencies']['dram_to_central'] / bus_freq for r in results])
    cgra_latencies = np.array([r['latencies']['central_to_cgra'] / bus_freq for r in results])
    comp_latencies = np.array([r['latencies']['computation'] / cgre_freq for r in results])
    
    # Create x-axis positions
    x_pos = np.arange(len(dims))
    width = 0.35
    
    # Create stacked bar plot - stack order: DRAM -> CGRA -> Computation
    plt.bar(x_pos, dram_latencies, width, label='DRAM-Central', color='#FF9E4A')
    plt.bar(x_pos, cgra_latencies, width, bottom=dram_latencies, label='Central-CGRA', color='#4A90E2')
    plt.bar(x_pos, comp_latencies, width, 
           bottom=dram_latencies + cgra_latencies, label='Computation', color='#50C878')
    
    # Set log scale for y-axis
    plt.yscale('log')
    
    # Set x-axis ticks and labels
    plt.xticks(x_pos, dims, rotation=0)
    
    plt.xlabel('Input Dimension')
    plt.ylabel('Time (seconds) - Log Scale')
    plt.title(f'{operation} Time Breakdown')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, which='both')
    
    # Ensure minimum y-axis range to show small values
    plt.ylim(bottom=1e-4)
    
    # Print the actual values
    print(f"\n{operation} Time Breakdown:")
    print("-" * 50)
    print(f"{'Dimension':<10} {'DRAM-Central':>12} {'Central-CGRA':>12} {'Computation':>12} {'Total':>12}")
    print("-" * 50)
    for i, dim in enumerate(dims):
        total = dram_latencies[i] + cgra_latencies[i] + comp_latencies[i]
        print(f"{dim:<10} {dram_latencies[i]:>12.6f} {cgra_latencies[i]:>12.6f} "
              f"{comp_latencies[i]:>12.6f} {total:>12.6f}")
    
    # Memory Utilization plot remains the same...
    # [Rest of the plotting code remains unchanged]
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_{operation.lower()}_analysis.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()