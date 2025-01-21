# main.py

import argparse
import yaml
from hardware_config import HardwareConfig, plot_results
from gemm_analyzer import GEMMAnalyzer, GEMMConfig
from conv_analyzer import ConvAnalyzer, ConvConfig

def load_yaml_config(filepath):
    """Load and parse YAML configuration file"""
    with open(filepath, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {filepath}: {e}")
            raise

def convert_hw_config(hw_config):
    """Convert hardware YAML config to HardwareConfig format"""
    hw = hw_config['hardware']
    return {
        'dram_size': hw['dram']['size'],
        'central_sram_size': hw['central_sram']['size'],
        'cgra_sram_size': hw['cgra_sram']['size'],
        'dram_latency': hw['dram']['latency'],
        'central_sram_latency': hw['central_sram']['latency'],
        'cgra_sram_latency': hw['cgra_sram']['latency'],
        'dma_transfer_rate': hw['dma']['transfer_rate'],
        'bits_per_cycle': hw['bits_per_cycle'],
        'bus_frequency': hw['bus_frequency'],
        'cgra_frequency': hw['cgra_frequency']
    }

def main():
    parser = argparse.ArgumentParser(description='CGRA Memory Analysis Tool')
    parser.add_argument('--hw-config', type=str, default="hardware_config.yaml",
                       help=f'Path to hardware configuration YAML file')
    parser.add_argument('--operation', type=str, choices=['gemm', 'conv'],
                       default="gemm",
                       help=f'Operation to analyze ')
    parser.add_argument('--op-config', type=str,default="gemm_config.yaml",
                       help='Path to operation-specific configuration YAML file')
    args = parser.parse_args()
    
    # Load configurations
    hw_config = load_yaml_config(args.hw_config)
    op_config = load_yaml_config(args.op_config)
    
    # Initialize memory configuration
    hw_config = HardwareConfig(convert_hw_config(hw_config))
    
    # Run operation-specific analysis
    if args.operation == 'gemm':
        analyzer = GEMMAnalyzer(hw_config)
        gemm_config = GEMMConfig(op_config)
        results = analyzer.analyze(gemm_config)
        
        # Get output configuration
        output_prefix = op_config['analysis']['output_prefix']
        save_detailed = op_config['analysis']['save_detailed_results']
        
    else:  # conv
        analyzer = ConvAnalyzer(hw_config)
        conv_config = ConvConfig(op_config)
        results = analyzer.analyze(conv_config)
        
        # Get output configuration
        output_prefix = op_config['analysis']['output_prefix']
        save_detailed = op_config['analysis']['save_detailed_results']
    
    # Generate outputs
    plot_results(results,  args.operation.upper(), output_prefix,hw_config)
    if save_detailed:
        # Save as YAML for consistency
        with open(f'{output_prefix}_results.yaml', 'w') as f:
            yaml.dump(results, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    main()