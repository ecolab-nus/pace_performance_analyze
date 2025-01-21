# conv_analyzer.py

from typing import Dict, List, Any
from hardware_config import BaseAnalyzer, HardwareConfig

class ConvConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        # Input dimensions
        self.input_dimensions = config_dict.get('input_dimensions', [16, 32, 64, 128])
        
        # Kernel configurations
        self.kernel_sizes = config_dict.get('kernel_sizes', [3])
        self.num_channels = config_dict.get('num_channels', 1)
        self.num_filters = config_dict.get('num_filters', 1)
        
        # Data type
        self.data_type_size = config_dict.get('data_type_size', 4)  # bytes per element
        
        # Convolution parameters
        self.padding = config_dict.get('padding', 0)
        self.stride = config_dict.get('stride', 1)
        self.dilation = config_dict.get('dilation', 1)
        self.groups = config_dict.get('groups', 1)
        
        # Tiling configuration
        tiling_config = config_dict.get('tiling', {})
        self.tiling_size = None
        if tiling_config.get('enabled', False):
            self.tiling_size = (
                tiling_config.get('input_tile_size', [32, 32])[0],
                tiling_config.get('input_tile_size', [32, 32])[1]
            )
        
        # Memory pattern
        self.memory_pattern = config_dict.get('memory_pattern', {})
        self.input_layout = self.memory_pattern.get('input_layout', 'NCHW')
        self.kernel_layout = self.memory_pattern.get('kernel_layout', 'KCRS')
        self.vectorization = self.memory_pattern.get('vectorization', 4)


class ConvAnalyzer(BaseAnalyzer):
    def __init__(self, hw_config: HardwareConfig):
        super().__init__(hw_config)

    def analyze(self, config: ConvConfig) -> List[Dict[str, Any]]:
        results = []
        for dim in config.input_dimensions:
            for kernel_size in config.kernel_sizes:
                mem_req = self._calculate_memory_requirements(dim, kernel_size, config)
                latencies = self._calculate_latencies(dim, kernel_size, mem_req, config)
                utilization = self._calculate_memory_utilization(mem_req['total'])
                
                results.append({
                    'input_dimension': dim,
                    'kernel_size': kernel_size,
                    'memory': mem_req,
                    'latencies': latencies,
                    'utilization': utilization,
                    'total_latency': sum(latencies.values())
                })
        return results
    def calculate_memory_access_latency(self, dim: int, kernel_size: int, config: ConvConfig) -> Dict[str, int]:
        """Calculate convolution memory access latency considering memory hierarchy and reuse"""
        # Calculate sizes
        input_size = dim * dim * config.num_channels * config.data_type_size
        kernel_size_bytes = kernel_size * kernel_size * config.num_channels * config.num_filters * config.data_type_size
        output_dim = ((dim + 2*config.padding - kernel_size) // config.stride) + 1
        output_size = output_dim * output_dim * config.num_filters * config.data_type_size
        
        total_size = input_size + kernel_size_bytes + output_size
        
        # Check if all data fits in Central SRAM
        if total_size <= self.hw_config.central_sram_size:
            # Everything fits in Central SRAM - no DRAM access needed
            dram_to_central = 0
            
            # Calculate Central SRAM to CGRA transfers with reuse
            if config.tiling_size:
                tile_h, tile_w = config.tiling_size
                num_tiles_h = (dim + tile_h - 1) // tile_h
                num_tiles_w = (dim + tile_w - 1) // tile_w
                
                # Consider overlap in tiles due to kernel
                effective_tile_size = input_size // (num_tiles_h * num_tiles_w)
                input_load = effective_tile_size * num_tiles_h * num_tiles_w * (1 + (kernel_size - 1))
                kernel_load = kernel_size_bytes  # Kernel can be kept in CGRA memory
                output_load = output_size
                
                total_central_cgra = input_load + kernel_load + output_load
            else:
                # Without tiling - consider sliding window pattern
                # Each input element is read kernel_size * kernel_size times
                input_load = input_size * kernel_size * kernel_size
                kernel_load = kernel_size_bytes  # Kernel loaded once
                output_load = output_size * 2    # Load and store
                
                total_central_cgra = input_load + kernel_load + output_load
                
        else:
            # Need to use DRAM
            if config.tiling_size:
                tile_h, tile_w = config.tiling_size
                num_tiles_h = (dim + tile_h - 1) // tile_h
                num_tiles_w = (dim + tile_w - 1) // tile_w
                
                # Calculate DRAM to Central SRAM transfers
                effective_tile_size = input_size // (num_tiles_h * num_tiles_w)
                input_load = effective_tile_size * num_tiles_h * num_tiles_w * (1 + (kernel_size - 1))
                kernel_load = kernel_size_bytes
                output_load = output_size
                
                total_dram_central = input_load + kernel_load + output_load
                total_central_cgra = total_dram_central
            else:
                # Full convolution without tiling
                input_load = input_size * kernel_size * kernel_size
                kernel_load = kernel_size_bytes
                output_load = output_size * 2
                
                total_dram_central = input_load + kernel_load + output_load
                total_central_cgra = total_dram_central
                
            dram_to_central = self._calculate_transfer_latency(total_dram_central)
        
        # Consider CGRA buffer size for local memory transfers
        buffer_size = self.hw_config.cgra_sram_size
        working_set = min(total_size, config.tiling_size[0] * config.tiling_size[1] * config.num_channels * config.data_type_size 
                        if config.tiling_size else total_size)
        
        reloads = max(1, working_set / buffer_size)
        total_central_cgra *= reloads
        
        central_to_cgra = self._calculate_transfer_latency(total_central_cgra)
        
        return {
            'dram_to_central': dram_to_central,
            'central_to_cgra': central_to_cgra
        }
    def _calculate_memory_requirements(self, input_dim: int, kernel_size: int, config: ConvConfig) -> Dict[str, Any]:
        input_size = input_dim * input_dim * config.num_channels * config.data_type_size
        kernel_total_size = kernel_size * kernel_size * config.num_channels * config.num_filters * config.data_type_size
        output_dim = ((input_dim + 2*config.padding - kernel_size) // config.stride) + 1
        output_size = output_dim * output_dim * config.num_filters * config.data_type_size
        
        total_memory = input_size + kernel_total_size + output_size
        return {
            'input': input_size,
            'kernel': kernel_total_size,
            'output': output_size,
            'total': total_memory,
            'dimensions': {
                'input': (input_dim, input_dim, config.num_channels),
                'kernel': (kernel_size, kernel_size, config.num_channels, config.num_filters),
                'output': (output_dim, output_dim, config.num_filters)
            }
        }

    def _calculate_latencies(self, input_dim: int, kernel_size: int, mem_req: Dict[str, Any], config: ConvConfig) -> Dict[str, int]:
        memory_latencies = self.calculate_memory_access_latency(input_dim, kernel_size, config)
        output_dim = ((input_dim + 2*config.padding - kernel_size) // config.stride) + 1
        computation = 2 * kernel_size * kernel_size * config.num_channels * output_dim * output_dim * config.num_filters
        
        return {
            'dram_to_central': memory_latencies['dram_to_central'],
            'central_to_cgra': memory_latencies['central_to_cgra'],
            'computation': computation
        }
