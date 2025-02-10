# gemm_analyzer.py

from typing import Dict, List, Any
from hardware_config import BaseAnalyzer, HardwareConfig

class GEMMConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        self.dimensions = config_dict.get('dimensions', [16, 32, 64, 128, 256])
        self.data_type_size = config_dict.get('data_type_size', 4)  # bytes per element
        self.tiling_size = config_dict.get('tiling_size', None)

class GEMMAnalyzer(BaseAnalyzer):
    def __init__(self, hw_config: HardwareConfig):
        super().__init__(hw_config)

    def analyze(self, config: GEMMConfig) -> List[Dict[str, Any]]:
        results = []
        for dim in config.dimensions:
            mem_req = self._calculate_memory_requirements(dim, config)
            latencies = self._calculate_latencies(dim, mem_req, config)
            utilization = self._calculate_memory_utilization(mem_req['total'])
            
            results.append({
                'dimension': dim,
                'memory': mem_req,
                'latencies': latencies,
                'utilization': utilization,
                'total_latency': sum(latencies.values())
            })
        return results

    def _calculate_memory_requirements(self, dim: int, config: GEMMConfig) -> Dict[str, Any]:
        matrix_size = dim * dim * config.data_type_size
        total_memory = matrix_size * 3  # Input matrices A, B and output matrix C
        return {
            'per_matrix': matrix_size,
            'total': total_memory,
            'breakdown': {
                'input_a': matrix_size,
                'input_b': matrix_size,
                'output': matrix_size
            }
        }
    
    def calculate_memory_access_latency(self, dim: int, mem_req: Dict[str, Any], config: GEMMConfig) -> Dict[str, int]:
        """Calculate memory access latency with correct tiled access pattern"""
        matrix_size = mem_req['per_matrix']  # Full matrix size in bytes
        total_size = matrix_size * 3  # Total size for A, B, and C
        
        # First check if all data fits in Central SRAM
        if total_size <= self.hw_config.central_sram_size:
            # No DRAM access needed
            dram_to_central = 0
            
            if config.tiling_size:
                tile_size = config.tiling_size
                # Calculate number of tiles in each dimension
                num_tiles_m = (dim + tile_size - 1) // tile_size  # rows
                num_tiles_n = (dim + tile_size - 1) // tile_size  # cols
                num_tiles_k = (dim + tile_size - 1) // tile_size  # reduction dimension
                
                tile_elements = tile_size * tile_size
                tile_bytes = tile_elements * config.data_type_size
                
                # Calculate tile-based data movement
                matrix_a_load = tile_bytes * num_tiles_m * num_tiles_k    # Load each tile once per k-strip
                matrix_b_load = tile_bytes * num_tiles_k * num_tiles_n    # Load each tile once per m-strip
                matrix_c_load = tile_bytes * num_tiles_m * num_tiles_n    # Initial load
                matrix_c_store = matrix_c_load                            # Final store
            else:
                # Without tiling - still need to load data to CGRA
                matrix_a_load = matrix_size
                matrix_b_load = matrix_size
                matrix_c_load = matrix_size
                matrix_c_store = matrix_size
                
            total_central_cgra = matrix_a_load + matrix_b_load + matrix_c_load + matrix_c_store
            
        else:
            # Need to use DRAM - similar pattern but with DRAM access
            if config.tiling_size:
                tile_size = config.tiling_size
                # Calculate number of tiles in each dimension
                num_tiles_m = (dim + tile_size - 1) // tile_size
                num_tiles_n = (dim + tile_size - 1) // tile_size
                num_tiles_k = (dim + tile_size - 1) // tile_size
                
                tile_elements = tile_size * tile_size
                tile_bytes = tile_elements * config.data_type_size
                
                # Calculate DRAM to Central SRAM transfers
                matrix_a_load = tile_bytes * num_tiles_m * num_tiles_k
                matrix_b_load = tile_bytes * num_tiles_k * num_tiles_n
                matrix_c_load = tile_bytes * num_tiles_m * num_tiles_n
                matrix_c_store = matrix_c_load
                
                total_dram_central = matrix_a_load + matrix_b_load + matrix_c_load + matrix_c_store
                total_central_cgra = total_dram_central  # Same pattern for CGRA transfers
                
            else:
                # Without tiling
                matrix_a_load = matrix_size
                matrix_b_load = matrix_size
                matrix_c_load = matrix_size
                matrix_c_store = matrix_size
                
                total_dram_central = matrix_a_load + matrix_b_load + matrix_c_load + matrix_c_store
                total_central_cgra = total_dram_central
                
            # Calculate DRAM to Central SRAM latency
            dram_to_central = self._calculate_transfer_latency(total_dram_central)
        
        # Consider CGRA buffer size for local memory accesses
        buffer_size = self.hw_config.cgra_sram_size
        if config.tiling_size:
            # Working set for one tile computation
            working_set = (2 * tile_size * tile_size + tile_size * tile_size) * config.data_type_size
        else:
            working_set = min(total_size, 3 * matrix_size)
        
        # Calculate reloads based on working set and buffer size
        reloads = max(1, working_set / buffer_size)
        total_central_cgra *= reloads

        # print( "dim:", dim, ",total data transfered from central SPM to cgra:", total_central_cgra)
        
        # Calculate Central SRAM to CGRA latency
        central_to_cgra = self._calculate_transfer_latency(total_central_cgra)
        
        return {
            'dram_to_central': dram_to_central,
            'central_to_cgra': central_to_cgra,
            'debug_info': {
                'matrix_a_load': matrix_a_load,
                'matrix_b_load': matrix_b_load,
                'matrix_c_load': matrix_c_load,
                'matrix_c_store': matrix_c_store,
                'working_set': working_set,
                'reloads': reloads
            }
        }

  

    def _calculate_latencies(self, dim: int, mem_req: Dict[str, Any], config: GEMMConfig) -> Dict[str, int]:
         
        """Calculate GEMM latencies including memory transfers and computation"""
        # Get memory transfer latencies with data reloading
        memory_latencies = self.calculate_memory_access_latency(dim, mem_req, config)
        computation = 2 * dim * dim * dim  # Basic model: 2 ops per element
        # print("operation count:", dim, computation)
        computation =  computation / 4 # the number of operation that a CGRA can execute per cycle
        
        
        return {
            'dram_to_central': memory_latencies['dram_to_central'],
            'central_to_cgra': memory_latencies['central_to_cgra'],
            'computation': computation
        }
