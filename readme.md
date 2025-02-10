this repo is to estimate the systematic CGRA performance for different kernels. We support two operations now: conv and gemm in different sizes. The resulted will be saved in gemm_analysis_gemm_analysis.png and conv_analysis_gemm_analysis.png.

Here are the commands: \
`python analyzer.py  --hw-config hardware_config.yaml --operation conv --op-config conv_config.yaml`

` python analyzer.py  --hw-config hardware_config.yaml --operation gemm --op-config gemm_config.yaml`