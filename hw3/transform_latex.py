import re

# 定义输入的日志字符串
log_data = """
GPU_Name,Total_Global_Memory,Shared_Memory_Per_Block,Max_Threads_Per_Block,nBodies,BLOCK_SIZE,BLOCK_STRIDE,Average_Kernel_Execution_Time_ms,Total_Simulation_Time_ms,Interactions_Per_Second_Billion
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,32,1,0.345,26.233,0.400
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,32,2,0.312,26.625,0.394
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,32,4,0.299,27.562,0.380
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,32,8,0.287,27.135,0.386
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,32,16,0.287,26.535,0.395
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,64,1,0.396,26.760,0.392
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,64,2,0.304,25.942,0.404
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,64,4,0.294,26.156,0.401
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,64,8,0.289,27.281,0.384
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,64,16,0.289,27.258,0.385
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,128,1,0.329,25.578,0.410
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,128,2,0.303,26.204,0.400
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,128,4,0.290,26.411,0.397
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,128,8,0.292,26.252,0.399
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,128,16,0.292,27.278,0.384
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,256,1,0.351,27.315,0.384
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,256,2,0.305,26.882,0.390
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,256,4,0.287,25.621,0.409
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,256,8,0.292,26.601,0.394
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,256,16,0.297,25.205,0.416
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,512,1,0.368,26.218,0.400
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,512,2,0.322,27.292,0.384
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,512,4,0.317,26.821,0.391
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,512,8,0.324,27.354,0.383
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,1024,512,16,0.322,27.457,0.382
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,32,1,0.396,27.260,1.539
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,32,2,0.343,26.104,1.607
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,32,4,0.308,27.178,1.543
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,32,8,0.304,25.786,1.627
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,32,16,0.298,26.797,1.565
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,64,1,0.391,26.904,1.559
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,64,2,0.331,26.279,1.596
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,64,4,0.306,26.301,1.595
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,64,8,0.300,26.992,1.554
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,64,16,0.294,26.123,1.606
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,128,1,0.385,26.663,1.573
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,128,2,0.347,26.341,1.592
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,128,4,0.308,27.017,1.552
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,128,8,0.296,25.895,1.620
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,128,16,0.296,25.919,1.618
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,256,1,0.398,26.477,1.584
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,256,2,0.334,25.701,1.632
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,256,4,0.307,26.299,1.595
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,256,8,0.293,26.036,1.611
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,256,16,0.309,27.371,1.532
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,512,1,0.462,28.451,1.474
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,512,2,0.369,26.030,1.611
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,512,4,0.319,26.069,1.609
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,512,8,0.318,26.329,1.593
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,2048,512,16,0.320,26.017,1.612
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,32,1,0.521,28.525,5.882
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,32,2,0.411,28.130,5.964
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,32,4,0.343,26.632,6.300
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,32,8,0.340,26.790,6.262
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,32,16,0.336,25.624,6.547
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,64,1,0.499,27.640,6.070
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,64,2,0.401,26.903,6.236
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,64,4,0.340,26.807,6.259
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,64,8,0.343,27.562,6.087
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,64,16,0.335,26.406,6.354
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,128,1,0.491,28.967,5.792
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,128,2,0.390,28.019,5.988
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,128,4,0.339,27.012,6.211
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,128,8,0.322,27.277,6.151
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,128,16,0.330,26.600,6.307
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,256,1,0.515,28.375,5.913
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,256,2,0.394,27.110,6.189
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,256,4,0.352,26.242,6.393
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,256,8,0.345,25.903,6.477
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,256,16,0.336,25.569,6.562
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,512,1,0.648,29.127,5.760
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,512,2,0.463,28.240,5.941
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,512,4,0.376,27.684,6.060
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,512,8,0.326,26.249,6.392
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,4096,512,16,0.372,26.527,6.325
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,32,1,0.772,30.108,22.289
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,32,2,0.545,28.064,23.913
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,32,4,0.470,27.750,24.183
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,32,8,0.454,27.203,24.670
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,32,16,0.436,27.257,24.621
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,64,1,0.742,29.641,22.641
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,64,2,0.545,29.125,23.042
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,64,4,0.486,27.924,24.033
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,64,8,0.462,27.576,24.336
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,64,16,0.432,27.241,24.635
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,128,1,0.718,29.524,22.730
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,128,2,0.522,29.080,23.077
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,128,4,0.466,26.916,24.933
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,128,8,0.441,26.985,24.869
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,128,16,0.438,26.770,25.069
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,256,1,0.789,32.238,20.817
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,256,2,0.542,29.157,23.016
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,256,4,0.499,28.569,23.490
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,256,8,0.435,28.636,23.435
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,256,16,0.420,26.720,25.116
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,512,1,1.035,33.369,20.111
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,512,2,0.663,30.254,22.182
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,512,4,0.504,28.779,23.319
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,512,8,0.482,27.299,24.583
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,8192,512,16,0.451,28.155,23.836
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,32,1,1.291,36.409,73.728
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,32,2,0.985,33.382,80.413
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,32,4,0.812,31.805,84.400
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,32,8,1.449,38.637,69.476
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,32,16,0.784,31.911,84.120
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,64,1,1.296,35.340,75.958
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,64,2,0.972,32.083,83.669
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,64,4,0.806,31.104,86.303
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,64,8,0.825,31.648,84.819
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,64,16,0.787,31.288,85.795
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,128,1,1.076,11.004,243.944
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,128,2,0.964,32.363,82.945
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,128,4,0.828,31.558,85.061
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,128,8,0.799,30.356,88.429
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,128,16,0.772,31.602,84.943
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,256,1,1.270,35.471,75.677
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,256,2,1.084,33.551,80.008
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,256,4,0.896,32.561,82.441
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,256,8,1.298,36.334,73.880
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,256,16,0.772,31.558,85.061
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,512,1,1.853,41.117,65.286
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,512,2,1.058,33.817,79.379
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,512,4,1.028,34.014,78.919
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,512,8,0.855,31.031,86.506
NVIDIA A100-SXM4-80GB,81037.75 MB,48.00 KB,1024,16384,512,16,0.764,31.198,86.043
"""

# 定义 LaTeX 表格头部
latex_table = r"""
\begin{longtable}{|c|c|c|c|c|c|}
\hline
\textbf{nBodies} & \textbf{BLOCK\_SIZE} & \textbf{BLOCK\_STRIDE} & \textbf{Average\_Kernel\_Execution\_Time\_ms} & \textbf{Total\_Simulation\_Time\_ms} & \textbf{Interactions\_Per\_Second\_Billion} \\
\hline
\endfirsthead
\hline
\textbf{nBodies} & \textbf{BLOCK\_SIZE} & \textbf{BLOCK\_STRIDE} & \textbf{Average\_Kernel\_Execution\_Time\_ms} & \textbf{Total\_Simulation\_Time\_ms} & \textbf{Interactions\_Per\_Second\_Billion} \\
\hline
\endhead
\hline
\endfoot
\hline
\endlastfoot
"""

# 使用正则表达式提取表格的每一行数据
pattern = re.compile(
    r"NVIDIA A100-SXM4-80GB,.*?,.*?,.*?,(?P<nBodies>\d+),(?P<BLOCK_SIZE>\d+),"
    r"(?P<BLOCK_STRIDE>\d+),(?P<Kernel_Time>\d+\.\d+),(?P<Simulation_Time>\d+\.\d+),"
    r"(?P<Interactions>\d+\.\d+)"
)

# 查找所有匹配行
matches = pattern.findall(log_data)

# 逐行格式化为 LaTeX 表格行
for match in matches:
    nBodies, BLOCK_SIZE, BLOCK_STRIDE, Kernel_Time, Simulation_Time, Interactions = match
    latex_table += f"{nBodies} & {BLOCK_SIZE} & {BLOCK_STRIDE} & {Kernel_Time} & {Simulation_Time} & {Interactions} \\\\\n\\hline\n"

# 添加 LaTeX 表格尾部
latex_table += r"\end{longtable}"

# 将生成的 LaTeX 表格输出到文件
with open("outputs/output_table.tex", "w") as f:
    f.write(latex_table)

print("LaTeX 表格已生成并保存到 output_table.tex")