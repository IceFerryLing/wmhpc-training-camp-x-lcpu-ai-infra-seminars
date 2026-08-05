# Assignment 01 实现日志

## 2026-08-05

### 盘点

通过递归搜索仓库中的 TODO、占位符和 `NotImplementedError`，确认待实现范围覆盖 CUDA Module 0、2、3、4，以及 Python 的 SIMT、Triton、TileLang 和选做 softmax。先读取每个源码的 contract 与 pytest 断言，再开始修改。

### CUDA 实现

- `cuda/m0_env/02_device_query.cu`：补全 `cudaDeviceProp` 字段。
- `cuda/m2_first_kernel/01_vector_add.cu`：补全 kernel 修饰符、线性索引、边界、拷贝方向、grid 尺寸和 launch。
- `02_vector_add_um.cu`：改为 `cudaMallocManaged`，移除拷贝并在 CPU 读结果前显式同步。
- `03_bug_launch.cu`：修正非法的 2048 threads/block，并补充 launch 错误检查。
- `04_matrix_add.cu`：补全二维行列索引、边界和二维 grid。
- `05_grid_stride.cu`：加入跨 grid 的循环。
- `saxpy.cu`：从零实现命令行解析、n=0 特判、数据生成、kernel、event 计时、错误检查和 double 校验和。
- `m3_simt/03_reduce.cu`：分别实现交错配对和连续配对的 shared-memory 归约。
- `m4_memory/01_stencil.cu`：完成静态/动态 shared-memory tile 和 halo 同步。
- `02_constant_coeff.cu`：加入 constant memory 系数表和 `cudaMemcpyToSymbol`。
- `03_histogram.cu`：使用全局原子加。
- `04_histogram_priv.cu`：实现 block 私有 shared histogram，再合并到 global histogram。

### Python 实现

- `simt_sim.py`：使用递归解释嵌套分支，维护 32 lane mask 和 add/mul cycle 计数。
- `vector_add.py`：补全 Triton program id、offset、mask 和 store。
- `fused_op.py`：把计算改为 `relu(a*x+b)`，并扩展 kernel 参数。
- `softmax.py`：实现每 program 一行、减最大值、exp、sum、mask 和按行 launch。
- `tilelang_scale_add.py` / `tilelang_copy2d.py`：补全二维 CTA grid、并行 tile、`T.copy` 和边界处理。
- `tilelang_matmul.py`：补全 shared tiles、fragment、K 维 pipeline、`T.copy` 和 `T.gemm`。
- `tilelang_softmax.py`：按形状缓存编译 kernel，使用二次幂宽度、`-inf` 尾部填充、最大值/求和归约和稳定归一化。

### 验证

1. 登录节点确认 CUDA 13，Slurm 分配 `gj-5090-1` 的 RTX 5090（compute capability 12.0）。
2. CUDA 修改题和 SAXPY judge 全部通过。
3. 首次 Python 评测发现系统 Python 缺少 `Python.h`，Triton 驱动辅助模块无法编译；没有把这个环境错误伪装成代码通过。
4. 使用 `uv` 安装 Python 3.12 并创建 `.venv312`，在 `WMHPC_GRADING=1` 下完整运行：`24 passed`。
5. 记录扩展性、分支 divergence、stride 带宽、occupancy、异步计时、PTX/SASS 和 histogram 性能数据，详见《作业完成说明》。
6. Bonus 的 TileLang 第五组配置申请 144 KiB 动态 shared memory，超过 5090 单 block 上限；将 `BLOCK_K` 从 64 调为 32，使配置满足目标硬件资源约束后重测。

### 当前状态

代码、完成说明和实现日志均已写入远端仓库。TileLang 0.1.12 产生的 21 条弃用警告来自仓库原有 `T.Buffer` API，不影响正确性；归约两版本在 5090 上测得相同短时延，也已按实测结果记录。
