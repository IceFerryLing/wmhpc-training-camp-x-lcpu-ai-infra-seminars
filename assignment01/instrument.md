# Assignment 01 作业完成说明

## 环境与总体验证

实现环境为 NVIDIA GeForce RTX 5090，compute capability 12.0，170 个 SM，warp 大小 32，最大常驻线程 1536/SM，显存约 32 GiB，shared memory 上限 48 KiB/block（occupancy 实验显示 100 KiB/SM）。

验证结果：

```text
CUDA 题：全部 PASS
SAXPY judge：7/7 cases PASS
WMHPC_GRADING=1 .venv312/bin/python -m pytest -q：24 passed
```

`.venv312` 使用带开发头文件的 Python 3.12；系统 Python 缺少 `Python.h`，不能用于 Triton 驱动辅助模块编译。TileLang 使用仓库固定版本 `0.1.12`。

## Module 0：环境准备

- `prob 0.1`：`01_hello.cu` 编译运行成功。block 的打印顺序由 GPU 调度器决定，不能假定为 block 编号顺序；同一线程内的输出顺序相对稳定。
- `prob 0.2`：填入 `multiProcessorCount`、`warpSize`、`sharedMemPerBlock`、`maxThreadsPerMultiProcessor`、`totalGlobalMem`。5090 的查询结果见环境段落。

## Module 1：为什么要用 GPU

- `prob 1.1`：
  - (a) 错。TFLOPS 是吞吐指标，不是单条指令延迟。
  - (b) 对。标称带宽通常对应大块、连续、合并访问；随机访问会受事务和延迟影响。
  - (c) 对。严格串行依赖链的关键限制是依赖深度和单步延迟，换更强 GPU 不能消除依赖。
  - (d) 错。1000 TFLOPS 表示单位时间可完成的运算量，不表示每次运算延迟为 (10^{-15}) 秒。
- `prob 1.2`：总 FLOP 很大并不代表严格串行任务能在几秒内完成。吞吐只适用于有足够并行宽度的工作；串行链的完成时间仍近似为步骤数乘单步延迟。
- `prob 1.3`：thread 是最小执行单位，使用寄存器；warp 是 32 个线程的调度/发射单位，可用寄存器和 warp-level primitive；block/CTA 在一个 SM 上协作，使用 shared memory 与 `__syncthreads()`；grid 由多个 block 组成，block 之间不能依赖执行顺序，通常以 kernel 边界同步。
- `prob 1.4`：SIMD 是固定宽度向量执行；SIMT 给每个线程独立状态的编程表象，但硬件按 warp 发射。Volta 之后独立 PC 不会消除 divergence 的代价，同一 warp 仍需串行执行不同路径。
- `prob 1.5`：实测为 CPU 单线程 12.129 ms、GPU `<<<1,1>>>` 137.482 ms、GPU `<<<1,256>>>` 2.349 ms、铺满 grid 0.023 ms。单线程 GPU 受启动开销和极低利用率支配；铺满 grid 的收益来自并行吞吐和延迟隐藏。
- `prob 1.6`（选做）：`kernels/simt_sim.py` 已实现带 mask、分支串行执行和汇合的递归解释器，相关 5 项测试通过。

## Module 2：第一个 CUDA 程序

- `prob 2.1`：向量加法 kernel 使用 `__global__`、线性全局下标、`idx < n` 边界保护、Host-to-Device 拷贝、向上取整的 block 数和合法 launch 配置。
- `prob 2.2`：
  - (a) `__global__`
  - (b) `__device__`
  - (c) `__host__ __device__`
  - (d) `__constant__`
  - (e) `__shared__`
- `prob 2.3`：已改成 `cudaMallocManaged` 版本。kernel 启动异步，CPU 在读取结果前必须同步；显式版本的同步由 device-to-host 的 `cudaMemcpy` 隐含完成。5090 上 unified-memory 计时窗口为 92.4 ms。
- `prob 2.4`：
  - (a) 错，kernel launch 默认异步。
  - (b) 对，同一 stream 中 D2H 拷贝会等待此前操作完成。
  - (c) 错，非法访存通常在后续同步或错误查询处暴露，不能只看启动语句。
- `prob 2.5`：原代码用 2048 threads/block，超过设备的 1024 上限；加入 `CUDA_CHECK_KERNEL()` 并改为 256 threads 后 PASS。原先没有报错是因为 launch 错误未被检查，而后续拷贝/宏只检查了 API 返回值。
- `prob 2.6`：使用 `threadIdx.(x,y)` 计算列/行号，二维边界保护，grid 两个方向分别向上取整，测试 PASS。
- `prob 2.7`：kernel 改为 grid-stride loop。它允许固定的小 launch 覆盖任意规模；代价是每个线程多次循环、重复索引和同步调度，通常比充分铺满的专用 launch 慢。
- `prob 2.8`：block 的先后由 SM 资源和调度器决定；正确性不能依赖 block 顺序。无序 block 才能让同一 kernel 随 SM 数量扩展。
- `prob 2.9`：新增独立错误检查、event 计时、`n=0` 特判和 grid-stride SAXPY。`judge_saxpy.sh` 的 `{0,1,31,1024,1025,2^20,2^20+3}` 全部 PASS。

## Module 3：SIMT 执行

- `prob 3.1`：`blockDim=(8,8,1)` 时 `(3,5,0)` 的线性编号为 `5*8+3=43`，属于 warp 1、lane 11；block 共 2 个 warp。`(33,1,1)` 需要 2 个 warp，第二个 warp 只有 1 个有效线程，其余 lane 空转。
- `prob 3.2`：5090 上按奇偶分支 0.378 ms，按 warp 边界分支 0.195 ms，比值 1.93。warp 内分歧会串行执行两条路径；若两路径工作量不等，奇偶版本由同一 warp 中较慢路径决定，warp 对齐版本则由各自路径的工作量和调度共同决定。
- `prob 3.3`：shared-memory 写入后必须 `__syncthreads()`，否则读取线程可能看到尚未写入的数据；不加同步时属于 data race。某些位置稳定正确，是因为对应的两个线程可能落在同一 warp，硬件执行顺序偶然提供了可见性，不能作为语义保证。
- `prob 3.4`：标准做法是拆成多个 kernel，用一次 kernel launch 边界作为全 grid 的全局同步；需要更强语义时使用 cooperative groups 且满足设备和 launch 条件。
- `prob 3.5`：交错和连续归约都使用 shared memory 和每轮同步，正确性通过。5090 实测均约 0.0062 ms、比值 1.00；本卡/编译器下差异被 kernel 极短执行时间和优化抵消，不能据此认为两种线程布局语义等价。

## Module 4：存储空间

`prob 4.1` 的层次结论：register 是单线程、线程生命周期、片上、编译器管理；local 是单线程/线程生命周期/逻辑私有但通常片外/编译器管理；shared 是 block 可见、block 生命周期、片上、程序员分配释放；global 是整个 grid 可见、由分配释放决定、片外、Runtime/Driver 管理；constant 是 grid 可见、程序生命周期、片外的只读空间、Host 通过 API 更新；L1/L2 是硬件管理的片上 cache，生命周期和可见性由硬件缓存策略决定。

- `prob 4.2`：静态和动态 shared-memory 三点 stencil 都已实现，halo 加载后同步，结果 PASS。
- `prob 4.3`：声明 `__constant__ COEF[8]`，用 `cudaMemcpyToSymbol` 更新，并完成 constant 版本 Horner kernel。5090 上 global 0.0721 ms、constant 0.0716 ms，比值 1.01；constant cache 的优势是一个 warp 读取同一地址的广播模式，而不是所有访问都自动变快。
- `prob 4.4`：两项判断都对。local 表示作用域私有，不代表物理位置在片上；运行期索引的数组可能寄存器溢出到 local memory。
- `prob 4.5`：使用 `atomicAdd(&hist[data[i]], 1u)`，正确性 PASS。
- `prob 4.6`：每个 block 在 shared memory 私有化 256 个 bucket，最后合并到 global histogram。naive 2.5325 ms，private 0.0145 ms，约 175.03x；提速来自把大量全局原子竞争压缩为 block 内原子和少量全局合并。
- `prob 4.7`：stride 1/2/4/8/16/32 的带宽分别为 1898.7/1197.5/730.7/595.8/623.9/877.0 GB/s。stride 增大先破坏合并访问，之后受事务粒度、cache 和访问模式变化影响出现回升。
- `prob 4.8`：5090 上六档为 `(shared/block, blocks/SM, occupancy, GB/s)`：`0 KB,6,100%,1568.3`；`13.2 KB,6,100%,1568.0`；`15 KB,6,100%,1571.5`；`18 KB,5,83.3%,1574.1`；`29 KB,3,50%,1595.1`；`55 KB,1,16.7%,814.9`。occupancy 降低不必然按比例降速；只要有足够常驻 warp 隐藏访存延迟，带宽可以维持，低到一个 block/SM 后才明显下降。

## Module 5：计时与异步初步

`prob 5.1` 实测：host 不同步 0.0082 ms，只测到 kernel 提交开销；host 同步 0.0698 ms，包含提交、等待和同步开销；cudaEvent 0.0691 ms，最适合作为 kernel performance index。`prob 5.2` 三项均对：同一 stream 按提交顺序执行，host 在 launch 后继续执行，Unified Memory 的 CPU/GPU 页所有权冲突可能触发缺页和迁移。

## Module 6：Tile 视角

- `prob 6.1`：(a) 错，tile 是编程抽象，不是可变显存区域；(b) 对，编译器把 tile 操作映射到 block 内执行；(c) 错，同一程序可以混用不同抽象层。
- `prob 6.2/6.3`：SIMT 显式处理 thread/block 索引、全局下标和边界；cuTile 以 block/tile 为并行单位，`bid` 决定 tile，线程到元素的映射交给编译器。SIMT 中的 `threadIdx`、`blockDim`、手工线性下标、线程级边界判断和显式 block 同步，在 cuTile 示例中不再出现。

## Module 7：TileLang 与 Triton

已实现 `vector_add.py`、`fused_op.py`、`tilelang_scale_add.py`、`tilelang_copy2d.py`、`tilelang_matmul.py`、`tilelang_softmax.py` 和选做 `softmax.py`。Triton 测试可在 interpreter 或 GPU 上验证，TileLang 测试在 5090 上验证。

`prob 7.5` 的责任划分：线程到数据映射和边界处理在 CUDA SIMT 中由用户承担，在 cuTile、Triton、TileLang 中主要由编译器/DSL 语义承担；tile/block 尺寸始终由用户选择；block 内同步在 SIMT 中显式由用户写出，在 DSL 中由编译器根据操作依赖插入。CUDA 与 TileLang 的 tile 尺寸仍需结合硬件和实测调参。

`prob 7.2` 的改动只在融合 kernel 的参数接口和逐元素表达式，加载、mask、store、grid 等主体不变，因为 tile/程序到线程的映射由 Triton 编译器处理。

`prob 7.6` 的五个空显式涉及 shared tile、寄存器 fragment、K 维流水、拷贝和 `T.gemm`，用来展示 TileLang 的控制粒度；Triton matmul 把这些映射交给编译器，因此不需要逐项指定。

`prob 7.7/7.8` 都按行实现数值稳定 softmax：先求最大值，再指数化和求和，最后归一化；TileLang wrapper 按 `(M,N)` 缓存编译结果，Triton 使用 `next_power_of_2(N)` 和 mask 处理尾部。

## Module 8：平台与编译

- `prob 8.1`：(a) 错，PTX 是虚拟 ISA；(b) 错，只有 sm_70 SASS 的程序不能直接在 9.0 上运行；(c) 对，fatbin 可携带多架构 SASS 与 PTX；(d) 对，驱动可在运行时把 PTX JIT 成目标机器码。
- `prob 8.2`：5090 运行只含 compute_75 PTX 的版本成功，说明驱动在运行时 JIT；只含 sm_90 SASS 的版本报 `cudaErrorNoKernelImageForDevice`，因为没有适配 12.0 的机器码且没有 PTX 兜底。
- `prob 8.3`：Runtime API 提供高层 CUDA C 运行时抽象；Driver API 提供模块、上下文、函数句柄等更底层控制；`cudaMalloc` 属于 Runtime API。

## Bonus：matmul

5090 上 naive CUDA 的 BS=8/16/32 分别为 5.632/7.058/6.144 TFLOPS。Triton 六组配置的最佳值为 176.5 TFLOPS，torch/cuBLAS 对照为 168.3 TFLOPS；TileLang 五组得到 90.0/131.0/152.3/166.2/171.1 TFLOPS。原第五组 `(128,256,64,256,3)` 需要 144 KiB 动态 shared memory，超过 5090 单 block 上限，因此改为 `(128,256,32,256,3)` 后五组均可运行。

性能差距主要来自 Tensor Core、shared-memory tiling、异步拷贝、软件流水和配置选择。TileLang 控制粒度更细不等于当前版本生成代码一定更快；编译器成熟度、资源限制和调参空间同样影响结果。一次 benchmark 中 Triton 略高于 cuBLAS 不代表普遍优于 cuBLAS，只代表当前尺寸、配置和测量窗口下的结果。
