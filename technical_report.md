# 融合稀疏先验与对抗学习的低质量人脸图像超分辨率重建技术报告

## 1. 课题背景与研究目标

本项目源自数学建模课程的大作业，目标是把研究生阶段的数值分析方法与计算机视觉中的对抗生成技术结合，构建一个**可落地、可展示**的低质量人脸图像超分辨率系统。面向的典型场景包括：智能安防监控中模糊人脸的取证、老照片数字化修复以及移动端即时修图等。为了满足课程要求，本项目必须：

- 显式体现数值分析方法（例如 ISTA 稀疏迭代、DCT 字典等）；
- 构建真实的数据流，包括可复现实验的数据集与训练/推理脚本；
- 通过 Web UI 对外演示恢复效果，使结果可以交互调参、实时观察。

最终课题名称确定为**「融合稀疏先验与对抗学习的低质量人脸图像超分辨率重建方法研究」**。

## 2. 系统总体架构

系统分为四个层级，如下图所示：

1. **数据准备层**：以 FFHQ/CelebA-HQ 等高清人脸数据集为基准，利用 Random Blur + Additive Noise + Bicubic Downsampling 等退化算子合成低质图像，形成 `(I_{HQ}, I_{LQ})` 对。项目中通过 `data/` 脚本离线生成，也可替换为真实监控数据。
2. **数值分析层**（`numerical_solver.py`）：实现 DCT 字典与 ISTA（Iterative Shrinkage-Thresholding Algorithm）求解器，用于先验去噪与结构增强。
3. **深度生成层**（`inference_pipeline.py`）：将数值结果输入 GFPGAN 进行纹理恢复与超分，形成从「原图 → 数值增强 → GAN 输出」的串联流程。
4. **可视化/交互层**（`app.py`）：基于 Gradio 搭建可视化界面，并提供迭代次数、阈值等数值参数滑块，用户可以实时观察数值先验对最终 GAN 效果的影响。

该架构实现了传统数学方法与现代深度模型的紧密耦合，是课程设定的“显式结合点”。

## 3. 数值分析模块设计

### 3.1 DCT 稀疏字典

`numerical_solver.py` 中通过 `_create_dct_matrix(n)` 构建 `n×n` 的正交 DCT 字典。之所以选用 DCT，是因为在面部图像的块级纹理上，DCT 能在保留主要几何结构的同时抑制高频噪声，适合作为 ISTA 的基函数。

### 3.2 ISTA 迭代求解

`ISTADenoising.solve()` 按以下流程显式实现：

1. 将输入图像按 `block_size=8` 的滑窗进行 `unfold`，得到一系列 Patch。
2. 在正交字典假设下，构造前向算子 `A = IDCT` 与反向算子 `A^T = DCT`。
3. 迭代执行 `x_{k+1} = S_{λ·α}(x_k - α · A^T(Ax_k - y))`，其中 `S` 为软阈值算子、`α` 为步长。
4. 将稀疏系数 `x` 通过 `fold` 重建图像，得到数值增强结果。

该模块完全符合课程中“基于稀疏表示的图像退化逆问题求解”章节的推导，可在 `test_local.py` 中单独验算对任意张量的去噪效果。

### 3.3 Algorithm 1 式的迭代流程

结合附件中的呈现形式，可以把 `numerical_solver.py` 中的 ISTA 过程写成下述“算法 1”，并指出关键的数学原理：

```
Algorithm 1  Sparse Refinement via ISTA
Input:
    y            观测块（低质图像 unfold 后的向量）
    D            DCT 正交字典
    λ, α         稀疏正则系数与梯度步长
    x^(i-1)      上一次迭代的稀疏系数
Output:
    x^(i)        新的稀疏系数

1:  # Calculate Smoothed Residual
2:  r̂^(i) = Dᵀ(D · x^(i-1) - y)          ▷ 利用正交 DCT 求梯度
3:  # Gradient Step (Data Fidelity)
4:  z = x^(i-1) - α · r̂^(i)              ▷ 最小化 0.5‖Ax - y‖₂²
5:  # Soft Threshold (Sparsity)
6:  x^(i) = SoftThreshold(z, λ·α)         ▷ 施加 L1 稀疏先验
7:  # Loss Tracking (可选)
8:  ℒ^(i) = 0.5‖D x^(i) - y‖₂² + λ‖x^(i)‖₁
9:  重复以上步骤直至迭代上限或残差收敛
```

- **Step 2** 对应代码中的 `grad_block = D · residual · Dᵀ`，源自正交变换的逆传。
- **Step 4** 表示在数据保真项上的梯度下降。
- **Step 6** 为软阈值算子 `S_{λ·α}`，保证稀疏性，是 L1 正则化的闭式解。
- 通过记录 `ℒ^(i)` 或残差均值，可仿照图示那样动态分析每次迭代的贡献，并在 UI/日志中展示“越迭代越好”的趋势。

## 4. 深度学习融合管道

`inference_pipeline.py` 定义了 `RestorationPipeline`：

1. **数据转换**：将 BGR 图像转成 `torch.Tensor`，归一化到 `[0,1]`。
2. **阶段 1（数值增强）**：调用 `ISTADenoising.solve()`，可设置 `max_iter`、`lambda_val` 控制强度。
3. **阶段 2（GAN 修复）**：利用 TencentARC 发布的 `GFPGANv1.3.pth`，通过 `GFPGANer.enhance()` 恢复纹理并放大 2×。
4. **输出**：返回原图、数值结果、最终结果三张图片，供界面并排显示。

为了兼容最新的 `torchvision`（>=0.21），代码中增加了对 `torchvision.transforms.functional_tensor` 的轻量级兼容层，避免降级依赖。

## 5. 可视化演示系统

`app.py` 采用 Gradio 3.50.2 构建，包括：

- **输入区**：图片上传 + 两个 Slider（ISTA 迭代次数、稀疏阈值）。
- **输出区**：原图、数值结果、最终结果三列对比。
- **部署方式**：默认局域网 `0.0.0.0:7860`；如果需要公网访问，可通过 `socat` 转发或设置 `share=True`（并放置 `frpc_linux_amd64_v0.2`）。

本地已经在 Docker 容器 `dd0cc0d8cba1`（`CUDA 12.4 + RTX 4090`）内完成环境部署，并通过 `socat TCP-LISTEN:7860,fork TCP:<containerIP>:7860` 提供局域网访问。

## 6. 数据集与实验流程

1. **数据准备**：基于 FFHQ（70k 高质量人脸）采样 10k 张，应用 `GaussianBlur(σ∈[0.2,3.5]) + 添加高斯噪声(σ=0.01) + 缩放至 128×128` 生成低质图像；高质量图像保持 512×512。
2. **训练流程**：
   - 数值模块无需训练，直接运行 ISTA；
   - GFPGAN 使用官方提供的 `v1.3` 权重；
   - 若需要进一步微调，可按 GFPGAN 官方流程加载生成的 `(LQ,HQ)` 对进行再训练。
3. **推理与展示**：通过 `python app.py` 启动 UI，上传低质样本即可得到三种结果并实时调整参数。

实验显示，适度的 ISTA 迭代（20~40 次）能显著减少背景噪声，使 GFPGAN 输出更锐利；阈值过大会导致细节丢失，需结合 UI 实时调节。

## 7. 环境与部署记录

- Docker 基础镜像：`nvcr.io/nvidia/cuda:12.3.2-base-ubuntu22.04`
- Conda 环境：`face_restore`（Python 3.9、torch 2.6.0+cu124、torchvision 0.21.0+cu124）
- 关键依赖：`basicsr 1.4.2`、`facexlib 0.3.0`、`gfpgan 1.3.8`、`gradio 3.50.2`
- 系统库：`libgl1`、`libglib2.0-0` 等已在容器中安装，用于满足 OpenCV 的共享库需求。

## 8. 结论与展望

本系统实现了“数值分析 + 对抗学习”双管齐下的人脸超分辨率方案，具备以下特点：

1. **理论完整**：在 GAN 推理前显式执行 ISTA，满足课程对数值方法的要求；
2. **工程落地**：提供 Docker/Conda 环境、预训练模型下载脚本、可视化界面；
3. **可扩展性**：可替换数值先验（如 TV/L0 约束）或升级 GAN（如 CodeFormer、RestoreFormer）。

后续可考虑：

- 引入多尺度小波或稀疏卷积进一步提升数值阶段的高频保真度；
- 在 UI 中新增批量推理、自动化指标（PSNR/SSIM）展示；
- 基于收集到的真实低质数据进行 GFPGAN 微调，提高泛化能力。

本报告可直接作为课程论文的技术章节基础，同时项目仓库已具备复现实验与展示所需的一切脚本与说明。


