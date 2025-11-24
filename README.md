# 融合稀疏先验与对抗学习的低质量人脸图像超分辨率重建

本项目结合了传统的数值分析方法（ISTA 稀疏求解）与先进的深度学习技术（GFPGAN），旨在解决低质量人脸图像的复原问题。

## 项目结构

- `numerical_solver.py`: 包含基于数值分析的 ISTA 算法实现（显式数学迭代）。
- `inference_pipeline.py`: 整合数值去噪与 GFPGAN 推理的完整流程。
- `app.py`: 基于 Gradio 的 Web 可视化演示界面。
- `download_weights.py`: 自动下载所需的预训练模型权重。

## 快速上手与部署

> 以下流程在 Ubuntu 22.04 + NVIDIA RTX 4090 + CUDA 12.4 上验证通过，其他 Linux 主机可直接移植。

### 1. 系统级依赖

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 socat
```

`libgl1`/`libglib2.0-0` 解决 OpenCV 运行时缺失的共享库，`socat` 可选，用于端口转发。

### 2. 准备 Python 环境

```bash
conda create -n face_restore python=3.9
conda activate face_restore
```

如未安装 Conda，也可以使用 `python3 -m venv venv` 创建虚拟环境，思路相同。

### 3. 安装匹配 CUDA 的 PyTorch

按照宿主机 CUDA 版本选择官方 wheel，例如 CUDA 12.4：

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

若使用 CUDA 11.x，只需替换 `--index-url` 中的版本后缀。

### 4. 安装项目依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 已固定 `basicsr/facexlib/gfpgan/gradio` 等版本，确保与当前实现兼容。

### 5. 下载模型权重

```bash
python download_weights.py
```

模型将保存在 `experiments/pretrained_models/GFPGANv1.3.pth`。

### 6. 运行演示

```bash
python app.py
```

默认监听 `0.0.0.0:7860`。如果容器或远程主机没有直接开放端口，可通过以下任一方式访问：

- **Docker 端口映射**：启动容器时加 `-p 7860:7860`。
- **socat 转发**：在宿主机运行 `socat TCP-LISTEN:7860,fork TCP:<container_ip>:7860`。
- **Gradio 公网分享**：在 `app.py` 中将 `share` 改为 `True`，并按提示放置 `frpc_linux_amd64_v0.2`。

### 7. 可选：Docker 部署

```bash
docker run -dit \
  --gpus all \
  --name face_restore \
  -p 7860:7860
```

进入容器后重复步骤 2~6 即可。若想多机复用，只需改变 `-v` 映射与端口号。

## 数值分析结合点说明

本项目在深度学习重建之前，引入了显式的 **ISTA (Iterative Shrinkage-Thresholding Algorithm)** 算法。
该算法基于压缩感知理论，假设图像在变换域（DCT）具有稀疏性，通过迭代求解优化问题：
$$ \min_x \frac{1}{2} \|Ax - y\|_2^2 + \lambda \|x\|_1 $$
其中 $x$ 为稀疏系数，$y$ 为观测图像，$A$ 为字典矩阵。

这一过程有效去除了部分非结构化噪声，为后续 GAN 网络提供了更好的先验信息。

