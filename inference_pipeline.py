import cv2
import torch
import numpy as np
import importlib.util
import sys
import types
import os

# basicsr (via gfpgan) 仍旧引用已在 torchvision 0.21+ 中移除的
# functional_tensor 模块，这里无条件填充一个轻量兼容层，避免强行降级 torchvision。
import torchvision.transforms.functional as _tv_functional
shim = types.ModuleType("torchvision.transforms.functional_tensor")
shim.rgb_to_grayscale = _tv_functional.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = shim

from gfpgan import GFPGANer
from numerical_solver import ISTADenoising

class RestorationPipeline:
    def __init__(self, model_path='experiments/pretrained_models/GFPGANv1.3.pth', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 初始化数值求解器
        self.solver = ISTADenoising(block_size=8, device=self.device)
        
        # 初始化 GFPGAN
        # upscale=1 表示我们只做人脸修复，保持原分辨率或者由外部控制放大
        # arch='clean' for v1.3
        # channel_multiplier=2 is default for v1.3
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}. Please run download_weights.py first.")
            self.gfpgan = None
        else:
            self.gfpgan = GFPGANer(
                model_path=model_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None, # 暂不使用背景增强以专注人脸
                device=self.device
            )

    def tensor_to_img(self, tensor):
        """
        Tensor [1, C, H, W] (RGB, 0-1) -> Numpy [H, W, C] (BGR, 0-255)
        """
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        # RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def img_to_tensor(self, img):
        """
        Numpy [H, W, C] (BGR, 0-255) -> Tensor [1, C, H, W] (RGB, 0-1)
        """
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def run(self, img, ista_iters=20, ista_lambda=0.05):
        """
        执行完整修复流程
        img: 输入图像 (OpenCV BGR format)
        
        Returns:
            original: 原图
            numerical_result: 经过 ISTA 数值去噪后的中间结果
            final_result: 最终 GAN 修复结果
        """
        if self.gfpgan is None:
            raise RuntimeError("GFPGAN model not loaded.")

        # 1. 准备数据
        input_tensor = self.img_to_tensor(img)
        
        # 2. 数值分析阶段: ISTA 稀疏去噪/结构提取
        # 这一步模拟“利用数学方法先验去除噪声，为 GAN 提供更好的起点”
        # 注意: 如果 lambda 设得太大，会丢失细节；设得太小，去噪效果不明显
        with torch.no_grad():
            denoised_tensor = self.solver.solve(
                input_tensor, 
                max_iter=ista_iters, 
                lambda_val=ista_lambda
            )
        
        numerical_result = self.tensor_to_img(denoised_tensor)
        
        # 3. 深度学习阶段: GFPGAN 修复
        # GFPGANer.enhance 接受 BGR numpy array
        # aligned=False 让 GFPGAN 自动检测和对齐人脸
        # 这里的输入是 numerical_result
        _, _, final_output = self.gfpgan.enhance(
            numerical_result, 
            has_aligned=False, 
            only_center_face=False, 
            paste_back=True
        )
        
        # 同时也跑一个不带 ISTA 的纯 GAN 结果用于对比 (可选，这里暂不返回以简化 UI)
        
        return img, numerical_result, final_output

if __name__ == "__main__":
    # 测试代码
    pass

