import torch
import torch.nn.functional as F
import numpy as np
import math

class ISTADenoising:
    """
    基于数值分析的 ISTA (Iterative Shrinkage-Thresholding Algorithm) 求解器。
    用于对图像进行稀疏去噪/结构提取。
    
    对应数学模型: min_x 0.5 * ||Ax - y||_2^2 + lambda * ||x||_1
    其中:
    - y: 观测到的低质量图像块
    - A: 稀疏变换字典 (这里采用 DCT 离散余弦变换基)
    - x: 稀疏系数
    """
    def __init__(self, block_size=8, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.block_size = block_size
        self.device = device
        self.dct_matrix = self._create_dct_matrix(block_size).to(device)
        
    def _create_dct_matrix(self, n):
        """
        构建 n x n 的 DCT 变换矩阵 (字典 D)
        显式构建矩阵以体现数值分析方法
        """
        matrix = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == 0:
                    factor = np.sqrt(1/n)
                else:
                    factor = np.sqrt(2/n)
                matrix[i, j] = factor * np.cos((2*j + 1) * i * np.pi / (2*n))
        return matrix

    def soft_thresholding(self, x, threshold):
        """
        软阈值算子 (Soft Thresholding Operator)
        S_lambda(x) = sign(x) * max(|x| - lambda, 0)
        """
        return torch.sign(x) * torch.max(torch.abs(x) - threshold, torch.zeros_like(x))

    def solve(self, img_tensor, lambda_val=0.1, max_iter=50, step_size=0.5):
        """
        使用 ISTA 算法求解稀疏表示并重构图像
        
        参数:
            img_tensor: 输入图像张量 [B, C, H, W], 值域 [0, 1]
            lambda_val: 稀疏惩罚项系数
            max_iter: 最大迭代次数
            step_size: 梯度下降步长
        """
        b, c, h, w = img_tensor.shape
        
        # 1. 将图像分块 (Unfold)
        # input: [B, C, H, W] -> [B, C*k*k, L], where L is number of blocks
        patches = F.unfold(img_tensor, kernel_size=self.block_size, stride=self.block_size)
        
        # 调整维度以便进行矩阵运算
        # patches: [B, C, k*k, L]
        # 我们对每个通道的每个块单独处理
        # Reshape to [B*C*L, k*k] (Batch of vectors)
        # 但为了保持结构清晰，我们先只处理单通道逻辑，利用 broadcasting
        
        # 简化的 patch 处理: [Batch_Total, Block_Size*Block_Size]
        # 这里 y 是我们提取出的含噪 patch 向量
        y = patches.permute(0, 2, 1).reshape(-1, self.block_size * self.block_size)
        
        # 构建 2D DCT 字典 A
        # A_1d 是 block_size x block_size
        # 对于 2D patch (向量化后), A = kron(A_1d, A_1d)
        # 或者更简单，利用 separability: DCT(X) = A_1d * X * A_1d^T
        # 这里的 sparse coefficients x 对应变换域系数
        
        # 初始化稀疏系数 x (全部为 0)
        x = torch.zeros_like(y)
        
        # 预计算 A (这里实际上不需要显式 Kronecker积，利用矩阵乘法性质更快，
        # 但为了符合 ISTA 通用形式 min ||Ax-y||，我们演示显式的梯度下降逻辑)
        
        # 为了演示最标准的 ISTA，我们定义前向变换 A(x_img) 和 逆变换 A^T(coef)
        # 在这里，x 是系数域，y 是像素域
        # 模型: y = InverseDCT(x_coef) + noise
        # 求解: min || InverseDCT(x) - y ||^2 + lambda ||x||_1
        # 梯度项: A^T (A x - y) 
        # 其中 A 是 InverseDCT (从系数到像素), A^T 是 ForwardDCT (从像素到系数)
        # 且 DCT 是正交阵，A^T A = I (理想情况下)
        
        # 迭代求解
        for k in range(max_iter):
            # 1. 计算残差/梯度
            # Forward model: pixel_approx = IDCT(x)
            # 但由于我们定义 x 为系数，A 为 IDCT 矩阵
            # 实际上对于正交基，ISTA 简化为一次阈值处理
            # 为了强行体现 "迭代" (比如针对非正交或为了去噪效果)，
            # 我们通常在 ISTA 中求解的是: x_{k+1} = S(x_k - step * grad)
            # grad = A^T (A x_k - y)
            
            # 将向量形式的 x 还原为块形式进行 DCT 操作
            x_block = x.view(-1, self.block_size, self.block_size)
            
            # A x_k (从系数恢复到像素域估计)
            # IDCT: D.T * Coef * D
            pixel_est = torch.matmul(torch.matmul(self.dct_matrix.t(), x_block), self.dct_matrix)
            
            # A x_k - y
            # y 也是向量，还原为块
            y_block = y.view(-1, self.block_size, self.block_size)
            residual = pixel_est - y_block
            
            # A^T (residual) -> 变换回系数域
            # DCT: D * Res * D.T
            grad_block = torch.matmul(torch.matmul(self.dct_matrix, residual), self.dct_matrix.t())
            grad = grad_block.view(-1, self.block_size * self.block_size)
            
            # 2. 梯度下降
            x_next = x - step_size * grad
            
            # 3. 软阈值操作
            x = self.soft_thresholding(x_next, lambda_val * step_size)
            
        # 重构最终图像
        # x 是最终的稀疏系数
        x_block = x.view(-1, self.block_size, self.block_size)
        
        # IDCT 恢复像素
        rec_patches_block = torch.matmul(torch.matmul(self.dct_matrix.t(), x_block), self.dct_matrix)
        rec_patches_vec = rec_patches_block.view(-1, self.block_size * self.block_size)
        
        # Fold back to image [B, C, H, W]
        # reshape back to [B, L, C*k*k] -> [B, C*k*k, L]
        rec_patches_vec = rec_patches_vec.reshape(b, -1, c * self.block_size * self.block_size).permute(0, 2, 1)
        
        output = F.fold(rec_patches_vec, output_size=(h, w), kernel_size=self.block_size, stride=self.block_size)
        
        # 处理重叠区域的平均 (由于我们使用 stride=block_size 无重叠，直接除以1即可，如果有重叠需处理计数)
        # 这里 stride=kernel_size，无重叠，直接输出
        
        return output

if __name__ == "__main__":
    # 简单测试
    solver = ISTADenoising(block_size=8, device='cpu')
    dummy_img = torch.rand(1, 3, 256, 256)
    out = solver.solve(dummy_img, max_iter=10)
    print(f"Input shape: {dummy_img.shape}, Output shape: {out.shape}")

