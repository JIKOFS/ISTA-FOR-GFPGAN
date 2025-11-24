import torch
import time
from numerical_solver import ISTADenoising

def test_solver():
    print("Testing ISTADenoising Numerical Solver...")
    
    # 1. 初始化
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    
    solver = ISTADenoising(block_size=8, device=device)
    
    # 2. 创建伪造数据 [Batch, Channel, H, W]
    # 模拟一个 256x256 的图像
    dummy_input = torch.rand(1, 3, 256, 256).to(device)
    
    # 3. 运行求解
    start_time = time.time()
    # 显式调用
    output = solver.solve(dummy_input, max_iter=20, lambda_val=0.05)
    end_time = time.time()
    
    # 4. 验证输出
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Processing time for 20 iterations: {end_time - start_time:.4f} seconds")
    
    if output.shape == dummy_input.shape:
        print("SUCCESS: Output shape matches input shape.")
    else:
        print("FAIL: Shape mismatch.")

    # 简单检查数值是否变化 (soft thresholding 应该会改变数值)
    diff = torch.abs(output - dummy_input).mean()
    print(f"Mean difference from input: {diff.item():.6f}")
    
    if diff > 0:
         print("SUCCESS: Image values have been modified by the algorithm.")
    else:
         print("WARNING: Output is identical to input (did lambda=0?)")

if __name__ == "__main__":
    test_solver()

