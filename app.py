import gradio as gr
import cv2
import numpy as np
import os
from inference_pipeline import RestorationPipeline

# 全局初始化 Pipeline (延迟加载)
pipeline = None

def process_image(img, ista_iters, ista_lambda):
    global pipeline
    
    # 检查图片是否为空
    if img is None:
        return None, None, None
    
    # 初始化模型
    if pipeline is None:
        model_path = 'experiments/pretrained_models/GFPGANv1.3.pth'
        if not os.path.exists(model_path):
            # 如果没下载模型，为了演示代码运行，先抛出错误或返回伪造图像
            # 这里我们假设用户会下载。如果文件不存在，RestorationPipeline 会打印警告
            pass
        pipeline = RestorationPipeline(model_path=model_path)
    
    if pipeline.gfpgan is None:
        return img, img, img # Fallback if model not loaded
        
    # Gradio 传入的是 RGB numpy array，转为 BGR 供 OpenCV/Pipeline 使用
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 运行管道
    try:
        orig_bgr, num_bgr, final_bgr = pipeline.run(img_bgr, int(ista_iters), float(ista_lambda))
        
        # 转回 RGB 供 Gradio 显示
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        num_rgb = cv2.cvtColor(num_bgr, cv2.COLOR_BGR2RGB)
        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
        
        return orig_rgb, num_rgb, final_rgb
    except Exception as e:
        print(f"Error during processing: {e}")
        return img, img, img

# 构建界面
title = "融合稀疏先验与对抗学习的人脸超分辨率重建"
description = """
本系统演示了将 **数值分析方法 (ISTA)** 与 **深度学习 (GFPGAN)** 结合的图像复原效果。
1. **稀疏先验 (Numerical)**: 使用 ISTA 算法显式求解图像稀疏表示，去除噪声结构。
2. **对抗生成 (GAN)**: 使用预训练的 GFPGAN 模型恢复纹理细节。

请上传一张低质量/模糊的人脸照片进行测试。
"""

with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="上传图片", type="numpy")
            
            gr.Markdown("### 数值分析参数 (ISTA)")
            ista_iters = gr.Slider(minimum=0, maximum=100, value=20, step=1, label="迭代次数 (Iterations)")
            ista_lambda = gr.Slider(minimum=0.0, maximum=0.5, value=0.05, step=0.01, label="稀疏阈值 (Lambda)")
            
            run_btn = gr.Button("开始重建", variant="primary")
            
        with gr.Column():
            with gr.Row():
                out_orig = gr.Image(label="原图")
                out_num = gr.Image(label="数值分析中间结果 (ISTA)")
            out_final = gr.Image(label="最终复原结果 (GFPGAN)")

    run_btn.click(
        fn=process_image,
        inputs=[input_img, ista_iters, ista_lambda],
        outputs=[out_orig, out_num, out_final]
    )

if __name__ == "__main__":
    # 默认仅在局域网暴露，如需公网分享可通过命令行参数或手动设置 share=True
    demo.queue().launch(server_name="0.0.0.0", share=False)

