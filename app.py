import gradio as gr
import cv2
import numpy as np
import os
from inference_pipeline import RestorationPipeline

# 全局初始化 Pipeline (延迟加载)
pipeline = None

def process_image(img, ista_iters, ista_lambda, enable_postprocess, postprocess_alpha):
    global pipeline
    
    # 检查图片是否为空
    if img is None:
        return None, None, None, None
    
    # 初始化模型
    if pipeline is None:
        model_path = 'experiments/pretrained_models/GFPGANv1.3.pth'
        if not os.path.exists(model_path):
            # 如果没下载模型，为了演示代码运行，先抛出错误或返回伪造图像
            # 这里我们假设用户会下载。如果文件不存在，RestorationPipeline 会打印警告
            pass
        pipeline = RestorationPipeline(model_path=model_path)
    
    if pipeline.gfpgan is None:
        return img, img, img, img
        
    # Gradio 传入的是 RGB numpy array，转为 BGR 供 OpenCV/Pipeline 使用
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 运行管道
    try:
        orig_bgr, num_bgr, gfpgan_bgr, final_bgr = pipeline.run(
            img_bgr, 
            int(ista_iters), 
            float(ista_lambda),
            enable_postprocess=bool(enable_postprocess),
            postprocess_alpha=float(postprocess_alpha)
        )
        
        # 转回 RGB 供 Gradio 显示
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        num_rgb = cv2.cvtColor(num_bgr, cv2.COLOR_BGR2RGB)
        gfpgan_rgb = cv2.cvtColor(gfpgan_bgr, cv2.COLOR_BGR2RGB)
        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
        
        return orig_rgb, num_rgb, gfpgan_rgb, final_rgb
    except Exception as e:
        print(f"Error during processing: {e}")
        return img, img, img, img

# 构建界面
title = "融合稀疏先验与对抗学习的人脸超分辨率重建"
description = """
本系统演示了将 **数值分析方法 (ISTA)** 与 **深度学习 (GFPGAN)** 结合的图像复原效果。
"""

# 自定义CSS隐藏footer并优化布局，添加模态框样式
custom_css = """
footer {display: none !important;}
.gradio-container {max-height: 100vh !important;}
.main {padding: 1rem !important;}

/* 图片点击样式 */
.output-image img, .image-container img {cursor: pointer !important;}
.output-image img:hover, .image-container img:hover {opacity: 0.85 !important;}

/* 模态框样式 */
#image-modal {
    display: none;
    position: fixed;
    z-index: 10000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.95);
    cursor: pointer;
    animation: fadeIn 0.2s;
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

#image-modal-content {
    position: relative;
    margin: auto;
    padding: 20px;
    width: 90%;
    height: 90%;
    display: flex;
    align-items: center;
    justify-content: center;
}

#image-modal img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    border-radius: 4px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}

#image-modal-close {
    position: absolute;
    top: 20px;
    right: 35px;
    color: #fff;
    font-size: 40px;
    font-weight: bold;
    cursor: pointer;
    z-index: 10001;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.5);
    border-radius: 50%;
    transition: background 0.3s;
}

#image-modal-close:hover {
    background: rgba(255, 255, 255, 0.2);
}

#image-modal-label {
    position: absolute;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    color: #fff;
    font-size: 18px;
    background: rgba(0, 0, 0, 0.6);
    padding: 8px 16px;
    border-radius: 4px;
}
"""

with gr.Blocks(title=title, css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    
    with gr.Row(equal_height=True):
        # 左侧：输入和控制面板
        with gr.Column(scale=1, min_width=280):
            input_img = gr.Image(label="上传图片", type="numpy", height=200)
            
            with gr.Accordion("数值分析参数 (ISTA)", open=True):
                ista_iters = gr.Slider(
                    minimum=0, maximum=100, value=20, step=1, 
                    label="迭代次数", info="建议范围: 10-30"
                )
                ista_lambda = gr.Slider(
                    minimum=0.0, maximum=0.5, value=0.05, step=0.01, 
                    label="稀疏阈值", info="建议范围: 0.03-0.1"
                )
            
            with gr.Accordion("后处理参数", open=False):
                enable_postprocess = gr.Checkbox(
                    value=True, label="启用局部一致性后处理"
                )
                postprocess_alpha = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                    label="后处理强度", info="较小值保留更多细节"
                )
            
            run_btn = gr.Button("开始重建", variant="primary", size="lg")
        
        # 右侧：输出结果展示（支持点击放大）
        with gr.Column(scale=2, min_width=600):
            with gr.Row():
                out_orig = gr.Image(
                    label="原图", 
                    height=240, 
                    show_label=True, 
                    show_download_button=True,
                    elem_id="output_orig",
                    container=True
                )
                out_num = gr.Image(
                    label="数值分析结果 (ISTA)", 
                    height=240, 
                    show_label=True, 
                    show_download_button=True,
                    elem_id="output_num",
                    container=True
                )
            with gr.Row():
                out_gfpgan = gr.Image(
                    label="GFPGAN 输出", 
                    height=240, 
                    show_label=True, 
                    show_download_button=True,
                    elem_id="output_gfpgan",
                    container=True
                )
                out_final = gr.Image(
                    label="最终结果 (GFPGAN+Post)", 
                    height=240, 
                    show_label=True, 
                    show_download_button=True,
                    elem_id="output_final",
                    container=True
                )

    run_btn.click(
        fn=process_image,
        inputs=[input_img, ista_iters, ista_lambda, enable_postprocess, postprocess_alpha],
        outputs=[out_orig, out_num, out_gfpgan, out_final]
    )
    
    # 添加 JavaScript 实现图片点击放大模态框（带左右切换功能）
    demo.load(
        None,
        None,
        None,
        _js="""
        () => {
            // 存储所有图片信息
            window.imageGallery = {
                images: [],
                currentIndex: 0,
                labels: ['原图', '数值分析结果 (ISTA)', 'GFPGAN 输出', '最终结果 (GFPGAN+Post)']
            };
            
            // 创建模态框
            let modal = document.getElementById('image-modal');
            if (!modal) {
                modal = document.createElement('div');
                modal.id = 'image-modal';
                modal.style.cssText = 'display:none;position:fixed;z-index:10000;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,0.95);';
                modal.innerHTML = `
                    <div id="modal-content" style="position:relative;margin:auto;padding:20px;width:90%;height:90%;display:flex;align-items:center;justify-content:center;overflow:hidden;">
                        <span id="modal-close" style="position:absolute;top:20px;right:35px;color:#fff;font-size:40px;font-weight:bold;cursor:pointer;z-index:10001;width:40px;height:40px;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.5);border-radius:50%;transition:background 0.3s;">×</span>
                        <button id="modal-prev" style="position:absolute;left:20px;top:50%;transform:translateY(-50%);color:#fff;font-size:50px;font-weight:bold;cursor:pointer;z-index:10001;width:80px;height:80px;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.6);border:none;border-radius:50%;transition:background 0.3s;box-shadow:0 2px 10px rgba(0,0,0,0.3);">‹</button>
                        <button id="modal-next" style="position:absolute;right:20px;top:50%;transform:translateY(-50%);color:#fff;font-size:50px;font-weight:bold;cursor:pointer;z-index:10001;width:80px;height:80px;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.6);border:none;border-radius:50%;transition:background 0.3s;box-shadow:0 2px 10px rgba(0,0,0,0.3);">›</button>
                        <div id="modal-img-container" style="position:relative;width:100%;height:100%;display:flex;align-items:center;justify-content:center;overflow:hidden;">
                            <img id="modal-img" src="" style="max-width:100%;max-height:100%;object-fit:contain;border-radius:4px;box-shadow:0 4px 20px rgba(0,0,0,0.5);transition:transform 0.2s;cursor:grab;user-select:none;">
                        </div>
                        <div id="modal-label" style="position:absolute;bottom:20px;left:50%;transform:translateX(-50%);color:#fff;font-size:18px;background:rgba(0,0,0,0.6);padding:8px 16px;border-radius:4px;z-index:10001;"></div>
                        <div id="modal-counter" style="position:absolute;top:20px;left:50%;transform:translateX(-50%);color:#fff;font-size:14px;background:rgba(0,0,0,0.6);padding:4px 12px;border-radius:4px;z-index:10001;"></div>
                        <div id="modal-zoom" style="position:absolute;top:20px;right:80px;color:#fff;font-size:14px;background:rgba(0,0,0,0.6);padding:4px 12px;border-radius:4px;z-index:10001;"></div>
                    </div>
                `;
                document.body.appendChild(modal);
                
                // 图片缩放和拖拽状态
                window.imageZoom = {
                    scale: 1,
                    minScale: 0.5,
                    maxScale: 5,
                    translateX: 0,
                    translateY: 0,
                    isDragging: false,
                    startX: 0,
                    startY: 0
                };
                
                // 悬停效果
                document.getElementById('modal-prev').onmouseenter = function() { this.style.background = 'rgba(255,255,255,0.3)'; };
                document.getElementById('modal-prev').onmouseleave = function() { this.style.background = 'rgba(0,0,0,0.6)'; };
                document.getElementById('modal-next').onmouseenter = function() { this.style.background = 'rgba(255,255,255,0.3)'; };
                document.getElementById('modal-next').onmouseleave = function() { this.style.background = 'rgba(0,0,0,0.6)'; };
                document.getElementById('modal-close').onmouseenter = function() { this.style.background = 'rgba(255,255,255,0.2)'; };
                document.getElementById('modal-close').onmouseleave = function() { this.style.background = 'rgba(0,0,0,0.5)'; };
                
                // 更新图片变换
                function updateImageTransform() {
                    const img = document.getElementById('modal-img');
                    img.style.transform = `translate(${window.imageZoom.translateX}px, ${window.imageZoom.translateY}px) scale(${window.imageZoom.scale})`;
                    img.style.cursor = window.imageZoom.scale > 1 ? 'grab' : 'default';
                    document.getElementById('modal-zoom').textContent = `${Math.round(window.imageZoom.scale * 100)}%`;
                }
                
                // 重置缩放和位置
                function resetZoom() {
                    window.imageZoom.scale = 1;
                    window.imageZoom.translateX = 0;
                    window.imageZoom.translateY = 0;
                    updateImageTransform();
                }
                
                // 缩放图片
                function zoomImage(delta, centerX, centerY) {
                    const img = document.getElementById('modal-img');
                    const rect = img.getBoundingClientRect();
                    const container = document.getElementById('modal-img-container');
                    const containerRect = container.getBoundingClientRect();
                    
                    const oldScale = window.imageZoom.scale;
                    const scaleFactor = delta > 0 ? 1.1 : 0.9;
                    const newScale = Math.max(window.imageZoom.minScale, Math.min(window.imageZoom.maxScale, window.imageZoom.scale * scaleFactor));
                    
                    if (newScale === oldScale) return;
                    
                    // 计算缩放中心点相对于图片的位置
                    const imgCenterX = centerX - containerRect.left - containerRect.width / 2;
                    const imgCenterY = centerY - containerRect.top - containerRect.height / 2;
                    
                    // 调整平移，使缩放中心点保持不变
                    window.imageZoom.translateX = imgCenterX - (imgCenterX - window.imageZoom.translateX) * (newScale / oldScale);
                    window.imageZoom.translateY = imgCenterY - (imgCenterY - window.imageZoom.translateY) * (newScale / oldScale);
                    window.imageZoom.scale = newScale;
                    
                    updateImageTransform();
                }
                
                // 鼠标滚轮缩放
                const modalContent = document.getElementById('modal-content');
                modalContent.addEventListener('wheel', (e) => {
                    if (modal.style.display === 'flex') {
                        e.preventDefault();
                        const delta = e.deltaY > 0 ? -1 : 1;
                        zoomImage(delta, e.clientX, e.clientY);
                    }
                }, { passive: false });
                
                // 图片拖拽（按住拖拽，优化跟手性）
                const modalImg = document.getElementById('modal-img');
                const modalImgContainer = document.getElementById('modal-img-container');
                let dragStartX = 0;
                let dragStartY = 0;
                let initialTranslateX = 0;
                let initialTranslateY = 0;
                let hasMoved = false;
                let rafId = null;
                
                // 使用 requestAnimationFrame 优化拖拽性能
                function handleDragMove(e) {
                    if (!window.imageZoom.isDragging) return;
                    
                    cancelAnimationFrame(rafId);
                    rafId = requestAnimationFrame(() => {
                        const deltaX = e.clientX - dragStartX;
                        const deltaY = e.clientY - dragStartY;
                        
                        // 检测是否移动了足够距离（避免误触）
                        if (Math.abs(deltaX) > 1 || Math.abs(deltaY) > 1) {
                            hasMoved = true;
                        }
                        
                        // 直接更新位置，更跟手
                        window.imageZoom.translateX = initialTranslateX + deltaX;
                        window.imageZoom.translateY = initialTranslateY + deltaY;
                        updateImageTransform();
                    });
                }
                
                modalImg.addEventListener('mousedown', (e) => {
                    // 只响应鼠标左键，且图片已放大
                    if (e.button === 0 && window.imageZoom.scale > 1) {
                        e.preventDefault();
                        e.stopPropagation();
                        window.imageZoom.isDragging = true;
                        hasMoved = false;
                        dragStartX = e.clientX;
                        dragStartY = e.clientY;
                        initialTranslateX = window.imageZoom.translateX;
                        initialTranslateY = window.imageZoom.translateY;
                        modalImg.style.cursor = 'grabbing';
                        modalImg.style.userSelect = 'none';
                        modalImgContainer.style.cursor = 'grabbing';
                        
                        // 在容器上监听，减少延迟
                        modalImgContainer.addEventListener('mousemove', handleDragMove, { passive: false });
                        document.addEventListener('mousemove', handleDragMove, { passive: false });
                    }
                });
                
                function stopDragging() {
                    if (window.imageZoom.isDragging) {
                        window.imageZoom.isDragging = false;
                        cancelAnimationFrame(rafId);
                        modalImg.style.cursor = window.imageZoom.scale > 1 ? 'grab' : 'default';
                        modalImg.style.userSelect = 'auto';
                        modalImgContainer.style.cursor = 'default';
                        
                        // 移除事件监听
                        modalImgContainer.removeEventListener('mousemove', handleDragMove);
                        document.removeEventListener('mousemove', handleDragMove);
                    }
                }
                
                document.addEventListener('mouseup', (e) => {
                    stopDragging();
                });
                
                // 鼠标离开窗口时也停止拖拽
                document.addEventListener('mouseleave', () => {
                    stopDragging();
                });
                
                // 防止在拖拽时触发其他事件
                modalImg.addEventListener('click', (e) => {
                    if (hasMoved) {
                        e.preventDefault();
                        e.stopPropagation();
                    }
                }, true);
                
                // 双击重置缩放
                modalImg.addEventListener('dblclick', () => {
                    resetZoom();
                });
                
                // 关闭功能
                function closeModal() {
                    modal.style.display = 'none';
                    document.body.style.overflow = '';
                    resetZoom();
                }
                
                document.getElementById('modal-close').onclick = (e) => { e.stopPropagation(); closeModal(); };
                modal.onclick = (e) => { if (e.target === modal) closeModal(); };
                
                // 切换图片功能
                function showImage(index) {
                    if (window.imageGallery.images.length === 0) return;
                    const idx = ((index % window.imageGallery.images.length) + window.imageGallery.images.length) % window.imageGallery.images.length;
                    window.imageGallery.currentIndex = idx;
                    const item = window.imageGallery.images[idx];
                    document.getElementById('modal-img').src = item.src;
                    document.getElementById('modal-label').textContent = item.label || '图片预览';
                    document.getElementById('modal-counter').textContent = `${idx + 1} / ${window.imageGallery.images.length}`;
                    
                    // 重置缩放和位置
                    resetZoom();
                    
                    // 更新按钮状态
                    document.getElementById('modal-prev').style.display = window.imageGallery.images.length > 1 ? 'flex' : 'none';
                    document.getElementById('modal-next').style.display = window.imageGallery.images.length > 1 ? 'flex' : 'none';
                }
                
                function prevImage() {
                    showImage(window.imageGallery.currentIndex - 1);
                }
                
                function nextImage() {
                    showImage(window.imageGallery.currentIndex + 1);
                }
                
                document.getElementById('modal-prev').onclick = (e) => { e.stopPropagation(); prevImage(); };
                document.getElementById('modal-next').onclick = (e) => { e.stopPropagation(); nextImage(); };
                
                // 键盘事件
                document.addEventListener('keydown', (e) => {
                    if (modal.style.display === 'flex') {
                        if (e.key === 'Escape') {
                            closeModal();
                        } else if (e.key === 'ArrowLeft') {
                            prevImage();
                        } else if (e.key === 'ArrowRight') {
                            nextImage();
                        }
                    }
                });
                
                // 打开模态框
                window.openImageModal = function(imgSrc, label, startIndex) {
                    // 收集所有输出图片
                    window.imageGallery.images = [];
                    const ids = ['output_orig', 'output_num', 'output_gfpgan', 'output_final'];
                    
                    ids.forEach((id, idx) => {
                        const container = document.getElementById(id);
                        if (container) {
                            const imgs = container.querySelectorAll('img');
                            imgs.forEach(img => {
                                if (img.src && !img.src.includes('data:image/svg') && img.src !== '') {
                                    window.imageGallery.images.push({
                                        src: img.src,
                                        label: window.imageGallery.labels[idx]
                                    });
                                }
                            });
                        }
                    });
                    
                    // 如果通过备用方法找到图片，也添加到列表
                    if (window.imageGallery.images.length === 0) {
                        document.querySelectorAll('form label').forEach(label => {
                            const text = label.textContent;
                            if (text.includes('原图') || text.includes('ISTA') || text.includes('GFPGAN') || text.includes('最终结果')) {
                                const form = label.closest('form');
                                if (form) {
                                    const imgs = form.querySelectorAll('img');
                                    imgs.forEach(img => {
                                        if (img.src && !img.src.includes('data:image/svg') && img.src !== '') {
                                            window.imageGallery.images.push({
                                                src: img.src,
                                                label: text
                                            });
                                        }
                                    });
                                }
                            }
                        });
                    }
                    
                    // 找到当前点击的图片索引
                    let currentIdx = 0;
                    if (startIndex !== undefined) {
                        currentIdx = startIndex;
                    } else {
                        const foundIdx = window.imageGallery.images.findIndex(item => item.src === imgSrc);
                        if (foundIdx !== -1) currentIdx = foundIdx;
                    }
                    
                    showImage(currentIdx);
                    modal.style.display = 'flex';
                    document.body.style.overflow = 'hidden';
                };
            }
            
            // 定期检查并绑定图片点击事件
            function bindImageClicks() {
                // 查找所有输出图片（通过 elem_id）
                const ids = ['output_orig', 'output_num', 'output_gfpgan', 'output_final'];
                const labels = ['原图', '数值分析结果 (ISTA)', 'GFPGAN 输出', '最终结果 (GFPGAN+Post)'];
                
                ids.forEach((id, idx) => {
                    const container = document.getElementById(id);
                    if (container) {
                        const imgs = container.querySelectorAll('img');
                        imgs.forEach(img => {
                            if (img.src && !img.src.includes('data:image/svg') && !img.hasAttribute('data-bound')) {
                                img.setAttribute('data-bound', 'true');
                                img.style.cursor = 'pointer';
                                img.onclick = (e) => {
                                    e.stopPropagation();
                                    e.preventDefault();
                                    window.openImageModal(img.src, labels[idx], idx);
                                };
                            }
                        });
                    }
                });
                
                // 备用方法：通过 label 文本查找
                document.querySelectorAll('form label').forEach((label, labelIdx) => {
                    const text = label.textContent;
                    if (text.includes('原图') || text.includes('ISTA') || text.includes('GFPGAN') || text.includes('最终结果')) {
                        const form = label.closest('form');
                        if (form) {
                            const imgs = form.querySelectorAll('img');
                            imgs.forEach(img => {
                                if (img.src && !img.src.includes('data:image/svg') && !img.hasAttribute('data-bound-alt')) {
                                    img.setAttribute('data-bound-alt', 'true');
                                    img.style.cursor = 'pointer';
                                    img.onclick = (e) => {
                                        e.stopPropagation();
                                        e.preventDefault();
                                        window.openImageModal(img.src, text);
                                    };
                                }
                            });
                        }
                    }
                });
            }
            
            // 立即执行一次
            bindImageClicks();
            
            // 定期检查（图片更新后重新绑定）
            setInterval(bindImageClicks, 500);
            
            // 监听 DOM 变化
            const observer = new MutationObserver(bindImageClicks);
            observer.observe(document.body, { childList: true, subtree: true });
        }
        """
    )

if __name__ == "__main__":
    # 默认仅在局域网暴露，如需公网分享可通过命令行参数或手动设置 share=True
    demo.queue().launch(server_name="0.0.0.0", share=False, show_error=True)

