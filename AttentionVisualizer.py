import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class AttentionVisualizer:
    def __init__(self, model, save_dir="attn_vis"):
        self.model = model
        self.weights = None
        self.hook_handle = None
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self._register_hook()

    def _register_hook(self):
        # 你的 Transformer 结构：model -> transformer -> decoder -> layers -> [-1] -> multihead_attn
        # 我们 hook 最后一层 decoder 的 multihead_attn (Cross Attention)
        # 注意：你需要确保传入的是 policy.model (即 DETRVAE 实例)
        if hasattr(self.model, 'transformer'):
            last_decoder_layer = self.model.transformer.decoder.layers[-1]
            
            def hook_fn(module, input, output):
                # output[1] 是 attention weights (Batch, Num_Heads, Tgt_Len, Src_Len)
                self.weights = output[1].detach().cpu()
                
            self.hook_handle = last_decoder_layer.multihead_attn.register_forward_hook(hook_fn)
        else:
            print("Error: Model does not have 'transformer' attribute. Visualization disabled.")

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()

    def visualize(self, image_tensor, camera_names, t, query_index=0):
        if self.weights is None:
            return

        # 1. 维度兼容处理 (Batch, Heads, Tgt, Src)
        if self.weights.dim() == 3:
            attn_map = self.weights[0, query_index, :] 
        elif self.weights.dim() == 4:
            attn_map = self.weights[0, :, query_index, :].mean(dim=0)
        else:
            return

        # ==========================================
        # 核心修复区域
        # ==========================================
        total_len = attn_map.shape[0] # 应该是 901
        num_cams = len(camera_names)  # 应该是 3
        
        # 自动判断非图像 Token 数量 (优先尝试 1，因为 900 能被 3 整除)
        if (total_len - 1) % num_cams == 0:
            non_image_tokens = 1
        elif (total_len - 2) % num_cams == 0:
            non_image_tokens = 2
        else:
            print(f"Error: Cannot divide {total_len} tokens among {num_cams} cameras.")
            return

        # 切片
        image_attn = attn_map[non_image_tokens:] # 剩下 900 个
        
        # 显式计算特征图尺寸 (根据你的调试信息：15x20)
        # H=480, W=640 -> /32 -> 15, 20
        _, _, _, H, W = image_tensor.shape
        feat_h = H // 32
        feat_w = W // 32
        
        # 再次校验尺寸是否匹配
        expected_pixels = feat_h * feat_w * num_cams # 15*20*3 = 900
        if image_attn.shape[0] != expected_pixels:
            print(f"Shape Mismatch: Expected {expected_pixels} (from {feat_h}x{feat_w}x{num_cams}), but got {image_attn.shape[0]}")
            # 强制回退策略：直接根据像素数开方
            pixels_per_cam = image_attn.shape[0] // num_cams
            feat_h = int(np.sqrt(pixels_per_cam * (H/W)))
            feat_w = int(pixels_per_cam / feat_h)
            print(f"Fallback to calculated shape: {feat_h}x{feat_w}")

        # 归一化
        if image_attn.max() > image_attn.min():
            image_attn = image_attn ** 2  # 增强对比度
            image_attn = (image_attn - image_attn.min()) / (image_attn.max() - image_attn.min())

        # Reshape: (15, 60) -> 因为 3 个相机在宽度上拼接 (20*3=60)
        try:
            attn_heatmap_all = image_attn.reshape(feat_h, feat_w * num_cams)
        except Exception as e:
            print(f"Reshape failed: {e}")
            return

        # 绘图逻辑
        fig, axes = plt.subplots(1, num_cams, figsize=(4 * num_cams, 3))
        if num_cams == 1: axes = [axes]

        for i, cam_name in enumerate(camera_names):
            # 截取当前相机的部分 (15, 20)
            sub_heatmap = attn_heatmap_all[:, i*feat_w : (i+1)*feat_w]
            sub_heatmap = sub_heatmap.detach().numpy()
            
            img = image_tensor[0, i].permute(1, 2, 0).cpu().numpy()
            
            # 放大回 480x640
            sub_heatmap_resized = cv2.resize(sub_heatmap, (W, H))
            
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * sub_heatmap_resized), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
            
            overlay = 0.5 * img + 0.5 * heatmap_colored
            
            axes[i].imshow(overlay)
            axes[i].set_title(f"{cam_name}")
            axes[i].axis('off')

        vis_path = os.path.join(self.save_dir, f"step_{t:04d}.jpg")
        plt.savefig(vis_path)
        plt.close()