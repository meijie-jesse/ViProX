import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

from collections import OrderedDict

try:
    from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
    from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D
    _HAS_ROBOMIMIC = True
except ImportError:
    ResNet18Conv = SpatialSoftmax = replace_bn_with_gn = ConditionalUnet1D = None
    _HAS_ROBOMIMIC = False


def _require_robomimic(policy_name):
    if not _HAS_ROBOMIMIC:
        raise ImportError(
            f"{policy_name} requires robomimic (tested with v0.3.0). "
            "Install it with: pip install robomimic==0.3.0  "
            "(or from source: git clone https://github.com/ARISE-Initiative/robomimic && pip install -e robomimic)"
        )


from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel


class FlowMatchingPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        _require_robomimic('FlowMatchingPolicy')

        self.camera_names = args_override['camera_names']
        self.observation_horizon = args_override['observation_horizon']
        self.action_horizon = args_override['action_horizon'] 
        self.prediction_horizon = args_override['prediction_horizon']
        self.num_inference_timesteps = args_override['num_inference_timesteps']
        self.ema_power = args_override['ema_power']
        self.lr = args_override['lr']
        self.weight_decay = 0

        self.num_kp = 32
        self.feature_dimension = 64
        self.ac_dim = args_override['action_dim'] 
        self.obs_dim = self.feature_dimension * len(self.camera_names) + 14 

        # --- Vision Encoders (Same as Diffusion) ---
        backbones = []
        pools = []
        linears = []
        for _ in self.camera_names:
            backbones.append(ResNet18Conv(**{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False}))
            pools.append(SpatialSoftmax(**{'input_shape': [512, 15, 20], 'num_kp': self.num_kp, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0}))
            linears.append(torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))
        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)
        
        backbones = replace_bn_with_gn(backbones)

        # --- Vector Field Predictor (v_theta) ---
        # 使用与 Diffusion 相同的 U-Net 结构来预测速度场 v
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim*self.observation_horizon
        )

        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'backbones': backbones,
                'pools': pools,
                'linears': linears,
                'noise_pred_net': noise_pred_net
            })
        })

        nets = nets.float().cuda()
        ENABLE_EMA = True
        if ENABLE_EMA:
            from diffusers.training_utils import EMAModel
            ema = EMAModel(model=nets, power=self.ema_power)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        n_parameters = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_parameters/1e6,))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        B = qpos.shape[0]
        
        # --- Encode Observations ---
        nets = self.nets
        # Inference time: use EMA model if available
        if actions is None and self.ema is not None:
            nets = self.ema.averaged_model

        all_features = []
        for cam_id in range(len(self.camera_names)):
            cam_image = image[:, cam_id]
            cam_features = nets['policy']['backbones'][cam_id](cam_image)
            pool_features = nets['policy']['pools'][cam_id](cam_features)
            pool_features = torch.flatten(pool_features, start_dim=1)
            out_features = nets['policy']['linears'][cam_id](pool_features)
            all_features.append(out_features)

        obs_cond = torch.cat(all_features + [qpos], dim=1)

        # --- Training: Flow Matching Loss ---
        if actions is not None: 
            # 1. Sample t uniform [0, 1]
            t = torch.rand((B,), device=obs_cond.device)
            
            # 2. Sample noise x_0 ~ N(0, 1)
            x_0 = torch.randn_like(actions)
            x_1 = actions # Data (Target)

            # 3. Compute x_t (Linear Interpolation)
            # x_t = (1 - t) * x_0 + t * x_1
            # Broadcasting t to shape [B, 1, 1]
            t_expand = t.view(-1, 1, 1)
            x_t = (1 - t_expand) * x_0 + t_expand * x_1

            # 4. Compute Target Vector Field u_t(x) = x_1 - x_0
            target_v = x_1 - x_0

            # 5. Predict Vector Field v_theta
            # Check: ConditionalUnet1D usually expects timesteps. 
            # Map t [0, 1] to a larger range for sinusoidal embeddings if necessary, 
            # but usually raw float works if the embedding layer supports it. 
            # To be safe and consistent with Diffusion implementations, we scale t * 999.
            model_output = nets['policy']['noise_pred_net'](
                x_t, 
                t * 999, 
                global_cond=obs_cond
            )

            # 6. Loss (MSE)
            loss_elementwise = F.mse_loss(model_output, target_v, reduction='none')
            # Mask out padding if necessary
            loss = (loss_elementwise * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {'loss': loss}
            
            if self.training and self.ema is not None:
                self.ema.step(nets)
            return loss_dict

        # --- Inference: Improved ODE Solver (Midpoint Method) ---
        else:
            Tp = self.prediction_horizon
            action_dim = self.ac_dim
            B = qpos.shape[0]
            
            # 1. Initialize x from N(0, 1)
            x = torch.randn((B, Tp, action_dim), device=obs_cond.device)
            
            # 2. Solver Config
            steps = self.num_inference_timesteps # 建议设为 10 或 15
            
            # 使用 Midpoint 方法 (2nd order)，比 Euler 更准
            # t 从 0 -> 1
            for i in range(steps):
                t_curr = i / steps
                t_next = (i + 1) / steps
                dt = t_next - t_curr
                
                # Step 1: Calculate v at t_curr
                t_input = torch.full((B,), t_curr, device=obs_cond.device)
                v_curr = nets['policy']['noise_pred_net'](x, t_input * 999, global_cond=obs_cond)
                
                # Step 2: Euler step to midpoint
                x_mid = x + v_curr * (0.5 * dt)
                t_mid = t_curr + 0.5 * dt
                
                # Step 3: Calculate v at midpoint
                t_mid_input = torch.full((B,), t_mid, device=obs_cond.device)
                v_mid = nets['policy']['noise_pred_net'](x_mid, t_mid_input * 999, global_cond=obs_cond)
                
                # Step 4: Update full step using midpoint velocity
                x = x + v_mid * dt

            return x

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')
            status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status


# 20251210 300000 成功率0.7 
# class FlowMatchingPolicy(nn.Module):
#     def __init__(self, args_override):
#         super().__init__()

#         # --- 1. 参数配置 ---
#         self.camera_names = args_override['camera_names']
#         self.action_dim = args_override['action_dim']
#         self.prediction_horizon = args_override['prediction_horizon']
#         # self.observation_horizon = args_override['observation_horizon']
#         # [建议] 推理时增加步数以提高精度，虽然慢一点点
#         self.num_inference_timesteps = args_override.get('num_inference_timesteps', 20) 
        
#         # --- 模型超参数 (Scaling Up) ---
#         self.num_kp = 32
#         self.feature_dimension = 64
#         self.hidden_dim = 512       # 保持 512
#         self.dim_feedforward = 3200 # [升级] 从默认 2048 提升到 3200 (ACT 原版配置)
#         self.nheads = 8
#         self.n_layers = 8           # [升级] 从 4 层加深到 8 层，大幅提升容量
#         self.dropout = 0.05         # [微调] 降低 Dropout，适合中小数据集，防止欠拟合
        
#         # --- 2. 视觉 Backbone ---
#         backbones = []
#         pools = []
#         linears = []
#         for _ in self.camera_names:
#             # 这里的 ResNet 默认是随机初始化的。
#             # 如果数据集很小 (<50 episodes)，建议考虑加载 ImageNet 预训练权重
#             backbones.append(ResNet18Conv(**{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False}))
#             pools.append(SpatialSoftmax(**{'input_shape': [512, 15, 20], 'num_kp': self.num_kp, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0}))
#             linears.append(torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))
            
#         self.backbones = nn.ModuleList(backbones)
#         self.pools = nn.ModuleList(pools)
#         self.linears = nn.ModuleList(linears)
#         self.backbones = replace_bn_with_gn(self.backbones)

#         # --- 3. 投影与编码 ---
#         self.input_proj = nn.Linear(self.feature_dimension, self.hidden_dim)
#         self.qpos_proj = nn.Linear(14, self.hidden_dim)
#         self.action_proj = nn.Linear(self.action_dim, self.hidden_dim)
        
#         self.time_emb = SinusoidalPositionalEmbedding(self.hidden_dim)
#         self.cam_pos_embed = nn.Parameter(torch.randn(len(self.camera_names), 1, self.hidden_dim))
#         self.action_pos_embed = nn.Parameter(torch.randn(1, self.prediction_horizon, self.hidden_dim))

#         # --- 4. Transformer Backbone (High Capacity) ---
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.hidden_dim, 
#             nhead=self.nheads, 
#             dim_feedforward=self.dim_feedforward, 
#             dropout=self.dropout, 
#             activation='gelu', # [微调] 使用 GeLU，通常比 ReLU 表现更好
#             batch_first=True,
#             norm_first=True    # [微调] Pre-Norm 结构，训练更稳定，容易收敛
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

#         # --- 5. 输出头 ---
#         self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        
#         # 初始化权重 (利于 Deep Transformer 收敛)
#         self._init_weights()

#         print(f"High-Capacity ACT-Flow Policy Initialized. Layers: {self.n_layers}, FFN: {self.dim_feedforward}")

#     def _init_weights(self):
#         # 对 Linear 层进行 Xavier 初始化，帮助深层网络训练
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def __call__(self, qpos, image, actions=None, is_pad=None):
#         B = qpos.shape[0]
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#         # === 1. 特征提取 ===
#         all_cam_features = []
#         for cam_id in range(len(self.camera_names)):
#             cam_img = image[:, cam_id]
#             cam_img = normalize(cam_img)
#             feat = self.backbones[cam_id](cam_img)
#             feat = self.pools[cam_id](feat)
#             feat = torch.flatten(feat, start_dim=1)
#             feat = self.linears[cam_id](feat)
#             feat = self.input_proj(feat)
#             feat = feat + self.cam_pos_embed[cam_id] 
#             all_cam_features.append(feat.unsqueeze(1))
        
#         vision_tokens = torch.cat(all_cam_features, dim=1)
#         qpos_token = self.qpos_proj(qpos).unsqueeze(1)
#         obs_tokens = torch.cat([qpos_token, vision_tokens], dim=1)

#         # === 2. 训练逻辑 ===
#         if actions is not None:
#             if is_pad is not None:
#                 mask = ~is_pad
#                 mask = mask.unsqueeze(-1)
#             else:
#                 mask = torch.ones((B, self.prediction_horizon, 1), device=qpos.device)

#             # --- [关键改进] Logit-Normal 或 Beta 采样 ---
#             # 相比均匀分布，让模型更多地在靠近 t=1 (Data) 和 t=0 (Noise) 的地方训练
#             # 这里使用 simple Beta(2,1) 稍微偏向于 t=1 (更精细的去噪)
#             # 或者依然使用 Uniform 但依赖强大的模型容量
#             # 实践建议：先试用 Uniform，因为 Rectified Flow 理论上偏好 straight line
#             t = torch.rand((B,), device=qpos.device) 
            
#             # 噪声注入
#             noise = torch.randn_like(actions)
#             x_1 = actions
            
#             t_expand = t.view(B, 1, 1)
#             x_t = t_expand * x_1 + (1 - t_expand) * noise
#             v_target = x_1 - noise

#             # 编码预测
#             action_tokens = self.action_proj(x_t)
#             action_tokens = action_tokens + self.action_pos_embed[:, :self.prediction_horizon, :]
#             time_token = self.time_emb(t)
#             action_tokens = action_tokens + time_token.unsqueeze(1)

#             transformer_input = torch.cat([obs_tokens, action_tokens], dim=1)
#             output = self.transformer(transformer_input)
#             action_output = output[:, obs_tokens.shape[1]:, :]
#             v_pred = self.action_head(action_output)

#             # Loss 计算
#             loss_mse = F.mse_loss(v_pred, v_target, reduction='none')
#             loss = (loss_mse * mask).sum() / (mask.sum() * self.action_dim + 1e-6)

#             return {
#                 'loss': loss,
#                 'l2_loss': loss
#             }

#         # === 3. 推理逻辑 ===
#         else:
#             # 初始化 x_0
#             x_t = torch.randn((B, self.prediction_horizon, self.action_dim), device=qpos.device)
            
#             # [关键改进] 增加推理步数
#             steps = self.num_inference_timesteps # 建议 20+
#             dt = 1.0 / steps
            
#             for i in range(steps):
#                 t_val = i / steps
#                 t_tensor = torch.full((B,), t_val, device=qpos.device)
                
#                 action_tokens = self.action_proj(x_t)
#                 action_tokens = action_tokens + self.action_pos_embed[:, :self.prediction_horizon, :]
#                 time_token = self.time_emb(t_tensor)
#                 action_tokens = action_tokens + time_token.unsqueeze(1)
                
#                 transformer_input = torch.cat([obs_tokens, action_tokens], dim=1)
#                 output = self.transformer(transformer_input)
#                 action_output = output[:, obs_tokens.shape[1]:, :]
                
#                 v_pred = self.action_head(action_output)
                
#                 x_t = x_t + v_pred * dt
            
#             return x_t

#     def configure_optimizers(self):
#         # 针对 ResNet Backbone 和 Transformer 使用不同的学习率可能是个好主意
#         # 但为了简单起见，这里保持统一，但可以适当增大 weight_decay
#         optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
#         return optimizer

#     def serialize(self):
#         return self.state_dict()

#     def deserialize(self, model_dict):
#         self.load_state_dict(model_dict)
#         print('Loaded High-Capacity Flow Matching Policy')









# 1209能训练 能有大致轨迹趋势 20000step 
# class FlowMatchingPolicy(nn.Module):
#     def __init__(self, args_override):
#         super().__init__()

#         # --- 1. 参数设置 ---
#         self.camera_names = args_override['camera_names']
#         self.action_dim = args_override['action_dim']
#         self.prediction_horizon = args_override['prediction_horizon'] # 100
#         # self.observation_horizon = args_override['observation_horizon']
#         self.num_inference_timesteps = args_override.get('num_inference_timesteps', 10) # 推荐 10-100
        
#         # 内部超参 (调优后适合 ACT)
#         self.num_kp = 32
#         self.feature_dimension = 64
#         self.hidden_dim = 512 # Transformer 维度
#         self.nheads = 8
#         self.n_layers = 4     # ACT 原始层数，足够了
        
#         # --- 2. 视觉编码器 (ResNet + SpatialSoftmax) ---
#         backbones = []
#         pools = []
#         linears = []
#         for _ in self.camera_names:
#             backbones.append(ResNet18Conv(**{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False}))
#             pools.append(SpatialSoftmax(**{'input_shape': [512, 15, 20], 'num_kp': self.num_kp, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0}))
#             linears.append(torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))
        
#         self.backbones = nn.ModuleList(backbones)
#         self.pools = nn.ModuleList(pools)
#         self.linears = nn.ModuleList(linears)
#         self.backbones = replace_bn_with_gn(self.backbones)

#         # --- 3. 投影层与 Embedding ---
#         # 将 ResNet 特征投影到 Transformer 维度
#         self.input_proj = nn.Linear(self.feature_dimension, self.hidden_dim)
        
#         # QPos (Proprioception) 投影
#         self.qpos_proj = nn.Linear(14, self.hidden_dim) # 假设 qpos 是 14 维

#         # 动作投影 (Noisy Action -> Hidden)
#         self.action_proj = nn.Linear(self.action_dim, self.hidden_dim)
        
#         # 时间嵌入 (Time Step Embedding)
#         self.time_emb = SinusoidalPositionalEmbedding(self.hidden_dim)
        
#         # Learnable Positional Encodings
#         # 为每个相机视角学习一个位置编码
#         self.cam_pos_embed = nn.Parameter(torch.randn(len(self.camera_names), 1, self.hidden_dim))
#         # 为动作序列学习位置编码 (Sequence Position)
#         self.action_pos_embed = nn.Parameter(torch.randn(1, self.prediction_horizon, self.hidden_dim))

#         # --- 4. Transformer Backbone (Early Fusion / Self-Attention) ---
#         # 这是解决 VLA 代码水土不服的关键：使用标准的 Encoder
#         encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.nheads, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

#         # --- 5. 输出头 ---
#         # 预测速度场 v
#         self.action_head = nn.Linear(self.hidden_dim, self.action_dim)

#         print("Initialized ACT-Native Flow Matching Policy.")

#     def __call__(self, qpos, image, actions=None, is_pad=None):
#         """
#         Input:
#             qpos: (B, 14)
#             image: (B, num_cams, 3, H, W)
#             actions: (B, T, D) [Training]
#             is_pad: (B, T) [Training]
#         """
#         B = qpos.shape[0]
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#         # === 1. 特征提取 ===
#         # 处理图像
#         all_cam_features = []
#         for cam_id in range(len(self.camera_names)):
#             cam_img = image[:, cam_id]
#             cam_img = normalize(cam_img)
#             feat = self.backbones[cam_id](cam_img)
#             feat = self.pools[cam_id](feat)
#             feat = torch.flatten(feat, start_dim=1)
#             feat = self.linears[cam_id](feat) # (B, 64)
#             feat = self.input_proj(feat)      # (B, 512)
#             # 加相机位置编码
#             feat = feat + self.cam_pos_embed[cam_id] 
#             all_cam_features.append(feat.unsqueeze(1))
        
#         # 视觉 Token: (B, Num_Cams, 512)
#         vision_tokens = torch.cat(all_cam_features, dim=1)
        
#         # 处理 QPos
#         qpos_token = self.qpos_proj(qpos).unsqueeze(1) # (B, 1, 512)
        
#         # 组合 Conditioning Token (Observation)
#         # 顺序: [Proprio, Cam1, Cam2...]
#         obs_tokens = torch.cat([qpos_token, vision_tokens], dim=1) # (B, 1+Num_Cams, 512)

#         # === 2. 训练逻辑 ===
#         if actions is not None:
#             # 构造 Mask
#             if is_pad is not None:
#                 # is_pad: True 表示是 padding
#                 mask = ~is_pad # True 表示有效数据
#                 mask = mask.unsqueeze(-1) # (B, T, 1)
#             else:
#                 mask = torch.ones((B, self.prediction_horizon, 1), device=qpos.device)

#             # --- Flow Matching 核心逻辑 (Rectified Flow) ---
#             # 1. 采样时间 t ~ Uniform[0, 1]
#             t = torch.rand((B,), device=qpos.device)
            
#             # 2. 采样噪声 x_0 ~ N(0, 1)
#             noise = torch.randn_like(actions) 
#             x_1 = actions # 真实动作
            
#             # 3. 插值得到 x_t = t * x_1 + (1 - t) * x_0
#             t_expand = t.view(B, 1, 1)
#             x_t = t_expand * x_1 + (1 - t_expand) * noise
            
#             # 4. 计算目标速度 v_target = x_1 - x_0
#             v_target = x_1 - noise

#             # 5. 模型前向预测 v_pred
#             # 将 x_t 编码
#             action_tokens = self.action_proj(x_t) # (B, T, 512)
#             # 加入序列位置编码
#             action_tokens = action_tokens + self.action_pos_embed[:, :self.prediction_horizon, :]
#             # 加入时间编码 t
#             time_token = self.time_emb(t) # (B, 512)
#             action_tokens = action_tokens + time_token.unsqueeze(1)

#             # 拼接输入: [Observation, Noisy Actions]
#             # 这是 Early Fusion 的关键，让 Attention 充分交互
#             transformer_input = torch.cat([obs_tokens, action_tokens], dim=1)
            
#             # 经过 Transformer
#             output = self.transformer(transformer_input)
            
#             # 取出对应 Action 部分的输出 (后半部分)
#             action_output = output[:, obs_tokens.shape[1]:, :]
            
#             # 投影回动作维度
#             v_pred = self.action_head(action_output)

#             # 6. 计算 Loss
#             loss_mse = F.mse_loss(v_pred, v_target, reduction='none')
#             # 只计算非 Padding 部分的 Loss
#             loss = (loss_mse * mask).sum() / (mask.sum() * self.action_dim + 1e-6)

#             return {
#                 'loss': loss,
#                 'l2_loss': loss
#             }

#         # === 3. 推理逻辑 (Euler ODE Solver) ===
#         else:
#             # 1. 初始化 x_0 ~ N(0, 1)
#             x_t = torch.randn((B, self.prediction_horizon, self.action_dim), device=qpos.device)
            
#             # 2. 设置时间步
#             steps = self.num_inference_timesteps
#             dt = 1.0 / steps
            
#             # 3. 逐步积分
#             for i in range(steps):
#                 # 当前时间 t
#                 t_val = i / steps
#                 t_tensor = torch.full((B,), t_val, device=qpos.device)
                
#                 # 编码当前 x_t
#                 action_tokens = self.action_proj(x_t)
#                 action_tokens = action_tokens + self.action_pos_embed[:, :self.prediction_horizon, :]
#                 time_token = self.time_emb(t_tensor)
#                 action_tokens = action_tokens + time_token.unsqueeze(1)
                
#                 # 拼接 + Transformer
#                 transformer_input = torch.cat([obs_tokens, action_tokens], dim=1)
#                 output = self.transformer(transformer_input)
#                 action_output = output[:, obs_tokens.shape[1]:, :]
                
#                 # 预测速度 v
#                 v_pred = self.action_head(action_output)
                
#                 # Euler 更新: x_{t+1} = x_t + v * dt
#                 x_t = x_t + v_pred * dt
            
#             return x_t

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
#         return optimizer

#     def serialize(self):
#         return self.state_dict()

#     def deserialize(self, model_dict):
#         self.load_state_dict(model_dict)
#         print('Loaded ACT-Native Flow Matching Policy')




class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        _require_robomimic('DiffusionPolicy')

        self.camera_names = args_override['camera_names']

        self.observation_horizon = args_override['observation_horizon'] ### TODO TODO TODO DO THIS
        self.action_horizon = args_override['action_horizon'] # apply chunk size
        self.prediction_horizon = args_override['prediction_horizon'] # chunk size
        self.num_inference_timesteps = args_override['num_inference_timesteps']
        self.ema_power = args_override['ema_power']
        self.lr = args_override['lr']
        self.weight_decay = 0

        self.num_kp = 32
        self.feature_dimension = 64
        self.ac_dim = args_override['action_dim'] # 14 + 2
        self.obs_dim = self.feature_dimension * len(self.camera_names) + 14 # camera features and proprio

        backbones = []
        pools = []
        linears = []
        for _ in self.camera_names:
            backbones.append(ResNet18Conv(**{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False}))
            pools.append(SpatialSoftmax(**{'input_shape': [512, 15, 20], 'num_kp': self.num_kp, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0}))
            linears.append(torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))
        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)
        
        backbones = replace_bn_with_gn(backbones) # TODO


        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim*self.observation_horizon
        )

        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'backbones': backbones,
                'pools': pools,
                'linears': linears,
                'noise_pred_net': noise_pred_net
            })
        })

        nets = nets.float().cuda()
        ENABLE_EMA = True
        if ENABLE_EMA:
            ema = EMAModel(model=nets, power=self.ema_power)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        # setup noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )

        n_parameters = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_parameters/1e6,))


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


    def __call__(self, qpos, image, actions=None, is_pad=None):
        B = qpos.shape[0]
        if actions is not None: # training time
            nets = self.nets
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=obs_cond.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=obs_cond.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss
            all_l2 = F.mse_loss(noise_pred, noise, reduction='none')
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {}
            loss_dict['l2_loss'] = loss
            loss_dict['loss'] = loss

            if self.training and self.ema is not None:
                self.ema.step(nets)
            return loss_dict
        else: # inference time
            To = self.observation_horizon
            Ta = self.action_horizon
            Tp = self.prediction_horizon
            action_dim = self.ac_dim
            
            nets = self.nets
            if self.ema is not None:
                nets = self.ema.averaged_model
            
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action
            
            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction, 
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')
            status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.vq = args_override['vq']
        self.proprio_weight = args_override.get('proprio_weight', 10.0)
        self.qvel_weight = args_override.get('qvel_weight', 0.5)
        self.use_aux_proprio = args_override.get('use_aux_proprio', True)
        print(
            f'KL Weight {self.kl_weight}, '
            f'Proprio Weight {self.proprio_weight}, '
            f'Qvel Weight {self.qvel_weight}, '
            f'Aux Proprio {self.use_aux_proprio}'
        )

    def __call__(self, qpos, qvel, image, actions=None, is_pad=None, vq_sample=None, flow_data=None):
        env_state = None
        # print('AAAAAAImage shape in ACTPolicy:', image.shape)

        # # [旧代码] 只能处理 3 通道
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])

        # [新代码] 适配 3T 通道 (T 帧历史, 每帧 3 通道; 默认 T=2 与原 6 通道一致)
        # 对所有历史帧使用相同的 ImageNet 统计数据
        stats_mean = [0.485, 0.456, 0.406]
        stats_std = [0.229, 0.224, 0.225]
        T = getattr(self.model, 'visual_history_len', 2)

        normalize = transforms.Normalize(mean=stats_mean * T,
                                         std=stats_std * T)

        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            loss_dict = dict()
            # a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(qpos, image, env_state, actions, is_pad, vq_sample)
            a_hat, is_pad_hat, (mu, logvar), probs, binaries, (mu_prior, logvar_prior), pred_qpos, pred_qvel = self.model(
                qpos, image, env_state, actions, is_pad, vq_sample, flow_data=flow_data
            )
            if self.vq or self.model.encoder is None:
                total_kld = [torch.tensor(0.0)]
            elif getattr(self.model, 'use_learned_prior', True) and mu_prior is not None:
                kl_loss = kl_divergence_with_prior(mu, logvar, mu_prior, logvar_prior)
            else:
                kl_loss = kl_divergence(mu, logvar)[0]
            if self.vq:
                loss_dict['vq_discrepancy'] = F.l1_loss(probs, binaries, reduction='mean')
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            # loss_dict['kl'] = total_kld[0]
            loss_dict['kl'] = kl_loss

            # proprio_loss = F.mse_loss(pred_qpos, qpos)

            loss_qpos = F.mse_loss(pred_qpos, qpos)
            loss_qvel = F.mse_loss(pred_qvel, qvel) # 新增速度 Loss

            if self.use_aux_proprio and self.proprio_weight > 0:
                loss_dict['proprio'] = loss_qpos + self.qvel_weight * loss_qvel
            else:
                loss_dict['proprio'] = torch.tensor(0.0, device=qpos.device)

            # 总 Loss
            loss_dict['loss'] = loss_dict['l1'] + \
                                self.kl_weight * loss_dict['kl'] + \
                                self.proprio_weight * loss_dict['proprio']


            # loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            # a_hat, _, (_, _), _, _ = self.model(qpos, image, env_state, vq_sample=vq_sample) # no action, sample from prior
            a_hat, _, _, _, _, _, _, _ = self.model(qpos, image, env_state, vq_sample=vq_sample, flow_data=flow_data)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]

        _, _, binaries, _, _ = self.model.encode(qpos, actions, is_pad)

        return binaries
        
    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        try:
            return self.load_state_dict(model_dict)
        except RuntimeError:
            # 向后兼容: 旧论文 ckpt 训练时还没有 concat_proj (仅 concat_only 消融分支使用,
            # implicit 模式下不参与前向, 随机初始化不影响推理)
            own_keys = set(self.state_dict().keys())
            ckpt_keys = set(model_dict.keys())
            missing = own_keys - ckpt_keys
            unexpected = ckpt_keys - own_keys
            if missing and all(k.startswith('model.concat_proj.') for k in missing) and not unexpected:
                print(f'[deserialize] tolerating missing concat_proj keys (unused in implicit mode): {len(missing)} keys')
                return self.load_state_dict(model_dict, strict=False)
            raise


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        # # [旧代码] 只能处理 3 通道
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        
        # [新代码] 适配 6 通道 (Prev 3 + Curr 3)
        # 我们对上一帧和当前帧使用相同的 ImageNet 统计数据
        stats_mean = [0.485, 0.456, 0.406]
        stats_std = [0.229, 0.224, 0.225]
        
        normalize = transforms.Normalize(mean=stats_mean + stats_mean,  # [r,g,b, r,g,b]
                                         std=stats_std + stats_std)     # [r,g,b, r,g,b]

        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def kl_divergence_with_prior(mu, logvar, mu_prior, logvar_prior):
    """
    计算 KL( N(mu, std) || N(mu_prior, std_prior) )
    """
    # 确保维度一致
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    
    # 计算 sigma (标准差)
    std = torch.exp(0.5 * logvar)
    std_prior = torch.exp(0.5 * logvar_prior)

    # KL Divergence 公式 (General Gaussian)
    # term1 = log(std_prior / std)
    # term2 = (std^2 + (mu - mu_prior)^2) / (2 * std_prior^2)
    # result = term1 + term2 - 0.5
    
    term1 = 0.5 * (logvar_prior - logvar)
    term2 = (torch.exp(logvar) + (mu - mu_prior)**2) / (2 * torch.exp(logvar_prior))
    
    kld = term1 + term2 - 0.5
    return kld.sum(1).mean(0)
