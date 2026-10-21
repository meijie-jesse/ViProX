# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from .raft_flow import FlowEncoder, RaftFlowExtractor

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        num_queries,
        camera_names,
        vq,
        vq_class,
        vq_dim,
        action_dim,
        dynamics_mode='implicit',
        use_learned_prior=True,
        use_aux_proprio=True,
        visual_history_len=2,
        flow_image_size=(240, 320),
        raft_iters=20,
        raft_backend='torchvision',
        raft_checkpoint=None,
        raft_small=True,
    ):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.vq, self.vq_class, self.vq_dim = vq, vq_class, vq_dim
        self.state_dim, self.action_dim = state_dim, action_dim
        self.dynamics_mode = dynamics_mode
        self.use_learned_prior = use_learned_prior
        self.use_aux_proprio = use_aux_proprio
        self.visual_history_len = max(1, int(visual_history_len))
        if self.dynamics_mode == 'raft' and self.visual_history_len != 2:
            raise ValueError('RAFT dynamics requires exactly 2 frames (visual_history_len=2)')
        self.flow_image_size = flow_image_size
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding

        print(f'Use VQ: {self.vq}, {self.vq_class}, {self.vq_dim}')
        if self.vq:
            self.latent_proj = nn.Linear(hidden_dim, self.vq_class * self.vq_dim)
        else:
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        # self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq
        self.register_buffer('pos_table', get_sinusoid_encoding_table(2+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        if self.vq:
            self.latent_out_proj = nn.Linear(self.vq_class * self.vq_dim, hidden_dim)
        else:
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(1, hidden_dim) # learned position embedding for proprio and latent

        # 假设我们用过去 k=3 帧的特征平均或拼接来代表时序信息
        # 为了简单，我们这里假设输入虽然变了，但维度我们用 MLP 映射回 hidden_dim
        # self.temporal_projector = nn.Linear(self.transformer.d_model, self.transformer.d_model)

        # 2. 【核心创新】Learned Visual Prior (视觉先验网络)
        # 输入：视觉特征 (hidden_dim)
        # 输出：Latent 的均值和对数方差 (2 * latent_dim)

        # 这个 MLP 的作用是: 从 T 帧特征中"混合"出速度和位置信息
        # 输入维度 = T * hidden_dim (T=2 即论文默认设置)
        T = self.visual_history_len
        self.time_mixing_mlp = nn.Sequential(
            nn.Linear(hidden_dim * T, hidden_dim * T),
            nn.ReLU(),
            nn.Linear(hidden_dim * T, hidden_dim) # 压缩回 hidden_dim
        )
        self.concat_proj = nn.Sequential(
            nn.Linear(hidden_dim * T, hidden_dim * T),
            nn.ReLU(),
            nn.Linear(hidden_dim * T, hidden_dim),
        )

        if self.dynamics_mode == 'raft':
            self.raft_extractor = RaftFlowExtractor(
                flow_image_size=self.flow_image_size,
                num_iters=raft_iters,
                backend=raft_backend,
                checkpoint_path=raft_checkpoint,
                small=raft_small,
                input_format='imagenet',
            )
            self.flow_encoder = FlowEncoder(len(camera_names), out_dim=hidden_dim)
        else:
            self.raft_extractor = None
            self.flow_encoder = None

        # 这里的 temporal_projector 依然保留，用于进一步处理
        self.temporal_projector = nn.Linear(hidden_dim, hidden_dim)

        # self.temporal_projector = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim)
        # )
        # 预测头：预测 qpos，用于计算 Loss 
        # self.proprio_pred_head = nn.Linear(hidden_dim, state_dim)

        # 新增：预测qpos和qvel
        self.proprio_pred_head = nn.Linear(hidden_dim, state_dim * 2)

        self.visual_prior_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim * 2) # 输出 mu_prior, logvar_prior
        )

        print(
            f'DETRVAE dynamics_mode={self.dynamics_mode}, '
            f'use_learned_prior={self.use_learned_prior}, '
            f'use_aux_proprio={self.use_aux_proprio}, '
            f'flow_image_size={self.flow_image_size}, '
            f'raft_iters={raft_iters}, raft_backend={raft_backend}'
        )

    def _build_temporal_context(self, pooled_hist, image_prev=None, image_curr=None, flow_data=None):
        # pooled_hist: [B, T, hidden_dim] — 逐帧池化后的多视角全局特征 (最旧 -> 最新)
        if self.dynamics_mode == 'implicit':
            temporal_input = pooled_hist.flatten(1)  # [B, T*hidden_dim]
            return self.time_mixing_mlp(temporal_input)

        if self.dynamics_mode == 'concat_only':
            temporal_input = pooled_hist.flatten(1)
            return self.concat_proj(temporal_input)

        if self.dynamics_mode == 'raft':
            if flow_data is not None:
                flow = flow_data
            else:
                flows = []
                for cam_id in range(len(self.camera_names)):
                    flow_cam = self.raft_extractor(
                        image_prev[:, cam_id],
                        image_curr[:, cam_id],
                    )
                    flows.append(flow_cam)
                flow = torch.cat(flows, dim=1)
            return self.flow_encoder(flow)

        raise ValueError(f'Unknown dynamics_mode: {self.dynamics_mode}')

    # def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
    def encode(self, qpos, actions=None, is_pad=None, vq_sample=None, projected_qpos_embed=None, learned_prior=None):
        bs = actions.shape[0] if actions is not None else projected_qpos_embed.shape[0] # 获取 batch size
        # bs, _ = qpos.shape
        if self.encoder is None:
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            probs = binaries = mu = logvar = None
        else:
            # cvae encoder
            is_training = actions is not None # train or val
            ### Obtain latent z from action sequence
            if is_training:
                # project action sequence to embedding dim, and concat with a CLS token
                action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
                # qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
                # qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)

                qpos_embed = projected_qpos_embed # (bs, hidden_dim)
                qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                
                cls_embed = self.cls_embed.weight # (1, hidden_dim)
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
                # encoder_input = torch.cat([cls_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
                # do not mask cls token
                # cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
                cls_joint_is_pad = torch.full((bs, 2), False).to(projected_qpos_embed.device)
                # cls_joint_is_pad = torch.full((bs, 1), False).to(qpos.device) # False: not a padding
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
                # obtain position embedding
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
                # query model
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                encoder_output = encoder_output[0] # take cls output only
                latent_info = self.latent_proj(encoder_output)
                
                if self.vq:
                    logits = latent_info.reshape([*latent_info.shape[:-1], self.vq_class, self.vq_dim])
                    probs = torch.softmax(logits, dim=-1)
                    binaries = F.one_hot(torch.multinomial(probs.view(-1, self.vq_dim), 1).squeeze(-1), self.vq_dim).view(-1, self.vq_class, self.vq_dim).float()
                    binaries_flat = binaries.view(-1, self.vq_class * self.vq_dim)
                    probs_flat = probs.view(-1, self.vq_class * self.vq_dim)
                    straigt_through = binaries_flat - probs_flat.detach() + probs_flat
                    latent_input = self.latent_out_proj(straigt_through)
                    mu = logvar = None
                else:
                    probs = binaries = None
                    mu = latent_info[:, :self.latent_dim]
                    logvar = latent_info[:, self.latent_dim:]
                    latent_sample = reparametrize(mu, logvar)
                    latent_input = self.latent_out_proj(latent_sample)

            else:
                mu = logvar = binaries = probs = None
                if self.vq:
                    latent_input = self.latent_out_proj(vq_sample.view(-1, self.vq_class * self.vq_dim))
                else:

                    # latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                    # latent_input = self.latent_out_proj(latent_sample)

                    if learned_prior is not None:
                        mu_prior, logvar_prior = learned_prior
                        # 从预测的分布中采样 z
                        latent_sample = reparametrize(mu_prior, logvar_prior)
                    else:
                        # Fallback 到全0 (原始逻辑)
                        latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(projected_qpos_embed.device)
                    
                    latent_input = self.latent_out_proj(latent_sample)

        return latent_input, probs, binaries, mu, logvar

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, vq_sample=None, flow_data=None):
        """
        qpos: batch, qpos_dim
        qvel: batch, qvel_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        # latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)
        batch_size = image.shape[0]
        T = self.visual_history_len
        num_cam = image.shape[1]

        # 输入 image: (Batch, Cam, 3T, H, W), 通道按 [最旧帧 ... 最新帧] 排列
        image_hist = image.view(batch_size, num_cam, T, 3, image.shape[-2], image.shape[-1])
        # RAFT 路径仍使用倒数第二帧/最后一帧 (T 固定为 2, 见 __init__ 检查)
        image_prev = image_hist[:, :, -2] if T >= 2 else image_hist[:, :, -1]
        image_curr = image_hist[:, :, -1]

        # cvae decoder
        if self.backbones is not None:
            # Image observation features and position embeddings
            features_last = []  # per camera: [B, hidden_dim, h, w], 仅最新帧进 Transformer
            pos_last = []       # per camera: [B, hidden_dim, h, w]
            pooled_per_cam = [] # per camera: [B, T, hidden_dim], 全部 T 帧池化特征
            for cam_id, cam_name in enumerate(self.camera_names):
                # 把 T 帧 fold 进 batch 维, 一次 Siamese 前向 (T=2 时与逐帧两次调用等价)
                frames = image_hist[:, cam_id]                       # [B, T, 3, H, W]
                frames_flat = frames.reshape(batch_size * T, 3, *frames.shape[-2:])
                out_all, pos_all = self.backbones[cam_id](frames_flat)
                feat = self.input_proj(out_all[0])                   # [B*T, hidden_dim, h, w]
                h, w = feat.shape[-2:]
                feat = feat.view(batch_size, T, -1, h, w)
                pooled_per_cam.append(feat.mean(dim=[3, 4]))         # [B, T, hidden_dim]
                features_last.append(feat[:, -1])
                # 正弦位置编码与 batch 无关 ([1, hidden_dim, h, w]), 直接使用
                pos_last.append(pos_all[0])

            # --- 3. 准备 Transformer 的空间输入 (仅最新帧) ---
            # 将多摄像头的特征在宽度方向拼接 (ACT 标准做法)
            src_for_transformer = torch.cat(features_last, axis=3)
            pos_for_transformer = torch.cat(pos_last, axis=3)

            # 多相机池化特征在相机维取平均 -> [B, T, hidden_dim]
            pooled_hist = torch.stack(pooled_per_cam, dim=0).mean(dim=0)

            # --- 4. 动力学编码：隐式 Time-Mixing / 显式 RAFT / 多帧 concat ---
            temporal_context = self._build_temporal_context(
                pooled_hist,
                image_prev=image_prev,
                image_curr=image_curr,
                flow_data=flow_data,
            )

            # --- 5. 辅助任务与先验 (Auxiliary Heads) ---
            pred_proprio = self.proprio_pred_head(temporal_context)
            pred_qpos = pred_proprio[:, :self.state_dim]
            pred_qvel = pred_proprio[:, self.state_dim:]

            if self.use_learned_prior:
                prior_out = self.visual_prior_head(temporal_context)
                mu_prior = prior_out[:, :self.latent_dim]
                logvar_prior = prior_out[:, self.latent_dim:]
                learned_prior = (mu_prior, logvar_prior)
            else:
                mu_prior = logvar_prior = None
                learned_prior = None

            # --- 6. CVAE Encoder ---
            latent_input, probs, binaries, mu, logvar = self.encode(
                qpos=None, # 显式禁用 qpos
                actions=actions,
                is_pad=is_pad,
                vq_sample=vq_sample,
                projected_qpos_embed=temporal_context, # 传入隐式状态
                learned_prior=learned_prior,
            )


            # # visual_global_feat = src.mean(dim=0)
            # visual_global_feat = src.mean(dim=[2, 3]) 
            # temporal_context = self.temporal_projector(visual_global_feat) # [Batch, Hidden_Dim]
            # pred_qpos = self.proprio_pred_head(temporal_context)

            # proprio_input = temporal_context

            # prior_out = self.visual_prior_head(visual_global_feat)
            # mu_prior = prior_out[:, :self.latent_dim]
            # logvar_prior = prior_out[:, self.latent_dim:]


            # latent_input, probs, binaries, mu, logvar = self.encode(
            #     qpos=None, # 不再依赖真实 qpos
            #     actions=actions, 
            #     is_pad=is_pad, 
            #     vq_sample=vq_sample,
            #     # 传入我们的隐式状态
            #     projected_qpos_embed=temporal_context, 
            #     # 传入先验，用于推理时采样
            #     learned_prior=(mu_prior, logvar_prior) 
            # )
            proprio_input = temporal_context

            hs = self.transformer(src_for_transformer, None, self.query_embed.weight, pos_for_transformer, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        # return a_hat, is_pad_hat, [mu, logvar], probs, binaries 
        return a_hat, is_pad_hat, [mu, logvar], probs, binaries, [mu_prior, logvar_prior], pred_qpos, pred_qvel



class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + state_dim
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=self.action_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    transformer = build_transformer(args)

    if args.no_encoder:
        encoder = None
    else:
        encoder = build_encoder(args)

    dynamics_mode = getattr(args, 'dynamics_mode', 'implicit')
    use_learned_prior = getattr(args, 'use_learned_prior', True)
    use_aux_proprio = getattr(args, 'use_aux_proprio', True)
    visual_history_len = getattr(args, 'visual_history_len', 2)
    flow_image_size = tuple(getattr(args, 'flow_image_size', (240, 320)))
    raft_iters = getattr(args, 'raft_iters', 20)
    raft_backend = getattr(args, 'raft_backend', 'torchvision')
    raft_checkpoint = getattr(args, 'raft_checkpoint', None)
    raft_small = getattr(args, 'raft_small', True)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        vq=args.vq,
        vq_class=args.vq_class,
        vq_dim=args.vq_dim,
        action_dim=args.action_dim,
        dynamics_mode=dynamics_mode,
        use_learned_prior=use_learned_prior,
        use_aux_proprio=use_aux_proprio,
        visual_history_len=visual_history_len,
        flow_image_size=flow_image_size,
        raft_iters=raft_iters,
        raft_backend=raft_backend,
        raft_checkpoint=raft_checkpoint,
        raft_small=raft_small,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

