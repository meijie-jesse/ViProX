import torch
import numpy as np
import os
# Headless servers need an offscreen MuJoCo renderer before dm_control is imported.
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('MPLBACKEND', 'Agg')
import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy
from itertools import repeat
from collections import deque
from tqdm import tqdm
from einops import rearrange
# import wandb
import time
from torchvision import transforms

from constants import FPS
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose, sample_figure8_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action # helper functions
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy, FlowMatchingPolicy
from visualize_episodes import save_videos

from detr.models.latent_model import Latent_Model_Transformer

from sim_env import BOX_POSE

import IPython
e = IPython.embed

from AttentionVisualizer import AttentionVisualizer




def apply_precise_mask(obs, cam_idx, width=150, height=120, center_y=240):
    """
    用于训练/推理阶段的高性能函数
    obs: [Batch, Cam, Channel, H, W] or [Cam, Channel, H, W]
    """
    # 计算 Y 轴范围
    y_start = max(0, center_y - height // 2)
    y_end = min(obs.shape[-2], center_y + height // 2)
    
    w_dim = obs.shape[-1]
    
    # 注意：这里假设 obs 是 Tensor，支持切片操作
    # 针对 Batch 维度做兼容处理
    if obs.ndim == 5: # [B, Cam, C, H, W]
        target = obs[:, cam_idx, :, y_start:y_end, :]
        # 遮左边
        target[..., :width] = 0
        # 遮右边
        target[..., w_dim-width:] = 0
    else: # [Cam, C, H, W]
        target = obs[cam_idx, :, y_start:y_end, :]
        target[..., :width] = 0
        target[..., w_dim-width:] = 0
        
    return obs


def visualize_and_precise_mask(input_tensor, top_cam_index=None, 
                               mask_width=150,    # 左右遮挡的深度
                               mask_height=200,   # 机械臂的垂直厚度
                               vertical_center=None): # 机械臂的中心位置(默认在正中间)
    """
    input_tensor: [1, 3, 3, 480, 640]
    top_cam_index: 顶部摄像头的索引 (0-2)
    mask_width: 从左右边缘向内遮挡的距离 (px)
    mask_height: 遮挡区域的垂直高度 (px) - 这决定了保留上下多少空间
    vertical_center: 遮挡区域的垂直中心点 (px)。如果不填，默认为图像高度的一半 (240)。
    """
    
    # 1. 准备数据
    # Clone 一份以免修改原始数据，便于调试
    # 移除 Batch 维度: [3, 3, 480, 640]
    images = input_tensor[0].clone() 
    
    _, _, H, W = images.shape # H=480, W=640
    
    # 确定垂直中心
    if vertical_center is None:
        vertical_center = H // 2
        
    # 计算遮挡的垂直范围 (Y轴: 从 y_start 到 y_end)
    # 确保不超出图像边界
    y_start = max(0, vertical_center - mask_height // 2)
    y_end = min(H, vertical_center + mask_height // 2)

    # 2. 应用 Mask
    if top_cam_index is not None:
        print(f"正在处理 Camera {top_cam_index}...")
        print(f"遮挡范围: Y轴 [{y_start}:{y_end}], X轴 [0:{mask_width}] 和 [{W-mask_width}:{W}]")
        
        # 遮挡左侧机械臂区域
        images[top_cam_index, :, y_start:y_end, 0:mask_width] = 0
        
        # 遮挡右侧机械臂区域
        images[top_cam_index, :, y_start:y_end, W-mask_width:] = 0

    # 3. 可视化
    imgs_to_show = images.permute(0, 2, 3, 1).cpu().detach().numpy()
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    title_suffix = f" (Cam {top_cam_index} Masked)" if top_cam_index is not None else " (No Mask)"
    fig.suptitle(f'ACT Camera Observations{title_suffix}', fontsize=16)

    camera_names = ["Camera 0", "Camera 1", "Camera 2"]
    
    for i in range(3):
        axs[i].imshow(imgs_to_show[i])
        axs[i].set_title(camera_names[i])
        
        # 为了让你看清楚遮挡边界，我在图中画出红框辅助线 (仅用于显示，不影响数据)
        if i == top_cam_index:
            # 左侧框
            rect_left = patches.Rectangle((0, y_start), mask_width, y_end-y_start, 
                                          linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
            # 右侧框
            rect_right = patches.Rectangle((W-mask_width, y_start), mask_width, y_end-y_start, 
                                           linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
            axs[i].add_patch(rect_left)
            axs[i].add_patch(rect_right)
            
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 恢复 Batch 维度并返回
    return images.unsqueeze(0)



def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'qpos_{i}.npy')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_steps = args['num_steps']
    eval_every = args['eval_every']
    validate_every = args['validate_every']
    save_every = args['save_every']
    resume_ckpt_path = args['resume_ckpt_path']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim or task_name == 'all':
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    # num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.99)
    name_filter = task_config.get('name_filter', lambda n: True)

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'vq': args['use_vq'],
                         'vq_class': args['vq_class'],
                         'vq_dim': args['vq_dim'],
                         'action_dim': 16,
                         'no_encoder': args['no_encoder'],
                         'proprio_weight': args['proprio_weight'],
                         'qvel_weight': args['qvel_weight'],
                         'visual_history_len': args['visual_history_len'],
                         'use_aux_proprio': args['use_aux_proprio'],
                         'dynamics_mode': args['dynamics_mode'],
                         'use_learned_prior': args['use_learned_prior'],
                         'flow_image_size': (args['flow_h'], args['flow_w']),
                         'raft_iters': args['raft_iters'],
                         'raft_backend': args['raft_backend'],
                         'raft_checkpoint': args['raft_checkpoint'],
                         'raft_small': not args['raft_large'],
                         }
        
    # elif policy_class == 'FlowMatching': # 新增这个分支
    #     policy_config = {
    #         'lr': args['lr'],
    #         'camera_names': camera_names,
    #         'action_dim': 16, # 14 + 2 gripper
    #         # 'observation_horizon': 1,
    #         'action_horizon': 8,
    #         'prediction_horizon': args['chunk_size'],
    #         'num_inference_timesteps': 32, # FM 通常只需要 10 步甚至更少
    #         'vq': False,
    #         'num_queries': args['chunk_size'],
    #     }


    elif policy_class == 'FlowMatching':
        policy_config = {'lr': args['lr'],
                         'camera_names': camera_names,
                         'action_dim': 16,
                         'observation_horizon': 1,
                         'action_horizon': 8,
                         'prediction_horizon': args['chunk_size'],
                         'num_queries': args['chunk_size'],
                         'num_inference_timesteps': 10, # Flow Matching 10步通常足够
                         'ema_power': 0.75,
                         'vq': False,
                         }

    elif policy_class == 'Diffusion':

        policy_config = {'lr': args['lr'],
                         'camera_names': camera_names,
                         'action_dim': 16,
                         'observation_horizon': 1,
                         'action_horizon': 8,
                         'prediction_horizon': args['chunk_size'],
                         'num_queries': args['chunk_size'],
                         'num_inference_timesteps': 10,
                         'ema_power': 0.75,
                         'vq': False,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    actuator_config = {
        'actuator_network_dir': args['actuator_network_dir'],
        'history_len': args['history_len'],
        'future_len': args['future_len'],
        'prediction_len': args['prediction_len'],
    }

    config = {
        'num_steps': num_steps,
        'eval_every': eval_every,
        'validate_every': validate_every,
        'save_every': save_every,
        'resume_save_every': args.get('resume_save_every', 0),
        'ckpt_dir': ckpt_dir,
        'resume_ckpt_path': resume_ckpt_path,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'load_pretrain': args['load_pretrain'],
        'actuator_config': actuator_config,
        'dynamics_mode': args['dynamics_mode'],
        'use_learned_prior': args['use_learned_prior'],
        'use_aux_proprio': args['use_aux_proprio'],
        'eval_photo_perturb': args['eval_photo_perturb'],
        'eval_frame_drop': args['eval_frame_drop'],
        'eval_frame_stride': args['eval_frame_stride'],
        'eval_qpos_noise_std': args['eval_qpos_noise_std'],
        'eval_qpos_drift': args['eval_qpos_drift'],
        'eval_qpos_dropout': args['eval_qpos_dropout'],
    }

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    expr_name = ckpt_dir.split('/')[-1]
    if not is_eval:
        # wandb.init(project="mobile-aloha2", reinit=True, entity="mobile-aloha2", name=expr_name)
        # wandb.config.update(config)
        # 评测模式下不回写 config.pkl, 避免污染被评测 ckpt 目录 (例如论文主实验目录)
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
    if is_eval:
        ckpt_names = [f'policy_last.ckpt']
        results = []

        # visualize_latent_dynamics(config, 'policy_last.ckpt')

        for ckpt_name in ckpt_names:
            # success_rate, avg_return = eval_bc_new(config, ckpt_name, save_episode=True, num_rollouts=10)  # 出图 计算 t-SNE
            num_rollouts = int(os.environ.get('VIPROX_EVAL_NUM_ROLLOUTS', '50'))  # override for quick smoke evals
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=num_rollouts)  # 原版
            # wandb.log({'success_rate': success_rate, 'avg_return': avg_return})
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir, name_filter, camera_names, batch_size_train, batch_size_val,
        args['chunk_size'], args['skip_mirrored_data'], config['load_pretrain'],
        policy_class, stats_dir_l=stats_dir, sample_weights=sample_weights,
        train_ratio=train_ratio,
        dynamics_mode=args['dynamics_mode'],
        flow_image_size=(args['flow_h'], args['flow_w']),
        visual_history_len=args['visual_history_len'],
        photo_aug=args['photo_aug'],
        frame_drop_prob=args['frame_drop_prob'],
    )

    # save dataset stats
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
    # wandb.finish()


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    elif policy_class == 'FlowMatching':
        policy = FlowMatchingPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'Diffusion':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'FlowMatching':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)
    
    return curr_image


class EvalPhotoPerturber:
    """评测期图像域光度扰动 (仿真光照 OOD 的代理).
    作用于 float [0,1] 图像 tensor [1, C, 3, H, W].
    - mild:   固定的中等 gamma/亮度偏移 (静态光照变化)
    - severe: 每个 rollout 重采样的强曝光/色偏 + 缓慢正弦时变调制 (模拟移动光源/动态阴影)
    """
    def __init__(self, mode, seed=0):
        self.mode = mode
        self.rng = np.random.RandomState(seed)
        self.resample()

    def resample(self):
        if self.mode == 'mild':
            self.gamma0, self.bright0, self.sat = 1.30, 1.15, 0.90
            self.mod_amp, self.mod_period = 0.0, 200.0
        elif self.mode == 'severe':
            self.gamma0 = self.rng.uniform(1.50, 1.90)   # 强 gamma 偏移 (压暗)
            self.bright0 = self.rng.uniform(0.55, 0.75)  # 强曝光下降
            self.sat = self.rng.uniform(0.50, 1.50)      # 色偏
            self.mod_amp, self.mod_period = 0.25, 150.0  # 时变调制 -> 移动光源代理

    def __call__(self, img, t):
        if self.mode == 'none':
            return img
        phase = 2.0 * np.pi * t / self.mod_period
        gamma = self.gamma0 + self.mod_amp * np.sin(phase)
        bright = self.bright0 + 0.15 * np.sin(phase + 1.0)
        x = torch.clamp(img * bright, 0.0, 1.0) ** gamma
        # saturation: 逐相机在通道维上与亮度图混合 ([1, C, 3, H, W])
        gray = (0.299 * x[:, :, 0:1] + 0.587 * x[:, :, 1:2] + 0.114 * x[:, :, 2:3])
        x = gray + (x - gray) * self.sat
        return torch.clamp(x, 0.0, 1.0)


def eval_bc(config, ckpt_name, save_episode=True, num_rollouts=50):   # 原版
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    vq = config['policy_config']['vq']
    actuator_config = config['actuator_config']
    use_actuator_net = actuator_config['actuator_network_dir'] is not None
    # 视觉历史长度与评测扰动配置
    T_hist = int(config['policy_config'].get('visual_history_len', 2))
    eval_photo_perturb = config.get('eval_photo_perturb', 'none')
    eval_frame_drop = float(config.get('eval_frame_drop', 0.0))
    eval_frame_stride = max(1, int(config.get('eval_frame_stride', 1)))
    eval_qpos_noise_std = float(config.get('eval_qpos_noise_std', 0.0))
    eval_qpos_drift = float(config.get('eval_qpos_drift', 0.0))
    eval_qpos_dropout = float(config.get('eval_qpos_dropout', 0.0))
    photo_perturber = EvalPhotoPerturber(eval_photo_perturb) if eval_photo_perturb != 'none' else None
    if eval_photo_perturb != 'none' or eval_frame_drop > 0 or eval_qpos_noise_std > 0 or eval_qpos_drift != 0 or eval_qpos_dropout > 0:
        print(f'[eval-stress] photo_perturb={eval_photo_perturb}, frame_drop={eval_frame_drop}, T_hist={T_hist}, frame_stride={eval_frame_stride}, '
              f'qpos_noise_std={eval_qpos_noise_std}, qpos_drift={eval_qpos_drift}, qpos_dropout={eval_qpos_dropout}')

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    if vq:
        vq_dim = config['policy_config']['vq_dim']
        vq_class = config['policy_config']['vq_class']
        latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
        latent_model_ckpt_path = os.path.join(ckpt_dir, 'latent_model_last.ckpt')
        latent_model.deserialize(torch.load(latent_model_ckpt_path))
        latent_model.eval()
        latent_model.cuda()
        print(f'Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}')
    else:
        print(f'Loaded: {ckpt_path}')


    # 如果你要把图存在 checkpoint 目录下的 attn_vis 文件夹里
    vis_save_dir = os.path.join(ckpt_dir, 'attn_vis_results')
    visualizer = AttentionVisualizer(policy.model, save_dir=vis_save_dir)

    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    # if use_actuator_net:
    #     prediction_len = actuator_config['prediction_len']
    #     future_len = actuator_config['future_len']
    #     history_len = actuator_config['history_len']
    #     actuator_network_dir = actuator_config['actuator_network_dir']

    #     from act.train_actuator_network import ActuatorNetwork
    #     actuator_network = ActuatorNetwork(prediction_len)
    #     actuator_network_path = os.path.join(actuator_network_dir, 'actuator_net_last.ckpt')
    #     loading_status = actuator_network.load_state_dict(torch.load(actuator_network_path))
    #     actuator_network.eval()
    #     actuator_network.cuda()
    #     print(f'Loaded actuator network from: {actuator_network_path}, {loading_status}')

    #     actuator_stats_path  = os.path.join(actuator_network_dir, 'actuator_net_stats.pkl')
    #     with open(actuator_stats_path, 'rb') as f:
    #         actuator_stats = pickle.load(f)
        
    #     actuator_unnorm = lambda x: x * actuator_stats['commanded_speed_std'] + actuator_stats['commanded_speed_std']
    #     actuator_norm = lambda x: (x - actuator_stats['observed_speed_mean']) / actuator_stats['observed_speed_mean']
    #     def collect_base_action(all_actions, norm_episode_all_base_actions):
    #         post_processed_actions = post_process(all_actions.squeeze(0).cpu().numpy())
    #         norm_episode_all_base_actions += actuator_norm(post_processed_actions[:, -2:]).tolist()

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True, setup_robots=True, setup_base=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    if real_robot:
        BASE_DELAY = 13
        query_frequency -= BASE_DELAY

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        if real_robot:
            e()
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset

        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset
        
        elif 'sim_figure8_scripted' in task_name:
            # 必须与录制端约定一致 (ee_sim_env.py: box_right_pose, box_left_pose = sample_figure8_pose()):
            # box_left(黄) <- socket_pose (x=-0.2 侧), box_right(品红) <- peg_pose (x=+0.2 侧)
            # 直接 concatenate(sample_figure8_pose()) 会把两个盒子左右对调, 奖励接触对永远不匹配
            box_right_pose, box_left_pose = sample_figure8_pose()
            BOX_POSE[0] = np.concatenate([box_left_pose, box_right_pose]) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, 16]).cuda()

        # qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        qpos_history_raw = np.zeros((max_timesteps, state_dim))
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        phys_traj = []  # per-step physics qpos (boxes at [-14:]), for geometric stage metrics
        rewards = []
        # if use_actuator_net:
        #     norm_episode_all_base_actions = [actuator_norm(np.zeros(history_len, 2)).tolist()]
        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            culmulated_delay = 0 
            frame_buffer = deque(maxlen=T_hist)  # 帧历史缓冲 (每 eval_frame_stride 步更新)
            held_image = None  # 评测帧丢弃模拟: 被保持的上一帧
            if photo_perturber is not None:
                photo_perturber.resample()  # 每个 rollout 重新采样光照参数
            for t in range(max_timesteps):
                time1 = time.time()
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                time2 = time.time()
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos_history_raw[t] = qpos_numpy
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                # qpos_history[:, t] = qpos

                # Scripted proprioceptive corruptions (OOD robustness study; applies to
                # baselines that consume qpos -- ViProX is vision-only and unaffected)
                if eval_qpos_noise_std > 0:
                    qpos = qpos + torch.normal(0.0, eval_qpos_noise_std, size=qpos.shape).cuda()
                if eval_qpos_drift != 0:
                    qpos = qpos + eval_qpos_drift * t
                if eval_qpos_dropout > 0:
                    qpos = qpos * (torch.rand_like(qpos) >= eval_qpos_dropout).float()

                # 每步获取观测帧 (渲染由 env.step 产生, 无额外开销);
                # 按 eval_frame_stride 采入帧缓冲 (默认 1 = 与训练相邻帧一致)
                curr_image = get_image(ts, camera_names, rand_crop_resize=(config['policy_class'] == 'Diffusion' or config['policy_class'] == 'FlowMatching'))
                if photo_perturber is not None:
                    curr_image = photo_perturber(curr_image, t)
                if eval_frame_drop > 0 and held_image is not None:
                    # 帧保持: 逐相机以概率 p 丢弃当前帧, 沿用上一帧
                    drop_mask = torch.rand(curr_image.shape[1], device=curr_image.device) < eval_frame_drop
                    curr_image = torch.where(drop_mask.view(1, -1, 1, 1, 1), held_image, curr_image)
                held_image = curr_image
                if t % eval_frame_stride == 0:
                    frame_buffer.append(curr_image)
                if t % query_frequency == 0:
                    # 不足 T 帧时用最旧帧填充 (序列开头)
                    frames = list(frame_buffer)
                    while len(frames) < T_hist:
                        frames.insert(0, frames[0])
                    image = torch.cat(frames, dim=2)  # [1, C, 3T, H, W]
                # print('get image: ', time.time() - time2)

                # visualize_and_precise_mask(curr_image, 0, 200, 70, 295)
                # curr_image = apply_precise_mask(curr_image, 0, 200, 70, 295)

                if t == 0:
                    # warm up
                    for _ in range(10):
                        # print('aaaaa', image.shape)
                        policy(qpos=None, qvel=None, image=image)
                    print('network warm up done')
                    time1 = time.time()

                ### query policy
                time3 = time.time()
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        if vq:
                            if rollout_id == 0:
                                for _ in range(10):
                                    vq_sample = latent_model.generate(1, temperature=1, x=None)
                                    print(torch.nonzero(vq_sample[0])[:, 1].cpu().numpy())
                            vq_sample = latent_model.generate(1, temperature=1, x=None)
                            all_actions = policy(qpos, curr_image, vq_sample=vq_sample)
                        else:
                            # e()
                            # print(qpos)   
                            all_actions = policy(qpos=None, qvel=None, image=image)

                            # if rollout_id == 0 and t % 10 == 0:


                        # if use_actuator_net:
                        #     collect_base_action(all_actions, norm_episode_all_base_actions)
                        if real_robot:
                            all_actions = torch.cat([all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)

                    # if t % 20 == 0:  # attention可视化 1209
                        visualizer.visualize(curr_image, camera_names, t, query_index=0)


                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                        # if t % query_frequency == query_frequency - 1:
                        #     # zero out base actions to avoid overshooting
                        #     raw_action[0, -2:] = 0
                elif config['policy_class'] == "Diffusion":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                        # if use_actuator_net:
                        #     collect_base_action(all_actions, norm_episode_all_base_actions)
                        if real_robot:
                            all_actions = torch.cat([all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
                    raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                    all_actions = raw_action.unsqueeze(0)
                    # if use_actuator_net:
                    #     collect_base_action(all_actions, norm_episode_all_base_actions)
                elif config['policy_class'] == "FlowMatching":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                        print(f"Pred Mean: {all_actions.mean().item():.3f}, Max: {all_actions.max().item():.3f}, Min: {all_actions.min().item():.3f}")  # 监测归一化
                        # raw_action = policy(qpos, curr_image)
                        # all_actions = raw_action.unsqueeze(0)
                        if real_robot:
                            all_actions = torch.cat([all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
                    raw_action = all_actions[:, t % query_frequency]

                else:
                    raise NotImplementedError
                # print('query policy: ', time.time() - time3)

                ### post-process actions
                time4 = time.time()
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action[:-2]

                # if use_actuator_net:
                #     assert(not temporal_agg)
                #     if t % prediction_len == 0:
                #         offset_start_ts = t + history_len
                #         actuator_net_in = np.array(norm_episode_all_base_actions[offset_start_ts - history_len: offset_start_ts + future_len])
                #         actuator_net_in = torch.from_numpy(actuator_net_in).float().unsqueeze(dim=0).cuda()
                #         pred = actuator_network(actuator_net_in)
                #         base_action_chunk = actuator_unnorm(pred.detach().cpu().numpy()[0])
                #     base_action = base_action_chunk[t % prediction_len]
                # else:
                base_action = action[-2:]
                # base_action = calibrate_linear_vel(base_action, c=0.19)
                # base_action = postprocess_base_action(base_action)
                # print('post process: ', time.time() - time4)

                ### step the environment
                time5 = time.time()
                if real_robot:
                    ts = env.step(target_qpos, base_action)
                else:
                    ts = env.step(target_qpos)
                # print('step env: ', time.time() - time5)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
                # 记录物理 qpos (含两个方块 free joint, 供 figure8 Cross/Comp 几何判据事后分析)
                phys_traj.append(env._physics.data.qpos.copy())
                duration = time.time() - time1
                sleep_time = max(0, DT - duration)
                # print(sleep_time)
                time.sleep(sleep_time)
                # time.sleep(max(0, DT - duration - culmulated_delay))
                if duration >= DT:
                    culmulated_delay += (duration - DT)
                    print(f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s')
                # else:
                #     culmulated_delay = max(0, culmulated_delay - (DT - duration))

            print(f'Avg fps {max_timesteps / (time.time() - time0)}')
            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            # save qpos_history_raw
            log_id = get_auto_index(ckpt_dir)
            np.save(os.path.join(ckpt_dir, f'qpos_{log_id}.npy'), qpos_history_raw)
            plt.figure(figsize=(10, 20))
            # plot qpos_history_raw for each qpos dim using subplots
            for i in range(state_dim):
                plt.subplot(state_dim, 1, i+1)
                plt.plot(qpos_history_raw[:, i])
                # remove x axis
                if i != state_dim - 1:
                    plt.xticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, f'qpos_{log_id}.png'))
            plt.close()


        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))
            np.save(os.path.join(ckpt_dir, f'phys_traj{rollout_id}.npy'), np.array(phys_traj))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def eval_bc_new(config, ckpt_name, save_episode=True, num_rollouts=1): # 出图 计算 t-SNE

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    set_seed(16037)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    vq = config['policy_config']['vq']
    actuator_config = config['actuator_config']
    use_actuator_net = actuator_config['actuator_network_dir'] is not None

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    if vq:
        vq_dim = config['policy_config']['vq_dim']
        vq_class = config['policy_config']['vq_class']
        latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
        latent_model_ckpt_path = os.path.join(ckpt_dir, 'latent_model_last.ckpt')
        latent_model.deserialize(torch.load(latent_model_ckpt_path))
        latent_model.eval()
        latent_model.cuda()
        print(f'Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}')
    else:
        print(f'Loaded: {ckpt_path}')


    # 如果你要把图存在 checkpoint 目录下的 attn_vis 文件夹里
    vis_save_dir = os.path.join(ckpt_dir, 'attn_vis_results')
    # visualizer = AttentionVisualizer(policy.action_head, save_dir=vis_save_dir)

    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    # if use_actuator_net:
    #     prediction_len = actuator_config['prediction_len']
    #     future_len = actuator_config['future_len']
    #     history_len = actuator_config['history_len']
    #     actuator_network_dir = actuator_config['actuator_network_dir']

    #     from act.train_actuator_network import ActuatorNetwork
    #     actuator_network = ActuatorNetwork(prediction_len)
    #     actuator_network_path = os.path.join(actuator_network_dir, 'actuator_net_last.ckpt')
    #     loading_status = actuator_network.load_state_dict(torch.load(actuator_network_path))
    #     actuator_network.eval()
    #     actuator_network.cuda()
    #     print(f'Loaded actuator network from: {actuator_network_path}, {loading_status}')

    #     actuator_stats_path  = os.path.join(actuator_network_dir, 'actuator_net_stats.pkl')
    #     with open(actuator_stats_path, 'rb') as f:
    #         actuator_stats = pickle.load(f)
        
    #     actuator_unnorm = lambda x: x * actuator_stats['commanded_speed_std'] + actuator_stats['commanded_speed_std']
    #     actuator_norm = lambda x: (x - actuator_stats['observed_speed_mean']) / actuator_stats['observed_speed_mean']
    #     def collect_base_action(all_actions, norm_episode_all_base_actions):
    #         post_processed_actions = post_process(all_actions.squeeze(0).cpu().numpy())
    #         norm_episode_all_base_actions += actuator_norm(post_processed_actions[:, -2:]).tolist()

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True, setup_robots=True, setup_base=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    if real_robot:
        BASE_DELAY = 13
        query_frequency -= BASE_DELAY

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        if real_robot:
            e()
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset

        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset
        
        elif 'sim_figure8_scripted' in task_name:
            # 必须与录制端约定一致 (ee_sim_env.py: box_right_pose, box_left_pose = sample_figure8_pose()):
            # box_left(黄) <- socket_pose (x=-0.2 侧), box_right(品红) <- peg_pose (x=+0.2 侧)
            # 直接 concatenate(sample_figure8_pose()) 会把两个盒子左右对调, 奖励接触对永远不匹配
            box_right_pose, box_left_pose = sample_figure8_pose()
            BOX_POSE[0] = np.concatenate([box_left_pose, box_right_pose]) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, 16]).cuda()

        # qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        qpos_history_raw = np.zeros((max_timesteps, state_dim))
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        phys_traj = []  # per-step physics qpos (boxes at [-14:]), for geometric stage metrics
        rewards = []

        latent_vectors = []  # 存储 mu_prior (来自视觉先验)
        trajectory_points = [] # 存储实际 qpos (用于画 3D 轨迹参考)
        velocities = []      # 存储实际 qvel (用于颜色编码方向)
        timesteps = []       # 存储时间步 t

        # if use_actuator_net:
        #     norm_episode_all_base_actions = [actuator_norm(np.zeros(history_len, 2)).tolist()]
        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            culmulated_delay = 0 
            prev_image = None  # 存储上一个时间步的图像
            for t in range(max_timesteps):
                time1 = time.time()
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                time2 = time.time()
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos_history_raw[t] = qpos_numpy
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                # qpos_history[:, t] = qpos

                # Scripted proprioceptive corruptions (same flags as eval_bc)
                if float(config.get('eval_qpos_noise_std', 0.0)) > 0:
                    qpos = qpos + torch.normal(0.0, float(config.get('eval_qpos_noise_std', 0.0)), size=qpos.shape).cuda()
                if float(config.get('eval_qpos_drift', 0.0)) != 0:
                    qpos = qpos + float(config.get('eval_qpos_drift', 0.0)) * t
                if float(config.get('eval_qpos_dropout', 0.0)) > 0:
                    qpos = qpos * (torch.rand_like(qpos) >= float(config.get('eval_qpos_dropout', 0.0))).float()

                if t % query_frequency == 0:
                    curr_image = get_image(ts, camera_names, rand_crop_resize=(config['policy_class'] == 'Diffusion' or config['policy_class'] == 'FlowMatching'))
                    if prev_image is None:  # 第一次图像，复制当前帧
                        prev_image = curr_image.clone()   
                    
                    image = torch.cat([prev_image, curr_image], dim=2)  # 在通道维度上拼接
                    # np.concatenate([img_prev, img_curr], axis=2)
                    prev_image = curr_image.clone()  # 更新为当前帧
                    # print('image shape: ', curr_image.shape)
                # print('get image: ', time.time() - time2)

                # visualize_and_precise_mask(curr_image, 0, 200, 70, 295)
                # curr_image = apply_precise_mask(curr_image, 0, 200, 70, 295)

                if t == 0:
                    # warm up
                    for _ in range(10):
                        # print('aaaaa', image.shape)
                        policy(qpos=None, qvel=None, image=image)
                    print('network warm up done')
                    time1 = time.time()

                ### query policy
                time3 = time.time()
                if config['policy_class'] == "ACT":

                    if t % 1 == 0:

                        curr_image = get_image(ts, camera_names, rand_crop_resize=(config['policy_class'] == 'Diffusion' or config['policy_class'] == 'FlowMatching'))
                        if prev_image is None:  # 第一次图像，复制当前帧
                            prev_image = curr_image.clone()   
                        
                        image = torch.cat([prev_image, curr_image], dim=2)  # 在通道维度上拼接
                        # np.concatenate([img_prev, img_curr], axis=2)
                        prev_image = curr_image.clone()  # 更新为当前帧

                        stats_mean = [0.485, 0.456, 0.406]
                        stats_std = [0.229, 0.224, 0.225]
    
                        normalize = transforms.Normalize(mean=stats_mean + stats_mean,  # [r,g,b, r,g,b]
                                        std=stats_std + stats_std)     # [r,g,b, r,g,b]

                        image1 = normalize(image)

                        model_output1  = policy.model(qpos=None, image=image1, env_state=None)
                        # if rollout_id == 0 and t % 10 == 0:

                        # all_actions = model_output[0] # a_hat

                        mu_prior1 = model_output1[5][0] # Index 5 is [mu_prior, logvar_prior], 0 is mu

                        latent_vectors.append(mu_prior1.cpu().numpy().flatten())
                        trajectory_points.append(qpos_numpy) # 记录真实位置
                        timesteps.append(t)

                        # 3. 计算速度 (用于后续画图的方向上色)
                        # 计算简单的速度方向 (当前位置 - 上一位置)
                        if len(trajectory_points) >= 2:
                            vel = trajectory_points[-1] - trajectory_points[-2]
                            velocities.append(vel)
                        else:
                            # 第一个点，速度设为0
                            velocities.append(np.zeros_like(qpos_numpy))



                    if t % query_frequency == 0:
                        if vq:
                            if rollout_id == 0:
                                for _ in range(10):
                                    vq_sample = latent_model.generate(1, temperature=1, x=None)
                                    print(torch.nonzero(vq_sample[0])[:, 1].cpu().numpy())
                            vq_sample = latent_model.generate(1, temperature=1, x=None)
                            all_actions = policy(qpos, curr_image, vq_sample=vq_sample)
                        else:
                            # e()
                            # print(qpos)   
                            all_actions = policy(qpos=None, qvel=None, image=image)

                            # stats_mean = [0.485, 0.456, 0.406]
                            # stats_std = [0.229, 0.224, 0.225]
        
                            # normalize = transforms.Normalize(mean=stats_mean + stats_mean,  # [r,g,b, r,g,b]
                            #              std=stats_std + stats_std)     # [r,g,b, r,g,b]

                            # image = normalize(image)

                            # model_output  = policy.model(qpos=None, image=image, env_state=None)
                            # # if rollout_id == 0 and t % 10 == 0:

                            # all_actions = model_output[0] # a_hat

                            # mu_prior = model_output[5][0] # Index 5 is [mu_prior, logvar_prior], 0 is mu

                            # latent_vectors.append(mu_prior.cpu().numpy().flatten())
                            # trajectory_points.append(qpos_numpy) # 记录真实位置
                            # timesteps.append(t)

                            # # 计算简单的速度方向 (当前位置 - 上一位置)
                            # if len(trajectory_points) >= 2:
                            #     vel = trajectory_points[-1] - trajectory_points[-2]
                            #     velocities.append(vel)
                            # else:
                            #     # 第一个点，速度设为0
                            #     velocities.append(np.zeros_like(qpos_numpy))



                        # if use_actuator_net:
                        #     collect_base_action(all_actions, norm_episode_all_base_actions)
                        if real_robot:
                            all_actions = torch.cat([all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)

                    # if t % 10 == 0:  # attention可视化 1209
                    #         visualizer.visualize(curr_image, camera_names, t, query_index=0)


                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                        # if t % query_frequency == query_frequency - 1:
                        #     # zero out base actions to avoid overshooting
                        #     raw_action[0, -2:] = 0
                elif config['policy_class'] == "Diffusion":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                        # if use_actuator_net:
                        #     collect_base_action(all_actions, norm_episode_all_base_actions)
                        if real_robot:
                            all_actions = torch.cat([all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
                    raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                    all_actions = raw_action.unsqueeze(0)
                    # if use_actuator_net:
                    #     collect_base_action(all_actions, norm_episode_all_base_actions)
                elif config['policy_class'] == "FlowMatching":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                        print(f"Pred Mean: {all_actions.mean().item():.3f}, Max: {all_actions.max().item():.3f}, Min: {all_actions.min().item():.3f}")  # 监测归一化
                        # raw_action = policy(qpos, curr_image)
                        # all_actions = raw_action.unsqueeze(0)
                        if real_robot:
                            all_actions = torch.cat([all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
                    raw_action = all_actions[:, t % query_frequency]

                else:
                    raise NotImplementedError
                # print('query policy: ', time.time() - time3)

                ### post-process actions
                time4 = time.time()
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action[:-2]

                # if use_actuator_net:
                #     assert(not temporal_agg)
                #     if t % prediction_len == 0:
                #         offset_start_ts = t + history_len
                #         actuator_net_in = np.array(norm_episode_all_base_actions[offset_start_ts - history_len: offset_start_ts + future_len])
                #         actuator_net_in = torch.from_numpy(actuator_net_in).float().unsqueeze(dim=0).cuda()
                #         pred = actuator_network(actuator_net_in)
                #         base_action_chunk = actuator_unnorm(pred.detach().cpu().numpy()[0])
                #     base_action = base_action_chunk[t % prediction_len]
                # else:
                base_action = action[-2:]
                # base_action = calibrate_linear_vel(base_action, c=0.19)
                # base_action = postprocess_base_action(base_action)
                # print('post process: ', time.time() - time4)

                ### step the environment
                time5 = time.time()
                if real_robot:
                    ts = env.step(target_qpos, base_action)
                else:
                    ts = env.step(target_qpos)
                # print('step env: ', time.time() - time5)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
                # 记录物理 qpos (含两个方块 free joint, 供 figure8 Cross/Comp 几何判据事后分析)
                phys_traj.append(env._physics.data.qpos.copy())
                duration = time.time() - time1
                sleep_time = max(0, DT - duration)
                # print(sleep_time)
                time.sleep(sleep_time)
                # time.sleep(max(0, DT - duration - culmulated_delay))
                if duration >= DT:
                    culmulated_delay += (duration - DT)
                    print(f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s')
                # else:
                #     culmulated_delay = max(0, culmulated_delay - (DT - duration))

            print(f'Avg fps {max_timesteps / (time.time() - time0)}')
            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            # save qpos_history_raw
            log_id = get_auto_index(ckpt_dir)
            np.save(os.path.join(ckpt_dir, f'qpos_{log_id}.npy'), qpos_history_raw)
            plt.figure(figsize=(10, 20))
            # plot qpos_history_raw for each qpos dim using subplots
            for i in range(state_dim):
                plt.subplot(state_dim, 1, i+1)
                plt.plot(qpos_history_raw[:, i])
                # remove x axis
                if i != state_dim - 1:
                    plt.xticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, f'qpos_{log_id}.png'))
            plt.close()


        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))
            np.save(os.path.join(ckpt_dir, f'phys_traj{rollout_id}.npy'), np.array(phys_traj))

        


        # 3. t-SNE 分析
        print("正在计算 t-SNE...")
        latent_vectors = np.array(latent_vectors)
        trajectory_points = np.array(trajectory_points)
        
        # 使用 perplexity=30 通常对轨迹数据效果较好
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        z_embedded = tsne.fit_transform(latent_vectors)

        # 4. 可视化绘图
        fig = plt.figure(figsize=(16, 6))

        # 子图1：物理空间轨迹 (Ground Truth)
        ax1 = fig.add_subplot(121)
        # 假设前两个关节决定了平面位置 (根据具体机器人调整，比如 x, y)
        # 对于 Aloha/Viper 这种臂，通常用末端坐标更好，这里简单用 qpos[0] vs qpos[1] 展示
        # 如果是 Wipe 任务，建议画末端 (End Effector) 轨迹
        sc1 = ax1.scatter(trajectory_points[:, 0], trajectory_points[:, 1], c=timesteps, cmap='viridis', s=10)
        ax1.set_title('Robot Trajectory (Physical Space)')
        ax1.set_xlabel('Joint 0 / X')
        ax1.set_ylabel('Joint 1 / Y')
        plt.colorbar(sc1, ax=ax1, label='Time Step')
        
        # 标记交叉点 (示意)
        center_idx = len(timesteps) // 2
        # ax1.scatter(trajectory_points[center_idx, 0], trajectory_points[center_idx, 1], c='red', s=100, marker='x', label='Crossing')

        # 子图2：Latent Space (t-SNE)
        ax2 = fig.add_subplot(122)
        sc2 = ax2.scatter(z_embedded[:, 0], z_embedded[:, 1], c=timesteps, cmap='viridis', s=15, alpha=0.8)
        
        # 关键：连接点以显示轨迹连续性
        ax2.plot(z_embedded[:, 0], z_embedded[:, 1], c='gray', alpha=0.3, linewidth=0.5)
        
        ax2.set_title('Latent Space Visualization (t-SNE)')
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')
        
        # # 添加文字说明
        # text_str = "Observation: Similar Images at Crossing\nLatent Space: Distinct Clusters based on Direction"
        # ax2.text(0.05, 0.95, text_str, transform=ax2.transAxes, fontsize=10,
        #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.colorbar(sc2, ax=ax2, label='Time Step')
        plt.suptitle(f'Figure 8 Task: Disentangling Dynamics from Vision\n({ckpt_name})', fontsize=16)
        
        save_path = os.path.join(ckpt_dir, 'latent_tsne_analysis.png')
        plt.savefig(save_path, dpi=300)
        print(f"图表已保存至: {save_path}")
        plt.show()

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, qvel_data, action_data, is_pad, flow_data = data
    image_data = image_data.cuda()
    qpos_data = qpos_data.cuda()
    qvel_data = qvel_data.cuda()
    action_data = action_data.cuda()
    is_pad = is_pad.cuda()
    if flow_data is not None:
        flow_data = flow_data.cuda()
    return policy(qpos_data, qvel_data, image_data, action_data, is_pad, flow_data=flow_data)


def train_bc(train_dataloader, val_dataloader, config):
    num_steps = config['num_steps']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    eval_every = config['eval_every']
    validate_every = config['validate_every']
    save_every = config['save_every']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    if config['load_pretrain']:
        pretrain_path = config.get('pretrain_ckpt_path') or os.path.join(ckpt_dir, 'pretrain_all', 'policy_step_50000_seed_0.ckpt')
        loading_status = policy.deserialize(torch.load(pretrain_path))
        print(f'loaded! {loading_status}')
    if config['resume_ckpt_path'] is not None:
        loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
        print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')

    # 滚动断点: 崩溃后自动从最近一步续训 (含 optimizer 状态), 由 --resume_save_every 启用
    resume_save_every = int(config.get('resume_save_every', 0) or 0)
    resume_path = os.path.join(ckpt_dir, 'policy_resume.ckpt')
    resume_blob = None
    if resume_save_every > 0 and config['resume_ckpt_path'] is None and os.path.isfile(resume_path):
        try:
            resume_blob = torch.load(resume_path, map_location='cpu')
            if resume_blob.get('task_name', config['task_name']) != config['task_name']:
                print(f"[auto-resume] task mismatch (ckpt: {resume_blob.get('task_name')}, current: {config['task_name']}), training from scratch")
                resume_blob = None
            else:
                print(f'[auto-resume] found {resume_path} @ step {resume_blob["step"]}')
        except Exception as e:
            print(f'[auto-resume] unreadable ({e}), training from scratch')
            resume_blob = None
    if resume_blob is not None:
        loading_status = policy.deserialize(resume_blob['policy'])
        print(f'[auto-resume] policy loaded, status: {loading_status}')
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)
    start_step = 0
    if resume_blob is not None:
        optimizer.load_state_dict(resume_blob['optimizer'])
        start_step = int(resume_blob['step']) + 1
        print(f'[auto-resume] optimizer state restored, resuming from step {start_step}')
        del resume_blob

    min_val_loss = np.inf
    best_ckpt_info = None

    train_history = []
    validation_history = []
    
    train_dataloader = repeater(train_dataloader)
    for step in tqdm(range(start_step, num_steps+1), initial=start_step, total=num_steps+1):
        # validation
        if step % validate_every == 0:
            print('validating')

            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    # print('111111111111111')
                    forward_dict = forward_pass(data, policy)
                    # print('22222222222222222')
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break

                validation_summary = compute_dict_mean(validation_dicts)
                validation_history.append(validation_summary)

                # print(f'Validation summary: {validation_summary}')

                epoch_val_loss = validation_summary['loss']
                if isinstance(epoch_val_loss, torch.Tensor):
                    epoch_val_loss = epoch_val_loss.item()
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)            
            # wandb.log(validation_summary, step=step)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                v = v.item() if isinstance(v, torch.Tensor) else v
                summary_string += f'{k}: {v:.3f} '
            print(summary_string)
                
        # evaluation
        if (step > 0) and (step % eval_every == 0):
            # first save then eval
            ckpt_name = f'policy_step_{step}_seed_{seed}.ckpt'
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save(policy.serialize(), ckpt_path)
            # success, _ = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10)  # 真机训练需要屏蔽
            # wandb.log({'success': success}, step=step)

        # training
        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        forward_dict = forward_pass(data, policy)
        # backward
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        train_history.append(detach_dict(forward_dict))
        # wandb.log(forward_dict, step=step) # not great, make training 1-2% slower

        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)

        if resume_save_every > 0 and step % resume_save_every == 0:
            # 原子写入滚动断点, 防止崩溃时写坏; 带 task_name 防止跨任务误续训
            tmp_path = resume_path + '.tmp'
            torch.save({'policy': policy.serialize(), 'optimizer': optimizer.state_dict(), 'step': step, 'task_name': config['task_name']}, tmp_path)
            os.replace(tmp_path, resume_path)
            
        if step % 500 == 0:
            plot_history(train_history, validation_history, step, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)
    if resume_save_every > 0 and os.path.isfile(resume_path):
        os.remove(resume_path)  # 训练完成, 删除滚动断点

    if best_ckpt_info is None:
        print('Training finished without any validation point; returning last-step info')
        best_ckpt_info = (num_steps, float('nan'), policy.serialize())

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_step_{best_step}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}')

    return best_ckpt_info

def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves

    key_mapping = {
        "loss": "val_loss",  # train的loss对应val的val_loss
        "acc": "val_acc"     # 若有其他键，同理补充
    }
    for train_key  in train_history[0]:
        val_key = key_mapping.get(train_key, train_key)  # 找不到映射则用原键
        # 跳过val中无对应键的情况
        if val_key not in validation_history[0]:
            # print(f"警告：validation_history 无 {val_key} 键，跳过绘图")
            continue

        plot_path = os.path.join(ckpt_dir, f'train_val_{train_key}_seed_{seed}.png')
        plt.figure()
        
        # 提取train值（train用train_key）
        train_values = [summary[train_key].cpu().item() for summary in train_history]
        # 提取val值（val用val_key）
        val_values = [summary[val_key].cpu().item() for summary in validation_history]
        
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(train_key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')





def visualize_latent_dynamics(config, ckpt_name):
    """
    专门用于分析“8字形”轨迹的 Latent Space 动力学特性
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    # 1. 加载模型和环境
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    
    # 加载 Checkpoint
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(f'Loaded: {ckpt_path} ({loading_status})')
    policy.cuda()
    policy.eval()

    # 加载统计数据用于归一化
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']

    # 初始化环境
    if real_robot:
        from aloha_scripts.real_env import make_real_env
        env = make_real_env(init_node=True, setup_robots=True, setup_base=True)
    else:
        from sim_env import make_sim_env
        # 强制设置 BOX_POSE 为 Figure 8 的起始姿态，保证轨迹标准
        # BOX_POSE[0] = np.concatenate(sample_figure8_pose()) # used in sim reset
        env = make_sim_env(config['task_name'])
        # 与录制端约定一致: box_left <- socket(-0.2), box_right <- peg(+0.2)
        box_right_pose, box_left_pose = sample_figure8_pose()
        BOX_POSE[0] = np.concatenate([box_left_pose, box_right_pose])

    # 2. 运行推理并收集 Latent
    ts = env.reset()
    max_timesteps = config['episode_len']
    
    latent_vectors = []  # 存储 mu_prior (来自视觉先验)
    trajectory_points = [] # 存储实际 qpos (用于画 3D 轨迹参考)
    velocities = []      # 存储实际 qvel (用于颜色编码方向)
    timesteps = []       # 存储时间步 t
    image_list = [] 

    query_frequency = policy_config['num_queries']
    prev_image = None
    
    print("正在收集 Latent 数据...")
    with torch.inference_mode():
        for t in range(max_timesteps):
            # 获取观测
            obs = ts.observation
            qpos_numpy = np.array(obs['qpos'])
            qpos = pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

            if 'images' in obs:
                image_list.append(obs['images'])
            else:
                image_list.append({'main': obs['image']})
            
            if t % query_frequency == 0:
                    curr_image = get_image(ts, camera_names, rand_crop_resize=(config['policy_class'] == 'Diffusion' or config['policy_class'] == 'FlowMatching'))
                    if prev_image is None:  # 第一次图像，复制当前帧
                        prev_image = curr_image.clone()   
                    
                    image = torch.cat([prev_image, curr_image], dim=2)  # 在通道维度上拼接
                    # np.concatenate([img_prev, img_curr], axis=2)
                    prev_image = curr_image.clone()  # 更新为当前帧
                    # print('image shape: ', curr_image.shape)
                # print('get image: ', time.time() - time2)

            if t == 0:
                    # warm up
                    for _ in range(10):
                        # print('aaaaa', image.shape)
                        policy(qpos=None, qvel=None, image=image)
                    print('network warm up done')
                    time1 = time.time()
            
            # --- 核心修改：直接调用 policy.model 获取内部 Latent ---
            # 我们不只想要 action，还想要 visual_prior_head 输出的 mu_prior
            # 注意：这里假设 policy 是 ACTPolicy，且包含 self.model (DETRVAE)
            
            # 构造输入
            # env_state = None # 你的代码没用到这个
            
            # 调用 forward (模拟 inference 过程)
            # DETRVAE forward 返回: a_hat, is_pad_hat, [mu, logvar], probs, binaries, [mu_prior, logvar_prior], ...

            stats_mean = [0.485, 0.456, 0.406]
            stats_std = [0.229, 0.224, 0.225]

            normalize = transforms.Normalize(mean=stats_mean + stats_mean,  # [r,g,b, r,g,b]
                            std=stats_std + stats_std)     # [r,g,b, r,g,b]

            image = normalize(image)

            model_output = policy.model(qpos=None, image=image, env_state=None)
            
            # 提取 mu_prior (这是纯视觉预测的 Latent，最能证明你学到了动力学)
            # 注意检查你的 detr_vae.py 返回值的顺序，根据你提供的文件：
            # return a_hat, is_pad_hat, [mu, logvar], probs, binaries, [mu_prior, logvar_prior], pred_qpos, pred_qvel
            mu_prior = model_output[5][0] # Index 5 is [mu_prior, logvar_prior], 0 is mu
            
            latent_vectors.append(mu_prior.cpu().numpy().flatten())
            trajectory_points.append(qpos_numpy) # 记录真实位置
            timesteps.append(t)
            
            # 执行 Action (为了让机器人动起来)
            all_actions = model_output[0] # a_hat
            raw_action = all_actions[:, t % query_frequency] 


            raw_action = raw_action.squeeze(0).cpu().numpy()
            post_process = lambda a: a * stats['action_std'] + stats['action_mean']
            action = post_process(raw_action)
            target_qpos = action[:-2]
            ts = env.step(target_qpos)

            # action = raw_action.squeeze(0).cpu().numpy() * stats['action_std'] + stats['action_mean']
            # target_qpos = action[:-2]
            # ts = env.step(target_qpos)
            
            # 计算简单的速度方向 (当前位置 - 上一位置)
            if t > 0:
                vel = trajectory_points[-1] - trajectory_points[-2]
                velocities.append(vel)
            else:
                velocities.append(np.zeros_like(qpos_numpy))

    DT = 1 / FPS

    save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video_analysis.mp4'))

    # 3. t-SNE 分析
    print("正在计算 t-SNE...")
    latent_vectors = np.array(latent_vectors)
    trajectory_points = np.array(trajectory_points)
    
    # 使用 perplexity=30 通常对轨迹数据效果较好
    tsne = TSNE(n_components=2, perplexity=100, random_state=42, init='pca', learning_rate='auto')
    z_embedded = tsne.fit_transform(latent_vectors)

    # 4. 可视化绘图
    fig = plt.figure(figsize=(16, 6))

    # 子图1：物理空间轨迹 (Ground Truth)
    ax1 = fig.add_subplot(121)
    # 假设前两个关节决定了平面位置 (根据具体机器人调整，比如 x, y)
    # 对于 Aloha/Viper 这种臂，通常用末端坐标更好，这里简单用 qpos[0] vs qpos[1] 展示
    # 如果是 Wipe 任务，建议画末端 (End Effector) 轨迹
    sc1 = ax1.scatter(trajectory_points[:, 0], trajectory_points[:, 1], c=timesteps, cmap='viridis', s=10)
    ax1.set_title('Robot Trajectory (Physical Space)')
    ax1.set_xlabel('Joint 0 / X')
    ax1.set_ylabel('Joint 1 / Y')
    plt.colorbar(sc1, ax=ax1, label='Time Step')
    
    # 标记交叉点 (示意)
    center_idx = len(timesteps) // 2
    ax1.scatter(trajectory_points[center_idx, 0], trajectory_points[center_idx, 1], c='red', s=100, marker='x', label='Crossing')

    # 子图2：Latent Space (t-SNE)
    ax2 = fig.add_subplot(122)
    sc2 = ax2.scatter(z_embedded[:, 0], z_embedded[:, 1], c=timesteps, cmap='viridis', s=15, alpha=0.8)
    
    # 关键：连接点以显示轨迹连续性
    ax2.plot(z_embedded[:, 0], z_embedded[:, 1], c='gray', alpha=0.3, linewidth=0.5)
    
    ax2.set_title('Latent Space Visualization (t-SNE)')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    
    # 添加文字说明
    text_str = "Observation: Similar Images at Crossing\nLatent Space: Distinct Clusters based on Direction"
    ax2.text(0.05, 0.95, text_str, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.colorbar(sc2, ax=ax2, label='Time Step')
    plt.suptitle(f'Figure 8 Task: Disentangling Dynamics from Vision\n({ckpt_name})', fontsize=16)
    
    save_path = os.path.join(ckpt_dir, 'latent_tsne_analysis.png')
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存至: {save_path}")
    plt.show()


def visualize_latent_dynamics_v2(config, ckpt_name):
    """
    改进版：使用速度方向（Phase）进行颜色编码，更清晰地展示动力学分离
    """
    # ... (前面的加载模型、环境代码保持不变，请复制之前的 setup 部分) ...
    # 假设 env, policy, stats 已经加载完毕

    # 1. 加载模型和环境
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    
    # 加载 Checkpoint
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(f'Loaded: {ckpt_path} ({loading_status})')
    policy.cuda()
    policy.eval()

    # 加载统计数据用于归一化
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']

    # 临时覆盖：强制使用 sim_env 并固定初始位置
    from sim_env import make_sim_env, BOX_POSE
    from utils import sample_figure8_pose 
    # 确保 BOX_POSE 与录制端约定一致: box_left <- socket(-0.2), box_right <- peg(+0.2)
    box_right_pose, box_left_pose = sample_figure8_pose()
    BOX_POSE[0] = np.concatenate([box_left_pose, box_right_pose])
    env = make_sim_env(config['task_name'])

    ts = env.reset()
    max_timesteps = config['episode_len']
    
    latent_vectors = []
    trajectory_points = [] 
    velocities = []      
    
    # 存储图像用于 debug (可选)
    # debug_images = []

    prev_image = None
    
    print("Collecting Trajectory Data...")
    with torch.inference_mode():
        for t in range(max_timesteps):
            # 1. 获取观测
            obs = ts.observation
            qpos_numpy = np.array(obs['qpos'])
            
            # TODO: 确认这里是画 End Effector (x,y) 还是 Joint (q0, q1)
            # 如果是仿真环境，obs['qpos'] 通常是关节角。
            # 为了左图好看，我们暂时存前两个关节，但建议你改为存 EE 坐标
            trajectory_points.append(qpos_numpy[:2]) 

            # 归一化 qpos 用于输入模型
            qpos = (qpos_numpy - stats['qpos_mean']) / stats['qpos_std']
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            
            # 2. 处理图像
            curr_image = get_image(ts, camera_names) # 确保这里返回的是 Float Tensor 且归一化过的
            if prev_image is None:
                prev_image = curr_image.clone()
            
            image = torch.cat([prev_image, curr_image], dim=2)
            prev_image = curr_image.clone()

            # 3. 模型推理
            # 注意：传入空的 env_state
            model_output = policy.model(qpos, image, None)
            
            # 4. 获取 Mu Prior (Visual Dynamics)
            # [mu_prior, logvar_prior] is at index 5
            mu_prior = model_output[5][0] 
            latent_vectors.append(mu_prior.cpu().numpy().flatten())
            
            # 5. Step Environment
            all_actions = model_output[0]
            raw_action = all_actions[:, t % config['policy_config']['num_queries']]
            action = raw_action.squeeze(0).cpu().numpy() * stats['action_std'] + stats['action_mean']
            target_qpos = action[:-2]
            ts = env.step(target_qpos)

    # --- 数据处理 ---
    latent_vectors = np.array(latent_vectors)
    trajectory_points = np.array(trajectory_points) # (T, 2)
    
    # 计算速度向量 (v = p_t - p_{t-1})
    vels = np.diff(trajectory_points, axis=0, prepend=trajectory_points[0:1])
    
    # 计算“速度方向角” (-pi 到 pi)
    # 这对于 8 字形特别重要，交叉点处位置相同，但角度相差 90度或180度
    angles = np.arctan2(vels[:, 1], vels[:, 0])

    # 过滤掉速度极小的点（起步和停止），避免噪音干扰 t-SNE
    speed = norm(vels, axis=1)
    mask = speed > 0.001 # 设定一个阈值
    
    latent_clean = latent_vectors[mask]
    traj_clean = trajectory_points[mask]
    angles_clean = angles[mask]

    print(f"Running t-SNE on {len(latent_clean)} points...")
    
    # Perplexity 调参技巧：通常设为数据点数量的 5% - 20%
    # 如果数据点少于 100，perplexity 设小一点（比如 5-10）
    perplex = min(30, len(latent_clean) // 10)
    tsne = TSNE(n_components=2, perplexity=perplex, random_state=42, init='pca')
    z_embedded = tsne.fit_transform(latent_clean)

    # --- 绘图 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 颜色映射使用 hsv (因为角度是循环的)
    cmap = 'hsv'

    # Plot 1: 物理轨迹 (Colored by Direction)
    sc1 = ax1.scatter(traj_clean[:, 0], traj_clean[:, 1], c=angles_clean, cmap=cmap, s=20)
    ax1.set_title('Physical Trajectory\n(Color = Movement Direction)')
    ax1.set_xlabel('X / Joint 0')
    ax1.set_ylabel('Y / Joint 1')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=ax1, label='Direction Angle (rad)')

    # Plot 2: Latent Space (Colored by Direction)
    sc2 = ax2.scatter(z_embedded[:, 0], z_embedded[:, 1], c=angles_clean, cmap=cmap, s=20, alpha=0.8)
    ax2.set_title('Latent Space (Learned Visual Prior)\n(Separation by Direction = Implicit Dynamics)')
    ax2.set_xlabel('t-SNE Dim 1')
    ax2.set_ylabel('t-SNE Dim 2')
    ax2.grid(True, alpha=0.3)
    
    # 标注说明
    plt.suptitle(f"Verification of Implicit Dynamics in '{config['task_name']}'", fontsize=16)
    
    save_path = os.path.join(config['ckpt_dir'], 'latent_dynamics_v2.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved visualization to {save_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_save_every', action='store', type=int, default=0, help='rolling resume checkpoint interval (0=off); auto-resumes from policy_resume.ckpt on restart', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)

    parser.add_argument('--proprio_weight', action='store', type=float, default=10.0,
                        help='Weight for implicit proprioception loss')
    parser.add_argument('--qvel_weight', action='store', type=float, default=0.5,
                        help='Weight of the qvel term inside the aux proprio loss (lambda_2; lambda_1 fixed to 1.0)')
    parser.add_argument('--visual_history_len', action='store', type=int, default=2,
                        help='Number of past frames T fed to the dual-stream dynamics module (default 2 = paper setting)')
    parser.add_argument('--photo_aug', action='store_true',
                        help='Enable photometric augmentation (color jitter / gamma / exposure) for vision-only training')
    parser.add_argument('--frame_drop_prob', action='store', type=float, default=0.0,
                        help='Per-slot probability of simulating a dropped camera frame (frame hold) during training')
    # eval-time visual OOD stress tests (M1)
    parser.add_argument('--eval_photo_perturb', action='store', type=str, default='none',
                        choices=['none', 'mild', 'severe'],
                        help='Image-space photometric OOD perturbation at eval (sim lighting proxy)')
    parser.add_argument('--eval_frame_drop', action='store', type=float, default=0.0,
                        help='Per-step probability of dropping (holding) camera frames at eval')
    # Scripted proprioceptive corruptions for the OOD robustness study (qpos channel of baselines)
    parser.add_argument('--eval_qpos_noise_std', action='store', type=float, default=0.0,
                        help='Std of i.i.d. Gaussian noise added to the qpos observation at eval (0 = off)')
    parser.add_argument('--eval_qpos_drift', action='store', type=float, default=0.0,
                        help='Per-step linear drift added to the qpos observation at eval (0 = off)')
    parser.add_argument('--eval_qpos_dropout', action='store', type=float, default=0.0,
                        help='Per-step probability of zeroing each qpos dimension at eval (0 = off)')
    parser.add_argument('--eval_frame_stride', action='store', type=int, default=1,
                        help='Add a frame to the eval history buffer every N env steps (1 = matches training adjacency; num_queries = legacy query-time pairing)')
    parser.add_argument('--use_aux_proprio', dest='use_aux_proprio', action='store_true',
                        help='Supervise qpos/qvel auxiliary heads (default: on)')
    parser.add_argument('--no_aux_proprio', dest='use_aux_proprio', action='store_false',
                        help='Disable aux proprio loss (ACT w/o qpos + RAFT baseline)')
    parser.set_defaults(use_aux_proprio=True)

    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')

    # V-IP / RAFT baseline experiment switches (see README.md, "Baselines")
    parser.add_argument('--dynamics_mode', type=str, default='implicit',
                        choices=['implicit', 'raft', 'concat_only'],
                        help='implicit=Time-Mixing MLP; raft=explicit RAFT flow; concat_only=multi-frame concat ablation')
    parser.add_argument('--use_learned_prior', dest='use_learned_prior', action='store_true',
                        help='Use visual learned prior in CVAE')
    parser.add_argument('--no_learned_prior', dest='use_learned_prior', action='store_false',
                        help='Disable learned prior (ACT w/o qpos + flow style baseline)')
    parser.set_defaults(use_learned_prior=True)
    parser.add_argument('--flow_h', type=int, default=240, help='RAFT flow height')
    parser.add_argument('--flow_w', type=int, default=320, help='RAFT flow width')
    parser.add_argument('--raft_iters', type=int, default=20, help='RAFT refinement iterations (official demo uses 20)')
    parser.add_argument('--raft_backend', type=str, default='torchvision', choices=['torchvision', 'official'])
    parser.add_argument('--raft_checkpoint', type=str, default=None, help='Path to official RAFT .pth (optional)')
    parser.add_argument('--raft_large', action='store_true', help='Use RAFT-large instead of RAFT-small')
    
    main(vars(parser.parse_args()))
