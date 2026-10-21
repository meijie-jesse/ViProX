import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

import IPython
e = IPython.embed

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def sample_photometric_params():
    """Sample one set of photometric parameters (shared across frames/cameras of a sample,
    mimicking a global lighting / camera-exposure change)."""
    return {
        'brightness': np.random.uniform(0.70, 1.30),
        'contrast': np.random.uniform(0.75, 1.25),
        'saturation': np.random.uniform(0.60, 1.40),
        'gamma': np.random.uniform(0.60, 1.60),
    }

def apply_photometric_aug(stacked_img, n_frames, params):
    """Apply identical photometric distortion to every 3-channel frame of a
    channel-stacked uint8 image (H, W, 3*n_frames). Returns uint8."""
    imgs = np.split(stacked_img, n_frames, axis=2)
    out = []
    for img in imgs:
        x = img.astype(np.float32)
        # brightness
        x = x * params['brightness']
        # contrast (PIL-style, around per-image mean intensity)
        mean = x.mean(axis=(0, 1), keepdims=True)
        x = (x - mean) * params['contrast'] + mean
        # saturation: blend with per-pixel luminance
        gray = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
        x = gray + (x - gray) * params['saturation']
        # gamma / exposure shift
        x = np.clip(x, 0.0, 255.0) / 255.0
        x = np.power(x, params['gamma']) * 255.0
        out.append(np.clip(x, 0.0, 255.0).astype(np.uint8))
    return np.concatenate(out, axis=2)

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, policy_class, dynamics_mode='implicit', flow_image_size=(240, 320), visual_history_len=2, photo_aug=False, frame_drop_prob=0.0):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        self.dynamics_mode = dynamics_mode
        self.flow_image_size = flow_image_size
        self.visual_history_len = max(1, int(visual_history_len))
        self.photo_aug = photo_aug
        self.frame_drop_prob = float(frame_drop_prob)
        if self.policy_class == 'Diffusion':
            self.augment_images = True
        else:
            self.augment_images = False
        self.transformations = None
        self.__getitem__(0) # initialize self.is_sim and self.transformations
        self.is_sim = False

    # def __len__(self):
    #     return sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index, _io_retry=0):
        sample_full_episode = False # hardcode

        # episode_id = index // self.max_episode_len
        # start_ts = index % self.max_episode_len # 当前时间步
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]

        # 1. 历史帧索引: [t-T+1, ..., t-1, t] (最旧在前; 序列开头用最旧可用帧填充)
        T = self.visual_history_len
        hist_indices = [max(0, start_ts - offset) for offset in range(T - 1, -1, -1)]

        # 随机帧丢弃模拟 (frame hold): 某个历史槽位以概率 p 显示更新的帧,
        # 模拟相机掉帧时 buffer 保持最近一帧的真实行为
        if self.frame_drop_prob > 0 and T > 1:
            for k in range(T - 1):
                if np.random.rand() < self.frame_drop_prob:
                    hist_indices[k] = hist_indices[k + 1]

        # 光度增强参数: 同一样本的所有帧、所有相机共享 (模拟全局光照/曝光变化)
        photo_params = sample_photometric_params() if self.photo_aug else None


        try:
            flow_list = None
            # print(dataset_path)
            with h5py.File(dataset_path, 'r') as root:
                try: # some legacy data does not have this attribute
                    is_sim = root.attrs['sim']
                except:
                    is_sim = False
                compressed = root.attrs.get('compress', False)
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root['/action'][()], base_action], axis=-1)
                else:
                    action = root['/action'][()]
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
                original_action_shape = action.shape
                episode_len = original_action_shape[0]
                # get observation at start_ts only
                qpos = root['/observations/qpos'][start_ts]
                qvel = root['/observations/qvel'][start_ts]
                image_dict = dict()
                for cam_name in self.camera_names:
                    # 读取 T 帧历史 (t-T+1 ... t)
                    frames = [root[f'/observations/images/{cam_name}'][idx] for idx in hist_indices]

                    # [关键步骤] 拼接图像: (H, W, 3) * T -> (H, W, 3T)
                    stacked_img = np.concatenate(frames, axis=2)

                    image_dict[cam_name] = stacked_img

                flow_data = None
                if self.dynamics_mode == 'raft':
                    flow_list = []
                    for cam_name in self.camera_names:
                        flow_path = f'/observations/flow/{cam_name}'
                        if flow_path not in root:
                            flow_list = None
                            break
                        flow_list.append(root[flow_path][start_ts])

                if compressed:
                    for cam_name in image_dict.keys():
                        decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                        image_dict[cam_name] = np.array(decompressed_image)

                # 光度增强: 对 stack 后的图像逐帧应用同一组参数 (uint8 域)
                if photo_params is not None:
                    for cam_name in image_dict.keys():
                        image_dict[cam_name] = apply_photometric_aug(image_dict[cam_name], T, photo_params)
                
                # get all actions after and including start_ts
                if is_sim:
                    action = action[start_ts:]
                    action_len = episode_len - start_ts
                else:
                    action = action[max(0, start_ts - 1):] # hack, to make timesteps more aligned
                    action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

            # self.is_sim = is_sim
            padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
            padded_action[:action_len] = action
            is_pad = np.zeros(self.max_episode_len)
            is_pad[action_len:] = 1

            padded_action = padded_action[:self.chunk_size]
            is_pad = is_pad[:self.chunk_size]

            # new axis for different cameras
            all_cam_images = []
            for cam_name in self.camera_names:
                all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=0)

            # construct observations
            image_data = torch.from_numpy(all_cam_images)
            qpos_data = torch.from_numpy(qpos).float()
            qvel_data = torch.from_numpy(qvel).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()

            # channel last
            image_data = torch.einsum('k h w c -> k c h w', image_data)

            # augmentation
            if self.transformations is None:
                print('Initializing transformations')
                original_size = image_data.shape[2:]
                ratio = 0.95
                self.transformations = [
                    transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                    transforms.Resize(original_size, antialias=True),
                    transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5) #, hue=0.08)
                ]

            if self.augment_images:
                for transform in self.transformations:
                    image_data = transform(image_data)

            # normalize image and change dtype to float
            image_data = image_data / 255.0

            if self.policy_class == 'Diffusion':
                # normalize to [-1, 1]
                action_data = ((action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
            else:
                # normalize to mean 0 std 1
                action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
            qvel_data = (qvel_data - self.norm_stats["qvel_mean"]) / self.norm_stats["qvel_std"]

            if self.dynamics_mode == 'raft' and flow_list is not None:
                flow_data = np.stack(flow_list, axis=0)  # [num_cams, H, W, 2]
                flow_data = torch.from_numpy(flow_data).float()
                flow_data = torch.einsum('k h w c -> k c h w', flow_data)
                flow_data = torch.cat([flow_data[i] for i in range(flow_data.shape[0])], dim=0)
            else:
                flow_data = None

        except Exception as e:
            # 硬件/IO 瞬时抖动导致的 hdf5 读取失败: 原地重试, 不要 quit()(quit 会以退出码 0 静默杀死训练, 绕过外层重试)
            print(f'Error loading {dataset_path} in __getitem__ (io_retry={_io_retry}): {e}')
            if _io_retry < 5:
                import time as _time
                _time.sleep(2)
                return self.__getitem__(index, _io_retry + 1)
            raise

        # print(image_data.dtype, qpos_data.dtype, action_data.dtype, is_pad.dtype)
        return image_data, qpos_data, qvel_data, action_data, is_pad, flow_data


def get_norm_stats(dataset_path_list):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []
    all_qvel_data = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                qvel = root['/observations/qvel'][()]
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root['/action'][()], base_action], axis=-1)
                else:
                    action = root['/action'][()]
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            raise  # 原为 quit(): 退出码 0 会绕过训练脚本的重试循环
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
        all_qvel_data.append(torch.from_numpy(qvel))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)
    all_qvel_data = torch.cat(all_qvel_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    # normalize qvel data
    qvel_mean = all_qvel_data.mean(dim=[0]).float()
    qvel_std = all_qvel_data.std(dim=[0]).float()
    qvel_std = torch.clip(qvel_std, 1e-2, np.inf) # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps,
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "qvel_mean": qvel_mean.numpy(), "qvel_std": qvel_std.numpy(),
             "example_qpos": qpos}

    return stats, all_episode_len

def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def collate_with_optional_flow(batch):
    image_data, qpos_data, qvel_data, action_data, is_pad, flow_data = zip(*batch)
    image_data = torch.stack(image_data, dim=0)
    qpos_data = torch.stack(qpos_data, dim=0)
    qvel_data = torch.stack(qvel_data, dim=0)
    action_data = torch.stack(action_data, dim=0)
    is_pad = torch.stack(is_pad, dim=0)
    if all(item is None for item in flow_data):
        flow_batch = None
    else:
        flow_batch = torch.stack(flow_data, dim=0)
    return image_data, qpos_data, qvel_data, action_data, is_pad, flow_batch


def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size, skip_mirrored_data=False, load_pretrain=False, policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.99, dynamics_mode='implicit', flow_image_size=(240, 320), visual_history_len=2, photo_aug=False, frame_drop_prob=0.0):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # obtain train test split on dataset_dir_l[0]
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')

    # obtain normalization stats for qpos and action
    # if load_pretrain:
    #     with open(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'dataset_stats.pkl'), 'rb') as f:
    #         norm_stats = pickle.load(f)
    #     print('Loaded pretrain dataset stats')
    _, all_episode_len = get_norm_stats(dataset_path_list)
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]))
    print(f'Norm stats from: {stats_dir_l}')

    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

    # print(f'train_episode_len: {train_episode_len}, val_episode_len: {val_episode_len}, train_episode_ids: {train_episode_ids}, val_episode_ids: {val_episode_ids}')

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, policy_class, dynamics_mode=dynamics_mode, flow_image_size=flow_image_size, visual_history_len=visual_history_len, photo_aug=photo_aug, frame_drop_prob=frame_drop_prob)
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size, policy_class, dynamics_mode=dynamics_mode, flow_image_size=flow_image_size, visual_history_len=visual_history_len, photo_aug=False, frame_drop_prob=0.0)
    train_num_workers = int(os.environ.get('ACT_TRAIN_NUM_WORKERS', (8 if os.getlogin() == 'zfu' else 16) if train_dataset.augment_images else 2))
    val_num_workers = int(os.environ.get('ACT_VAL_NUM_WORKERS', 8 if train_dataset.augment_images else 2))
    print(f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')

    def _seed_worker(worker_id):
        # fork 后 numpy RNG 状态被复制, 需按 worker 重新播种, 否则各 worker 增强序列相关
        np.random.seed((torch.initial_seed() + worker_id) % (2 ** 32))

    train_pf = 2 if train_num_workers > 0 else None  # num_workers=0 时 prefetch_factor 必须为 None
    val_pf = 2 if val_num_workers > 0 else None
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=train_num_workers, prefetch_factor=train_pf, collate_fn=collate_with_optional_flow, worker_init_fn=_seed_worker)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=val_num_workers, prefetch_factor=val_pf, collate_fn=collate_with_optional_flow, worker_init_fn=_seed_worker)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0 # 0.19
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action

def smooth_base_action(base_action):
    return np.stack([
        np.convolve(base_action[:, i], np.ones(5)/5, mode='same') for i in range(base_action.shape[1])
    ], axis=-1).astype(np.float32)

def preprocess_base_action(base_action):
    # base_action = calibrate_linear_vel(base_action)
    base_action = smooth_base_action(base_action)

    return base_action

def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0
    # angular_vel = 0
    # if np.abs(linear_vel) < 0.05:
    #     linear_vel = 0
    return np.array([linear_vel, angular_vel])

### env utils


def sample_figure8_pose():

    box_right_pose = np.array([0.2, 0.5, 0.025, 1, 0, 0, 0])
    box_left_pose = np.array([-0.2, 0.5, 0.025, 1, 0, 0, 0]) 


    # Peg
    x_range = [0.18, 0.22]
    y_range = [0.48, 0.52]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.22, -0.18]
    y_range = [0.48, 0.52]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value = epoch_dict[k]
            if isinstance(value, torch.Tensor):
                value = value.detach()
            value_sum += value
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach() if isinstance(v, torch.Tensor) else v
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
