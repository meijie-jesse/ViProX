import os
# ---------------------------------------------------------
# 强制使用 EGL 离屏渲染，防止多进程 X11 冲突
os.environ['MUJOCO_GL'] = 'egl' 
# ---------------------------------------------------------

import time
import numpy as np
import argparse
import h5py
import multiprocessing

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy, Figure8Policy

import IPython
e = IPython.embed

def generate_episode(inputs):
    """
    Worker 函数现在接收一个元组 inputs = (episode_idx, random_seed, args)
    """
    episode_idx, random_seed, args = inputs
    
    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    min_reward = args['min_reward']
    
    # 强制关闭渲染
    onscreen_render = False 
    
    # 【关键】使用传入的特定 seed，而不是只用 episode_idx
    # 这样重试时可以使用不同的随机性
    np.random.seed(random_seed)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']

    # 策略选择
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    elif task_name == 'sim_transfer_cube_scripted_mirror':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_figure8_scripted':
        policy_cls = Figure8Policy
    else:
        raise NotImplementedError

    # print(f'Worker processing Episode {episode_idx} (seed: {random_seed})...')

    # ---------------------------------------------------------
    # 阶段 1: EE Space Policy Rollout
    # ---------------------------------------------------------
    try:
        env = make_ee_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(inject_noise=False)

        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
        
        # --- 校验阶段 1 ---
        ee_episode_return = np.sum([ts.reward for ts in episode[1:]])
        ee_max_reward = np.max([ts.reward for ts in episode[1:]])
        
        if (ee_max_reward != env.task.max_reward) or (ee_episode_return < min_reward):
            # print(f"Episode {episode_idx}: Failed at EE Stage. Return: {ee_episode_return}")
            del env, episode, policy
            return episode_idx, False # 返回 idx 和 失败状态

        # 准备关节数据
        joint_traj = [ts.observation['qpos'] for ts in episode]
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
            joint[6] = left_ctrl
            joint[6+7] = right_ctrl

        subtask_info = episode[0].observation['env_state'].copy()
        del env, episode, policy

        # ---------------------------------------------------------
        # 阶段 2: Sim Env Replay
        # ---------------------------------------------------------
        env = make_sim_env(task_name)
        BOX_POSE[0] = subtask_info 
        ts = env.reset()
        episode_replay = [ts]

        for t in range(len(joint_traj)):
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)

        # --- 校验阶段 2 ---
        replay_return = np.sum([ts.reward for ts in episode_replay[1:]])
        replay_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        
        if (replay_max_reward != env.task.max_reward) or (replay_return < min_reward):
            # print(f"Episode {episode_idx}: Failed at Replay Stage. Return: {replay_return}")
            del env, episode_replay
            return episode_idx, False

        # ---------------------------------------------------------
        # 保存数据
        # ---------------------------------------------------------
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]
        max_timesteps = len(joint_traj)

        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                        chunks=(1, 480, 640, 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        
        print(f"Episode {episode_idx}: SUCCESS (Return: {replay_return})")
        return episode_idx, True

    except Exception as e:
        print(f"Episode {episode_idx}: Exception occurred - {e}")
        return episode_idx, False


def main(args):
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass 

    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    num_workers = args.get('num_workers', 4)
    args['onscreen_render'] = False 

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    # 初始化：需要生成的 episode 索引列表 (0 到 49)
    pending_indices = list(range(num_episodes))
    
    print(f"Target: Generate {num_episodes} successful episodes.")
    print(f"Workers: {num_workers}")
    print(f"Min Reward: {args['min_reward']}")
    
    total_start_time = time.time()
    
    # 只要还有未完成的索引，就继续循环
    while len(pending_indices) > 0:
        print(f"\n--- Starting batch for {len(pending_indices)} episodes ---")
        
        # 准备任务参数
        # 关键：为每个任务生成一个新的随机种子
        # 使用 time.time() 确保每次重试时的随机性都不同
        current_batch_args = []
        for idx in pending_indices:
            # 随机种子 = 基础种子 + 索引 + 当前时间戳微秒，保证绝对随机
            seed = int(time.time() * 1000) % 10000000 + idx 
            current_batch_args.append((idx, seed, args))
        
        # 运行多进程
        if num_workers > 1:
            with multiprocessing.Pool(num_workers) as pool:
                results = pool.map(generate_episode, current_batch_args)
        else:
            results = []
            for inputs in current_batch_args:
                results.append(generate_episode(inputs))
        
        # 检查结果
        new_pending_indices = []
        success_count_batch = 0
        
        for idx, is_success in results:
            if is_success:
                success_count_batch += 1
            else:
                # 如果失败了，加入下一轮的待处理列表
                new_pending_indices.append(idx)
        
        # 更新待处理列表
        pending_indices = new_pending_indices
        
        if len(pending_indices) > 0:
            print(f"Batch finished. {success_count_batch} succeeded. {len(pending_indices)} failed/low-score. Retrying failures...")
        else:
            print(f"Batch finished. All remaining episodes succeeded!")

    total_time = time.time() - total_start_time
    print(f'\nAll {num_episodes} episodes generated successfully!')
    print(f'Saved to {dataset_dir}')
    print(f'Total time: {total_time:.1f} secs')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--num_workers', action='store', type=int, default=4)
    parser.add_argument('--min_reward', action='store', type=int, default=600, help='Minimum reward to save the episode')
    
    main(vars(parser.parse_args()))