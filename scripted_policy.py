import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env

import IPython
e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])



class Figure8Policy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        # 从 env_state 获取物体位置
        obj_info = np.array(ts_first.observation['env_state'])
        box_left_xyz = obj_info[:3]
        box_right_xyz = obj_info[7:10]

        # 准备抓取姿态 (垂直向下)
        gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])
        # 稍微调整角度以适应方块
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)
        
        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        # 提起后的中心位置 (增加一点高度，防止大幅度动作撞击桌面)
        lift_height = 0.20 # 原来是0.15，稍微提高一点以适应更大的动作
        center_left = box_left_xyz + np.array([0, 0, lift_height])
        center_right = box_right_xyz + np.array([0, 0, lift_height])

        # 初始化轨迹
        self.left_trajectory = [{"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}]
        self.right_trajectory = [{"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}]

        # --- 第一阶段：抓取 (Approach -> Down -> Grasp -> Lift) ---
        t_approach = 50
        t_grasp_down = 80
        t_grasp_close = 120
        t_lift = 150 #稍微给多一点时间提起来
        
        # 左手动作
        self.left_trajectory.append({"t": t_approach, "xyz": box_left_xyz + np.array([0, 0, 0.12]), "quat": gripper_pick_quat_left.elements, "gripper": 1})
        self.left_trajectory.append({"t": t_grasp_down, "xyz": box_left_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_left.elements, "gripper": 1})
        self.left_trajectory.append({"t": t_grasp_close, "xyz": box_left_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_left.elements, "gripper": 0})
        self.left_trajectory.append({"t": t_lift, "xyz": center_left, "quat": gripper_pick_quat_left.elements, "gripper": 0})

        # 右手动作
        self.right_trajectory.append({"t": t_approach, "xyz": box_right_xyz + np.array([0, 0, 0.12]), "quat": gripper_pick_quat_right.elements, "gripper": 1})
        self.right_trajectory.append({"t": t_grasp_down, "xyz": box_right_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_right.elements, "gripper": 1})
        self.right_trajectory.append({"t": t_grasp_close, "xyz": box_right_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_right.elements, "gripper": 0})
        self.right_trajectory.append({"t": t_lift, "xyz": center_right, "quat": gripper_pick_quat_right.elements, "gripper": 0})

        # --- 第二阶段：画“8”字 (优化版) ---
        # 1. 旋转坐标系：X轴大幅度(前后)，Y轴小幅度(左右)
        # 2. 更加圆润：调整宽高比，增加周期步数
        # 3. 速度平滑：添加 ramp_factor 缓冲启动
        
        period = 300      # 增加周期 (原200)，降低角速度，使轨迹更平滑
        start_t = t_lift
        duration = 800    # 增加持续时间，画满2个周期
        
        # 动作幅度设定 (单位: 米)
        # 机械臂 X 是前后，Y 是左右
        # 为了画纵向的 8 字，X 轴震荡频率应该是 Y 轴的 2 倍
        amp_x = 0.07      # 前后幅度加大 (原0.06)
        amp_y = 0.14      # 左右幅度调整 (原0.10)，2:1 比例通常比较圆润
        
        ramp_steps = 20   # 缓冲步数，用于平滑启动

        for i in range(duration):
            t_curr = start_t + i + 1
            
            # 计算角度
            angle = (i / period) * 2 * np.pi
            
            # 计算缓冲系数 (0.0 -> 1.0)
            # 这会让动作从中心点慢慢扩大，而不是一开始就以最大速度切入
            if i < ramp_steps:
                ramp_factor = (i / ramp_steps)
                # 使用平滑曲线 (Smoothstep) 进一步优化启动手感
                ramp_factor = ramp_factor * ramp_factor * (3 - 2 * ramp_factor)
            else:
                ramp_factor = 1.0

            # 轨迹方程 (Lissajous curve 变体)
            # sin(2*angle) 在 X 轴 (前后) 制造两个波峰波谷
            # sin(angle)   在 Y 轴 (左右) 制造一个波峰波谷
            dx = (amp_x * ramp_factor) * np.sin(2 * angle)
            dy = (amp_y * ramp_factor) * np.sin(angle)

            self.left_trajectory.append({
                "t": t_curr,
                "xyz": center_left + np.array([dx, dy, 0]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 0
            })
            
            self.right_trajectory.append({
                "t": t_curr,
                "xyz": center_right + np.array([dx, dy, 0]), 
                "quat": gripper_pick_quat_right.elements,
                "gripper": 0
            })

class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0, 0.5, 0.25])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        ]


# class InsertionPolicy(BasePolicy):  # old official version

#     def generate_trajectory(self, ts_first):
#         init_mocap_pose_right = ts_first.observation['mocap_pose_right']
#         init_mocap_pose_left = ts_first.observation['mocap_pose_left']

#         peg_info = np.array(ts_first.observation['env_state'])[:7]
#         peg_xyz = peg_info[:3]
#         peg_quat = peg_info[3:]

#         socket_info = np.array(ts_first.observation['env_state'])[7:]
#         socket_xyz = socket_info[:3]
#         socket_quat = socket_info[3:]

#         gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
#         gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

#         gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
#         gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

#         meet_xyz = np.array([0, 0.5, 0.15])
#         lift_right = 0.00715

#         self.left_trajectory = [
#             {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
#             {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
#             {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
#             {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
#             {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach meet position
#             {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # insertion
#             {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # insertion
#         ]

#         self.right_trajectory = [
#             {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
#             {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
#             {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
#             {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
#             {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
#             {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion
#             {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion

#         ]




class InsertionPolicy(BasePolicy):   # 20251227 增大整体速度，分为快慢两部分，在插入之前刹车，体现算法效果
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:]) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)
        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:]) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        # --- 时间轴规划 (Total: 400 steps) ---
        # 1. 抓取三步曲 (修复点)
        t_approach = 70      # 移动到物体上方
        t_down = 100         # 垂直下放到抓取高度 (保持张开) <-- 之前少了这一步
        t_grasp = 130        # 闭合夹爪
        
        # 2. 蓄力拉开 (Pull Back)
        t_pull_back = 180    # 抬起并拉远
        
        # 3. 极速冲刺 (Sprint)
        t_sprint_end = 220   # 60步冲刺
        
        # 4. 晚刹车插入 (Insert)
        t_end = 400

        # --- 空间参数 ---
        offset_start = 0.35   # 蓄力起点 (远)
        offset_brake = 0.08   # 刹车点 (近 - 晚刹车)
        offset_finish = 0.04  # 插入终点

        # --- 左手轨迹 (Socket) ---
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            
            # Step 1: Approach (High & Open)
            {"t": t_approach, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
            
            # Step 2: Down (Low & Open) - 关键修复：先下去，再闭合
            {"t": t_down, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1},
            
            # Step 3: Grasp (Low & Closed)
            {"t": t_grasp, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            
            # Phase 1: Pull Back (蓄力拉远)
            {"t": t_pull_back, "xyz": meet_xyz + np.array([-offset_start, 0, 0.1]), "quat": gripper_pick_quat_left.elements, "gripper": 0},

            # Phase 2: Sprint (极速冲刺)
            {"t": t_sprint_end, "xyz": meet_xyz + np.array([-offset_brake, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            
            # Phase 3: Late Brake & Insert (晚刹车插入)
            {"t": t_end, "xyz": meet_xyz + np.array([-offset_finish, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},
        ]

        # --- 右手轨迹 (Peg) ---
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
            
            # Step 1: Approach
            {"t": t_approach, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1},
            
            # Step 2: Down (修复：保持张开 1)
            {"t": t_down, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1},
            
            # Step 3: Grasp (闭合 0)
            {"t": t_grasp, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0},

            # Phase 1: Pull Back
            {"t": t_pull_back, "xyz": meet_xyz + np.array([offset_start, 0, 0.1 + lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            
            # Phase 2: Sprint
            {"t": t_sprint_end, "xyz": meet_xyz + np.array([offset_brake, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            
            # Phase 3: Late Brake & Insert
            {"t": t_end, "xyz": meet_xyz + np.array([offset_finish, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
        ]


def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['angle'])
            plt.ion()

        policy = PickAndTransferPolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'sim_transfer_cube_scripted'
    test_policy(test_task_name)

