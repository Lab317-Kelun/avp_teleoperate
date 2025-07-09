import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Array, Lock
import threading
import os 
import sys
import math
import lcm
import matplotlib.pyplot as plt

# 设置当前工作目录，方便导入上级目录的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入自己写的模块（机械臂、机械手、图像客户端等）
from teleop.open_television.tv_wrapper import TeleVisionWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller, Gripper_Controller
from teleop.robot_control.robot_hand_inspire import Inspire_Controller
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter

sys.path.append('/home/asus/unitreerobotics/avp_teleoperate/teleop/robot_control')
# lcm 通信包
from arm_vel import arm_vel_lcmt

class VisionProHeadTracker:
    """
    一个用于处理和分析 Vision Pro 4x4 头部姿态矩阵的类。

    该类可以计算头部在 x, y 方向的速度、yaw (偏航) 角速度以及 z 方向的高度。
    """
    def __init__(self):
        self.previous_matrix = None
        self.last_timestamp = None
        
        # 存储计算结果
        self.velocity_x = 0.0  # X 方向速度 (m/s)
        self.velocity_y = 0.0  # Y 方向速度 (m/s)
        self.height_z = 0.0    # Z 方向高度 (m)
        self.yaw_velocity = 0.0 # Yaw 角速度 (rad/s)

    def update_matrix(self, current_matrix: np.ndarray):
        """
        用新的 4x4 矩阵更新追踪器状态并计算相关指标。

        参数:
            current_matrix (np.ndarray): 一个 4x4 的 NumPy 数组，代表当前的头部姿态。
        """
        current_timestamp = time.time()
        
        # Z 方向高度总是可以直接从当前矩阵中获取
        self.height_z = current_matrix[2, 3]

        if self.previous_matrix is not None and self.last_timestamp is not None:
            delta_t = current_timestamp - self.last_timestamp
            
            # 防止除以零
            if delta_t == 0:
                return

            # --- 1. 计算 X, Y 方向速度 ---
            prev_pos = self.previous_matrix[:3, 3]
            current_pos = current_matrix[:3, 3]
            
            delta_pos = current_pos - prev_pos
            
            self.velocity_x = delta_pos[0] / delta_t
            self.velocity_y = delta_pos[1] / delta_t

            # --- 2. 计算 Yaw 速度 ---
            # 从旋转矩阵中提取 Yaw 角
            # Yaw = atan2(R31, R11)
            prev_yaw = math.atan2(self.previous_matrix[2, 0], self.previous_matrix[0, 0])
            current_yaw = math.atan2(current_matrix[2, 0], current_matrix[0, 0])
            
            # 处理角度环绕问题 (e.g., from +pi to -pi)
            delta_yaw = current_yaw - prev_yaw
            if delta_yaw > math.pi:
                delta_yaw -= 2 * math.pi
            if delta_yaw < -math.pi:
                delta_yaw += 2 * math.pi
            
            self.yaw_velocity = delta_yaw / delta_t
        
        # 更新状态以备下次计算
        self.previous_matrix = current_matrix
        self.last_timestamp = current_timestamp

    def get_metrics(self):
        """
        返回一个包含所有计算指标的字典。
        """
        return self.velocity_x, self.velocity_y, self.height_z, self.yaw_velocity, math.degrees(self.yaw_velocity)


class SlidingFilter:
    def __init__(self, window_size=51, group_size=3):
        assert window_size % group_size == 0, "Window size must be divisible by group size"
        self.window_size = window_size
        self.group_size = group_size
        self.data = []

    def add(self, value):
        self.data.append(value)
        if len(self.data) > self.window_size:
            self.data.pop(0)

    def ready(self):
        return len(self.data) == self.window_size

    def get_filtered_value(self):
        if not self.ready():
            return 0.0  # 或者返回 None / 上一次结果

        groups = [self.data[i:i + self.group_size] for i in range(0, self.window_size, self.group_size)]
        group_means = []
        for g in groups:
            if len(g) < 3:
                continue
            trimmed = sorted(g)[1:-1]  # 去掉最大最小值，留下中间值
            group_means.append(np.mean(trimmed))

        return np.mean(group_means) if group_means else 0.0


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type=str, default='./utils/data', help='数据保存路径')
    parser.add_argument('--frequency', type=int, default=30.0, help='数据保存频率')
    parser.add_argument('--record', action='store_true', help='是否保存数据')
    parser.add_argument('--no-record', dest='record', action='store_false', help='不保存数据')
    parser.set_defaults(record=False)
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='选择机械臂型号')
    parser.add_argument('--hand', type=str, choices=['dex3', 'gripper', 'inspire1'], help='选择机械手型号')

    args = parser.parse_args()
    print(f"args:{args}\n")

    tracker = VisionProHeadTracker()
    
    # 初始化滤波器
    vx_filter = SlidingFilter()
    vy_filter = SlidingFilter()
    yaw_filter = SlidingFilter()
    
    # 存储数据
    vel_x_list = []
    vel_y_list = []
    yaw_rad_list = []
    yaw_deg_list = []
    time_list = []

    # 图像客户端配置，需与机器人端图像服务器一致
    img_config = {
        'fps': 30,
        'head_camera_type': 'realsense',
        'head_camera_image_shape': [720, 1280],  # 头部摄像头分辨率
        'head_camera_id_numbers': ['337122070060'],
        # 'wrist_camera_type': 'opencv',
        # 'wrist_camera_image_shape': [480, 640],  # 手腕摄像头分辨率
        # 'wrist_camera_id_numbers': [2, 4],
    }

    # 判断是否是双目摄像头
    ASPECT_RATIO_THRESHOLD = 2.0
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False

    # 判断是否有手腕摄像头
    WRIST = 'wrist_camera_type' in img_config

    # 设置tv图像shape
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    # 创建共享内存，存储头部摄像头图像
    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)

    if WRIST:
        # 创建共享内存，存储手腕摄像头图像
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name, 
                                 wrist_img_shape=wrist_img_shape, wrist_img_shm_name=wrist_img_shm.name)
    else:
        img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name)

    # 开启图像接收线程
    image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.start()

    # 创建tv对象，用于XR设备传输头部图像、接收手腕姿态数据
    tv_wrapper = TeleVisionWrapper(BINOCULAR, tv_img_shape, tv_img_shm.name)
    # tv_wrapper = TeleVisionWrapper()

    # 初始化机械臂控制器和逆解器
    if args.arm == 'G1_29':
        arm_ctrl = G1_29_ArmController()
        arm_ik = G1_29_ArmIK()
    elif args.arm == 'G1_23':
        arm_ctrl = G1_23_ArmController()
        arm_ik = G1_23_ArmIK()
    elif args.arm == 'H1_2':
        arm_ctrl = H1_2_ArmController()
        arm_ik = H1_2_ArmIK()
    elif args.arm == 'H1':
        arm_ctrl = H1_ArmController()
        arm_ik = H1_ArmIK()

    # 初始化机械手控制器
    if args.hand == "dex3":
        left_hand_array = Array('d', 75, lock=True)
        right_hand_array = Array('d', 75, lock=True)
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock=False)
        dual_hand_action_array = Array('d', 14, lock=False)
        hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
    elif args.hand == "gripper":
        left_hand_array = Array('d', 75, lock=True)
        right_hand_array = Array('d', 75, lock=True)
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)
        dual_gripper_action_array = Array('d', 2, lock=False)
        gripper_ctrl = Gripper_Controller(left_hand_array, right_hand_array, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)
    elif args.hand == "inspire1":
        left_hand_array = Array('d', 75, lock=True)
        right_hand_array = Array('d', 75, lock=True)
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock=False)
        dual_hand_action_array = Array('d', 12, lock=False)
        hand_ctrl_ = Inspire_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)

    # 如果要录制数据，初始化录制器
    if args.record:
        recorder = EpisodeWriter(task_dir=args.task_dir, frequency=args.frequency, rerun_log=True)
        recording = False

    try:
        # 等待用户输入，确认开始运行
        user_input = input("请输入开始信号（输入'r'开始后续程序）：\n")
        if user_input.lower() == 'r':
            arm_ctrl.speed_gradual_max()

            # 初始化LCM通信
            lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
            msg_vel = arm_vel_lcmt()
            
            running = True
            start_time0 = time.time()
            while running:
                start_time = time.time()
                # 从XR设备获取当前头部矩阵、手腕姿态、手骨架数据
                head_mat, left_wrist, right_wrist, left_hand, right_hand = tv_wrapper.get_data()
                # print('right_hand:',right_hand)
                tracker.update_matrix(head_mat)
                raw_vel_x, raw_vel_y, height, raw_yaw_rad, _ = tracker.get_metrics()

                vx_filter.add(raw_vel_x)
                vy_filter.add(raw_vel_y)
                yaw_filter.add(raw_yaw_rad)

                if vx_filter.ready() and vy_filter.ready() and yaw_filter.ready():
                    vel_x = vx_filter.get_filtered_value()
                    vel_y = vy_filter.get_filtered_value()
                    yaw_rad = yaw_filter.get_filtered_value()
                    yaw_deg = math.degrees(yaw_rad)

                    # 保存数据
                    now = time.time() - start_time0
                    time_list.append(now)
                    vel_x_list.append(vel_x * 2)
                    vel_y_list.append(vel_y * 2)
                    yaw_rad_list.append(yaw_rad)
                    yaw_deg_list.append(yaw_deg)
                
                    msg_vel.vel_x = vel_x * 2
                    msg_vel.vel_y = vel_y * 2
                    msg_vel.height = height
                    msg_vel.yaw_rad = 0
                    msg_vel.yaw_deg = 0

                    # print('vel_x:',msg_vel.vel_x )
                    # print('vel_y:',msg_vel.vel_y)
                    lc.publish("arm_vel", msg_vel.encode())
                    
                
                # 更新机械手输入数据
                if args.hand:
                    left_hand_array[:] = left_hand.flatten()
                    right_hand_array[:] = right_hand.flatten()
                    # print('left_hand_array:', left_hand_array[:])

                # 获取机械臂当前关节位置、速度
                current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
                current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

                # 使用IK求解逆解，得到目标关节角和关节力矩
                sol_q, sol_tauff = arm_ik.solve_ik(left_wrist, right_wrist, current_lr_arm_q, current_lr_arm_dq)

                # 控制机械臂动作
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

                # 显示头部图像（缩小一半）
                tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                cv2.imshow("record image", tv_resized_image)

                # 键盘监听
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                elif key == ord('s') and args.record:
                    recording = not recording
                    if recording:
                        if not recorder.create_episode():
                            recording = False
                    else:
                        recorder.save_episode()

                # 如果在录制状态，保存当前数据
                if args.record:
                    # 获取机械手当前状态
                    if args.hand == "dex3":
                        with dual_hand_data_lock:
                            left_hand_state = dual_hand_state_array[:7]
                            right_hand_state = dual_hand_state_array[-7:]
                            left_hand_action = dual_hand_action_array[:7]
                            right_hand_action = dual_hand_action_array[-7:]
                    elif args.hand == "gripper":
                        with dual_gripper_data_lock:
                            left_hand_state = [dual_gripper_state_array[1]]
                            right_hand_state = [dual_gripper_state_array[0]]
                            left_hand_action = [dual_gripper_action_array[1]]
                            right_hand_action = [dual_gripper_action_array[0]]
                    elif args.hand == "inspire1":
                        left_hand_state = dual_hand_state_array[:6]
                        right_hand_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]

                    # 复制当前图像
                    current_tv_image = tv_img_array.copy()
                    if WRIST:
                        current_wrist_image = wrist_img_array.copy()

                    # 机械臂状态和动作
                    left_arm_state = current_lr_arm_q[:7]
                    right_arm_state = current_lr_arm_q[-7:]
                    left_arm_action = sol_q[:7]
                    right_arm_action = sol_q[-7:]

                    # 整理录制数据格式
                    if recording:
                        colors = {}
                        depths = {}

                        if BINOCULAR:
                            colors[f"color_{0}"] = current_tv_image[:, :tv_img_shape[1] // 2]
                            colors[f"color_{1}"] = current_tv_image[:, tv_img_shape[1] // 2:]
                            if WRIST:
                                colors[f"color_{2}"] = current_wrist_image[:, :wrist_img_shape[1] // 2]
                                colors[f"color_{3}"] = current_wrist_image[:, wrist_img_shape[1] // 2:]
                        else:
                            colors[f"color_{0}"] = current_tv_image
                            if WRIST:
                                colors[f"color_{1}"] = current_wrist_image[:, :wrist_img_shape[1] // 2]
                                colors[f"color_{2}"] = current_wrist_image[:, wrist_img_shape[1] // 2:]

                        states = {
                            "left_arm": {"qpos": left_arm_state.tolist(), "qvel": [], "torque": []},
                            "right_arm": {"qpos": right_arm_state.tolist(), "qvel": [], "torque": []},
                            "left_hand": {"qpos": left_hand_state, "qvel": [], "torque": []},
                            "right_hand": {"qpos": right_hand_state, "qvel": [], "torque": []},
                            "body": None,
                        }
                        actions = {
                            "left_arm": {"qpos": left_arm_action.tolist(), "qvel": [], "torque": []},
                            "right_arm": {"qpos": right_arm_action.tolist(), "qvel": [], "torque": []},
                            "left_hand": {"qpos": left_hand_action, "qvel": [], "torque": []},
                            "right_hand": {"qpos": right_hand_action, "qvel": [], "torque": []},
                            "body": None,
                        }

                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

                # 控制程序循环频率
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / float(args.frequency)) - time_elapsed)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("捕获到Ctrl+C, 程序终止...")
        print("停止程序，准备绘图...")

        # 绘图
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(time_list, vel_x_list, label='vel_x')
        plt.title("Velocity X")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity X")
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(time_list, vel_y_list, label='vel_y')
        plt.title("Velocity Y")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity Y")
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(time_list, yaw_rad_list, label='yaw_rad')
        plt.title("Yaw (rad)")
        plt.xlabel("Time (s)")
        plt.ylabel("Yaw (rad)")
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.plot(time_list, yaw_deg_list, label='yaw_deg')
        plt.title("Yaw (deg)")
        plt.xlabel("Time (s)")
        plt.ylabel("Yaw (deg)")
        plt.grid()

        plt.tight_layout()
        plt.show()
    finally:
        # 结束时释放资源
        arm_ctrl.ctrl_dual_arm_go_home()
        tv_img_shm.unlink()
        tv_img_shm.close()
        if WRIST:
            wrist_img_shm.unlink()
            wrist_img_shm.close()
        if args.record:
            recorder.close()
        print("程序退出...")
        exit(0)
