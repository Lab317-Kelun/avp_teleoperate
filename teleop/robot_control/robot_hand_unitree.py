# for dex3-1
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_                               # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
# for gripper
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
from enum import IntEnum
import time
import os
import sys
import threading
from multiprocessing import Process, shared_memory, Array, Lock

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from teleop.utils.weighted_moving_filter import WeightedMovingFilter


unitree_tip_indices = [4, 9, 14] # [thumb, index, middle] in OpenXR
Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"


class Dex3_1_Controller:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock = None, dual_hand_state_array = None,
                       dual_hand_action_array = None, fps = 100.0, Unit_Test = False):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_hand_array: [input] Left hand skeleton data (required from XR device) to hand_ctrl.control_process

        right_hand_array: [input] Right hand skeleton data (required from XR device) to hand_ctrl.control_process

        dual_hand_data_lock: Data synchronization lock for dual_hand_state_array and dual_hand_action_array

        dual_hand_state_array: [output] Return left(7), right(7) hand motor state

        dual_hand_action_array: [output] Return left(7), right(7) hand motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing
        """
        print("Initialize Dex3_1_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)
        else:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3_Unit_Test)
            ChannelFactoryInitialize(0)

        # initialize handcmd publisher and handstate subscriber
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', Dex3_Num_Motors, lock=True)  
        self.right_hand_state_array = Array('d', Dex3_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.left_hand_state_array) and any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            print("[Dex3_1_Controller] Waiting to subscribe dds...")

        hand_control_process = Process(target=self.control_process, args=(left_hand_array, right_hand_array,  self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array))
        hand_control_process.daemon = True
        hand_control_process.start()

        print("Initialize Dex3_1_Controller OK!\n")

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None and right_hand_msg is not None:
                # Update left hand state
                for idx, id in enumerate(Dex3_1_Left_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q
                # Update right hand state
                for idx, id in enumerate(Dex3_1_Right_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q
            time.sleep(0.002)
    
    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """set current left, right hand motor state target q"""
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_target[idx]
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_target[idx]

        self.LeftHandCmb_publisher.Write(self.left_msg)
        self.RightHandCmb_publisher.Write(self.right_msg)
        # print("hand ctrl publish ok.")
    
    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None):
        self.running = True

        left_q_target  = np.full(Dex3_Num_Motors, 0)
        right_q_target = np.full(Dex3_Num_Motors, 0)

        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.2

        # initialize dex3-1's left hand cmd msg
        self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q    = q
            self.left_msg.motor_cmd[id].dq   = dq
            self.left_msg.motor_cmd[id].tau  = tau
            self.left_msg.motor_cmd[id].kp   = kp
            self.left_msg.motor_cmd[id].kd   = kd

        # initialize dex3-1's right hand cmd msg
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode  
            self.right_msg.motor_cmd[id].q    = q
            self.right_msg.motor_cmd[id].dq   = dq
            self.right_msg.motor_cmd[id].tau  = tau
            self.right_msg.motor_cmd[id].kp   = kp
            self.right_msg.motor_cmd[id].kd   = kd  

        try:
            while self.running:
                start_time = time.time()
                # get dual hand state
                left_hand_mat  = np.array(left_hand_array[:]).reshape(25, 3).copy()
                right_hand_mat = np.array(right_hand_array[:]).reshape(25, 3).copy()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if not np.all(right_hand_mat == 0.0) and not np.all(left_hand_mat[4] == np.array([-1.13, 0.3, 0.15])): # if hand data has been initialized.
                    ref_left_value = left_hand_mat[unitree_tip_indices]
                    ref_right_value = right_hand_mat[unitree_tip_indices]
                    
                    ref_left_value[0] = ref_left_value[0] * 1.15
                    ref_left_value[1] = ref_left_value[1] * 1.05
                    ref_left_value[2] = ref_left_value[2] * 0.95
                    
                    # 左手固定
                    # ref_left_value[0] = 0
                    # ref_left_value[1] = 0
                    # ref_left_value[2] = 0
                    
                    ref_right_value[0] = ref_right_value[0] * 1.15
                    ref_right_value[1] = ref_right_value[1] * 1.05
                    ref_right_value[2] = ref_right_value[2] * 0.95

                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))    
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            print("Dex3_1_Controller has been closed.")

class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


unitree_gripper_indices = [4, 9] # [thumb, index]
Gripper_Num_Motors = 2
kTopicGripperCommand = "rt/unitree_actuator/cmd"
kTopicGripperState = "rt/unitree_actuator/state"

class Gripper_Controller:
    def __init__(self, left_hand_array, right_hand_array, dual_gripper_data_lock=None, dual_gripper_state_out=None, dual_gripper_action_out=None, 
                       filter=True, fps=200.0, Unit_Test=False):
        """
        初始化夹爪控制器 Gripper_Controller。

        参数说明：
        left_hand_array:         [输入] 左手骨骼数据数组 (25x3)，通过 multiprocessing.Array 传入；
        right_hand_array:        [输入] 右手骨骼数据数组 (25x3)，通过 multiprocessing.Array 传入；
        dual_gripper_data_lock:  [同步] 数据写入时的共享锁；
        dual_gripper_state_out:  [输出] 输出的夹爪电机状态值（左、右）；
        dual_gripper_action_out: [输出] 输出的夹爪目标动作值（左、右）；
        filter:                  [可选] 是否对输出动作使用加权滑动滤波（默认启用）；
        fps:                     [可选] 控制频率（默认 200Hz) ;
        Unit_Test:               [可选] 是否启用单元测试（不使用 DDS 时开启）。
        """

        print("初始化 Gripper_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test

        # 如果启用了滤波器，则初始化加权滑动滤波器（默认系数）
        if filter:
            self.smooth_filter = WeightedMovingFilter(np.array([0.5, 0.3, 0.2]), Gripper_Num_Motors)
        else:
            self.smooth_filter = None

        # 单元测试模式初始化 DDS 通道（仅测试时启用）
        if self.Unit_Test:
            ChannelFactoryInitialize(0)

        # 初始化夹爪命令发布通道
        self.GripperCmb_publisher = ChannelPublisher(kTopicGripperCommand, MotorCmds_)
        self.GripperCmb_publisher.Init()

        # 初始化夹爪状态订阅通道
        self.GripperState_subscriber = ChannelSubscriber(kTopicGripperState, MotorStates_)
        self.GripperState_subscriber.Init()

        # 初始化内部夹爪状态列表（用于接收 DDS 状态）
        self.dual_gripper_state = [0.0] * len(Gripper_JointIndex)

        # 启动夹爪状态订阅线程
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_gripper_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # 阻塞等待 DDS 成功订阅夹爪状态
        # while True:
        #     if any(state != 0.0 for state in self.dual_gripper_state):
        #         break
        #     time.sleep(0.01)
        #     print("[Gripper_Controller] 正在等待 DDS 夹爪状态订阅...")

        # 启动控制线程（实时从 XR 设备读取骨骼数据控制夹爪）
        self.gripper_control_thread = threading.Thread(target=self.control_thread, args=(
            left_hand_array, right_hand_array, self.dual_gripper_state,
            dual_gripper_data_lock, dual_gripper_state_out, dual_gripper_action_out
        ))
        self.gripper_control_thread.daemon = True
        self.gripper_control_thread.start()

        print("Gripper_Controller 初始化完成！\n")

    def _subscribe_gripper_state(self):
        """ 订阅夹爪电机状态线程（从 DDS 读取并保存到 self.dual_gripper_state) """
        while True:
            gripper_msg = self.GripperState_subscriber.Read()
            if gripper_msg is not None:
                for idx, id in enumerate(Gripper_JointIndex):
                    self.dual_gripper_state[idx] = gripper_msg.states[id].q  # 读取电机角度 q 值
            time.sleep(0.002)  # 稍微延迟避免过度占用 CPU
    
    def ctrl_dual_gripper(self, gripper_q_target):
        """
        向 DDS 发布当前夹爪电机的目标角度指令

        参数：
        gripper_q_target: 包含两个元素的数组，表示左右夹爪的目标位置
        """
        for idx, id in enumerate(Gripper_JointIndex):
            self.gripper_msg.cmds[id].q = gripper_q_target[idx]  # 设置目标角度

        self.GripperCmb_publisher.Write(self.gripper_msg)  # 发布控制消息

    def control_thread(self, left_hand_array, right_hand_array, dual_gripper_state_in,
                             dual_hand_data_lock=None, dual_gripper_state_out=None, dual_gripper_action_out=None):
        """
        控制线程函数，实时读取手势数据并计算控制指令发送到夹爪。

        参数：
        left_hand_array, right_hand_array:   输入, 25x3 骨骼点坐标数组（由 XR 设备提供）
        dual_gripper_state_in:               输入, 夹爪当前状态
        dual_hand_data_lock:                 输出状态使用的锁（用于写入共享变量）
        dual_gripper_state_out:              输出夹爪当前状态（外部使用）
        dual_gripper_action_out:             输出夹爪目标动作值（外部使用）
        """

        self.running = True

        # 控制参数设置
        DELTA_GRIPPER_CMD = 0.18         # 控制速度限制，每次最大只能移动 0.18 rad（相当于夹爪开合 3mm）
        THUMB_INDEX_DISTANCE_MIN = 0.05  # XR 检测到的拇指与食指最近距离（5cm）
        THUMB_INDEX_DISTANCE_MAX = 0.07  # XR 检测到的拇指与食指最远距离（7cm）
        LEFT_MAPPED_MIN  = 0.0           # 左夹爪起始闭合角度
        RIGHT_MAPPED_MIN = 0.0           # 右夹爪起始闭合角度
        LEFT_MAPPED_MAX  = LEFT_MAPPED_MIN + 7.0   # 左夹爪最大张开角度（7.0cm）
        RIGHT_MAPPED_MAX = RIGHT_MAPPED_MIN + 7.0  # 右夹爪最大张开角度（7.0cm）

        # 初始目标动作值为中间位置
        left_target_action  = (LEFT_MAPPED_MAX - LEFT_MAPPED_MIN) / 2.0
        right_target_action = (RIGHT_MAPPED_MAX - RIGHT_MAPPED_MIN) / 2.0

        # 初始化 DDS 控制消息结构
        dq, tau = 0.0, 0.0
        kp, kd = 5.00, 0.05
        self.gripper_msg = MotorCmds_()
        self.gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Gripper_JointIndex))]
        for id in Gripper_JointIndex:
            self.gripper_msg.cmds[id].dq  = dq
            self.gripper_msg.cmds[id].tau = tau
            self.gripper_msg.cmds[id].kp  = kp
            self.gripper_msg.cmds[id].kd  = kd

        try:
            while self.running:
                start_time = time.time()

                # 读取 XR 设备的手部数据（25 个关键点 × 3D 坐标）
                left_hand_mat  = np.array(left_hand_array[:]).reshape(25, 3).copy()
                right_hand_mat = np.array(right_hand_array[:]).reshape(25, 3).copy()

                # 如果手部数据已初始化（非全 0）
                if not np.all(right_hand_mat == 0.0) and not np.all(left_hand_mat[4] == np.array([-1.13, 0.3, 0.15])):
                    # 计算左右手拇指与食指之间的距离
                    left_euclidean_distance  = np.linalg.norm(left_hand_mat[unitree_gripper_indices[1]] - left_hand_mat[unitree_gripper_indices[0]])
                    right_euclidean_distance = np.linalg.norm(right_hand_mat[unitree_gripper_indices[1]] - right_hand_mat[unitree_gripper_indices[0]])

                    # 将欧氏距离映射为夹爪开合范围
                    left_target_action  = np.interp(left_euclidean_distance, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [LEFT_MAPPED_MIN, LEFT_MAPPED_MAX])
                    right_target_action = np.interp(right_euclidean_distance, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [RIGHT_MAPPED_MIN, RIGHT_MAPPED_MAX])

                    
                # 当前夹爪状态（从 DDS 订阅来的状态）
                #dual_gripper_state = np.array(dual_gripper_state_in[:])

                # 对目标动作做限制，避免一次跳变太大
                #left_actual_action  = np.clip(left_target_action,  dual_gripper_state[1] - DELTA_GRIPPER_CMD, dual_gripper_state[1] + DELTA_GRIPPER_CMD)
                #right_actual_action = np.clip(right_target_action, dual_gripper_state[0] - DELTA_GRIPPER_CMD, dual_gripper_state[0] + DELTA_GRIPPER_CMD)

                # 目标动作数组：顺序是 [左, 右]
                dual_gripper_action = np.array([left_target_action, right_target_action])
                
                
                # 如果启用了滤波器，则平滑输出
                if self.smooth_filter:
                    self.smooth_filter.add_data(dual_gripper_action)
                    dual_gripper_action = self.smooth_filter.filtered_data
                
                
                # 如果有输出数组，则写入当前状态与动作（用于外部显示或记录）
                if dual_gripper_state_out and dual_gripper_action_out:
                    with dual_hand_data_lock:
                        # dual_gripper_state_out[:] = dual_gripper_state - np.array([RIGHT_MAPPED_MIN, LEFT_MAPPED_MIN])
                        dual_gripper_action_out[:] = dual_gripper_action - np.array([RIGHT_MAPPED_MIN, LEFT_MAPPED_MIN])

                # 发送控制指令到 DDS
                self.ctrl_dual_gripper(dual_gripper_action)
                # print('dual_gripper_action', dual_gripper_action)
                
                # 控制频率控制：保持目标 fps
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)

        finally:
            print("Gripper_Controller 控制线程已关闭。")

# 定义夹爪的电机索引（用于读取状态或发送控制）
class Gripper_JointIndex(IntEnum):
    kLeftGripper = 0   # 左夹爪电机索引
    kRightGripper = 1  # 右夹爪电机索引



if __name__ == "__main__":
    import argparse
    from teleop.open_television.tv_wrapper import TeleVisionWrapper
    from teleop.image_server.image_client import ImageClient

    parser = argparse.ArgumentParser()
    parser.add_argument('--dex', action='store_true', help='Use dex3-1 hand')
    parser.add_argument('--gripper', dest='dex', action='store_false', help='Use gripper')
    parser.set_defaults(dex=True)
    args = parser.parse_args()
    print(f"args:{args}\n")

    # image
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 1280],  # Head camera resolution
        'head_camera_id_numbers': [0],
    }
    ASPECT_RATIO_THRESHOLD = 2.0  # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    # image
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = img_shm.buf)
    img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = img_shm.name)
    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    # television and arm
    tv_wrapper = TeleVisionWrapper(BINOCULAR, tv_img_shape, img_shm.name)

    if args.dex:
        left_hand_array = Array('d', 75, lock=True)
        right_hand_array = Array('d', 75, lock=True)
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock=False)  # current left, right hand state(14) data.
        dual_hand_action_array = Array('d', 14, lock=False) # current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, Unit_Test = True)
    else:
        left_hand_array = Array('d', 75, lock=True)
        right_hand_array = Array('d', 75, lock=True)
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        gripper_ctrl = Gripper_Controller(left_hand_array, right_hand_array, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array, Unit_Test = True)


    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):\n")
    if user_input.lower() == 's':
        while True:
            head_rmat, left_wrist, right_wrist, left_hand, right_hand = tv_wrapper.get_data()
            # send hand skeleton data to hand_ctrl.control_process
            left_hand_array[:] = left_hand.flatten()
            right_hand_array[:] = right_hand.flatten()

            # with dual_hand_data_lock:
            #     print(f"state : {list(dual_hand_state_array)} \naction: {list(dual_hand_action_array)} \n")
            time.sleep(0.01)
