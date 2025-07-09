# 这个文件是旧版实现，需要维护或重构。
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize  # DDS通信模块
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # 手部电机指令与状态消息类型
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType  # 手部动作重定向
import numpy as np
from enum import IntEnum
import threading
import time
from multiprocessing import Process, shared_memory, Array, Lock  # 多进程共享内存等

# Inspire 手指尖关节索引（共5个）
inspire_tip_indices = [4, 9, 14, 19, 24]
# Inspire 机械手电机数量（6个关节）
Inspire_Num_Motors = 6
# DDS 通信话题名称
kTopicInspireCommand = "rt/inspire/cmd"    # 手部控制指令话题
kTopicInspireState   = "rt/inspire/state"  # 手部状态反馈话题

class Inspire_Controller:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock = None, dual_hand_state_array = None,
                       dual_hand_action_array = None, fps = 100.0, Unit_Test = False):
        print("初始化 Inspire_Controller...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        if not self.Unit_Test:
            # 实际运行：使用真实 Inspire 手模型
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        else:
            # 单元测试模式：使用模拟手模型，初始化DDS通道工厂
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)
            ChannelFactoryInitialize(0)

        # 初始化控制发布器和状态订阅器
        self.HandCmb_publisher = ChannelPublisher(kTopicInspireCommand, MotorCmds_)
        self.HandCmb_publisher.Init()

        self.HandState_subscriber = ChannelSubscriber(kTopicInspireState, MotorStates_)
        self.HandState_subscriber.Init()

        # 创建左右手状态的共享数组
        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)

        # 创建后台线程，实时接收手部状态
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # 等待DDS订阅完成
        while True:
            if any(self.right_hand_state_array):  # 右手有任何非零数据则认为已接收到状态
                break
            time.sleep(0.01)
            print("[Inspire_Controller] 正在等待 DDS 状态订阅...")

        # 启动控制子进程
        hand_control_process = Process(target=self.control_process, args=(left_hand_array, right_hand_array,
                                                                          self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array))
        hand_control_process.daemon = True
        hand_control_process.start()

        print("Inspire_Controller 初始化完成！\n")

    def _subscribe_hand_state(self):
        """ 后台线程回调函数：持续接收手部状态数据，并更新共享数组 """
        while True:
            hand_msg = self.HandState_subscriber.Read()
            if hand_msg is not None:
                # 提取右手状态（前6个）
                for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
                    self.right_hand_state_array[idx] = hand_msg.states[id].q
                    # print('right_hand_state_array:',self.right_hand_state_array[idx])
                # 提取左手状态（后6个）
                for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
                    self.left_hand_state_array[idx] = hand_msg.states[id].q
                    
            time.sleep(0.005)

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """ 发布左右手目标关节角度控制命令 """
        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[id].q = left_q_target[idx]
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[id].q = right_q_target[idx]
            # print('hand_msg_q:',self.hand_msg.cmds[id].q)
        self.HandCmb_publisher.Write(self.hand_msg)
        

    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None):
        """ 控制主循环进程 """
        self.running = True
        left_q_target  = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        # 初始化手部控制命令消息对象
        self.hand_msg = MotorCmds_()
        # 初始化双手
        self.hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Inspire_Right_Hand_JointIndex) + len(Inspire_Left_Hand_JointIndex))]
        # 初始化右手       
        # self.hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Inspire_Right_Hand_JointIndex))]
        
        # 初始设置所有关节为打开（q=1.0）
        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
             self.hand_msg.cmds[id].q = 1.0
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0

        try:
            while self.running:
                start_time = time.time()

                # 获取左右手的原始3D手势数据 这是目标
                left_hand_mat  = np.array(left_hand_array[:]).reshape(25, 3).copy()
                right_hand_mat = np.array(right_hand_array[:]).reshape(25, 3).copy()

                # 从共享内存中读取当前的手部状态（电机角度）
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                # 检查手势数据是否已初始化
                if not np.all(right_hand_mat == 0.0) and not np.all(left_hand_mat[4] == np.array([-1.13, 0.3, 0.15])):
                    ref_left_value = left_hand_mat[inspire_tip_indices]
                    ref_right_value = right_hand_mat[inspire_tip_indices]

                    # 调用重定向器将 3D 坐标转化为关节角度目标值
                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                    # 归一化电机角度为 [0,1] 范围（参考官方文档）
                    def normalize(val, min_val, max_val):
                        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                    for idx in range(Inspire_Num_Motors):
                        if idx <= 3:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 1.7)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.7)
                        elif idx == 4:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 0.5)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.5)
                        elif idx == 5:
                            left_q_target[idx]  = normalize(left_q_target[idx], -0.1, 1.3)
                            right_q_target[idx] = normalize(right_q_target[idx], -0.1, 1.3)

                # 更新共享内存中的手部状态和动作数据
                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                # 发布控制命令
                self.ctrl_dual_hand(left_q_target, right_q_target)
                
                # print('left_q_target',left_q_target)
                # print('right_q_target',right_q_target)
                
                # 保持固定频率
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            print("Dex3_1_Controller 已关闭。")

# 手部关节编号定义（与官方文档一致）
# https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand
# 0~5 为右手，6~11 为左手
class Inspire_Right_Hand_JointIndex(IntEnum):
    kRightHandPinky = 0
    kRightHandRing = 1
    kRightHandMiddle = 2
    kRightHandIndex = 3
    kRightHandThumbBend = 4
    kRightHandThumbRotation = 5

class Inspire_Left_Hand_JointIndex(IntEnum):
    kLeftHandPinky = 6
    kLeftHandRing = 7
    kLeftHandMiddle = 8
    kLeftHandIndex = 9
    kLeftHandThumbBend = 10
    kLeftHandThumbRotation = 11
