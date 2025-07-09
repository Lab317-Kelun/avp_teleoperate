from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorCmd_

import time

kTopicInspireCommand = "rt/inspire/cmd"

def main():
    ChannelFactoryInitialize(0)
    publisher = ChannelPublisher(kTopicInspireCommand, MotorCmds_)
    publisher.Init()

    # 初始化cmds列表，传入所有必需参数
    # 参数顺序：mode, q, dq, tau, kp, kd, reserve
    # reserve 是长度3的整数列表，初始化为0
    cmds = []
    for _ in range(6):
        cmd = MotorCmd_(mode=10, q=0.6, dq=0.0, tau=0.0, kp=0.0, kd=0.0, reserve=[0, 0, 0])
        cmds.append(cmd)

    msg = MotorCmds_()
    msg.cmds = cmds

    print("开始发布控制命令...")

    try:
        while True:
            success = publisher.Write(msg)
            print(f"是否成功发布控制命令？ {success}")
            print(f"控制命令内容： {msg}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("测试停止。")

if __name__ == "__main__":
    main()

