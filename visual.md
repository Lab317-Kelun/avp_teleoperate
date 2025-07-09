avp_teleoperate/
│
├── assets                    [存储机器人 URDF 相关文件]
│
├── teleop
│   ├── image_server
│   │     ├── image_client.py      [用于从机器人图像服务器接收图像数据]
│   │     ├── image_server.py      [从摄像头捕获图像并通过网络发送（在机器人板载计算单元上运行）]
│   │
│   ├── open_television
│   │      ├── television.py       [使用 Vuer 从 Apple Vision Pro 捕获腕部和手部数据] 
│   │      ├── tv_wrapper.py       [对捕获的数据进行后处理]
│   │
│   ├── robot_control
│   │      ├── robot_arm_ik.py     [手臂的逆运动学] 
│   │      ├── robot_arm.py        [控制双臂关节并锁定其他部分]
│   │      ├── robot_hand_inspire.py  [控制因时灵巧手]
│   │      ├── robot_hand_unitree.py  [控制宇树灵巧手]
│   │
│   ├── utils
│   │      ├── episode_writer.py          [用于记录模仿学习的数据] 
│   │      ├── mat_tool.py                [一些小的数学工具]
│   │      ├── weighted_moving_filter.py  [用于过滤关节数据的滤波器]
│   │      ├── rerun_visualizer.py        [用于可视化录制数据]
│   │
│   │──teleop_hand_and_arm.py    [遥操作的启动执行代码]
