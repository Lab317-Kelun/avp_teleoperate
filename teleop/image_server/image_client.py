import cv2
import zmq
import numpy as np
import time
import struct
from collections import deque
from multiprocessing import shared_memory

class ImageClient:
    def __init__(self, tv_img_shape = None, tv_img_shm_name = None, wrist_img_shape = None, wrist_img_shm_name = None, 
                       image_show = False, server_address = "192.168.123.164", port = 5555, Unit_Test = False):
        """
        初始化图像客户端。

        参数说明：
        tv_img_shape: 头部摄像头图像的分辨率 (高, 宽, 通道数)，需与服务器端一致。
        tv_img_shm_name: 用于头部图像共享内存的名称（跨进程访问）。
        wrist_img_shape: 手腕摄像头图像的分辨率 (高, 宽, 通道数)，通常与 tv_img_shape 相同。
        wrist_img_shm_name: 用于手腕图像共享内存的名称。
        image_show: 是否实时显示接收到的图像。
        server_address: 服务器端 IP 地址（运行 image server 脚本的机器）。
        port: 连接服务器的端口号，需要与服务器端设置一致。
        Unit_Test: 是否启用单元测试模式，用于评估图像传输性能（延迟、丢帧率、抖动等）。
        """
        self.running = True
        self._image_show = image_show
        self._server_address = server_address
        self._port = port

        self.tv_img_shape = tv_img_shape
        self.wrist_img_shape = wrist_img_shape

        # 初始化头部相机共享内存
        self.tv_enable_shm = False
        if self.tv_img_shape is not None and tv_img_shm_name is not None:
            self.tv_image_shm = shared_memory.SharedMemory(name=tv_img_shm_name)
            self.tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = self.tv_image_shm.buf)
            self.tv_enable_shm = True
        
        # 初始化手腕相机共享内存
        self.wrist_enable_shm = False
        if self.wrist_img_shape is not None and wrist_img_shm_name is not None:
            self.wrist_image_shm = shared_memory.SharedMemory(name=wrist_img_shm_name)
            self.wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = self.wrist_image_shm.buf)
            self.wrist_enable_shm = True

        # 是否启用性能评估
        self._enable_performance_eval = Unit_Test
        if self._enable_performance_eval:
            self._init_performance_metrics()

    def _init_performance_metrics(self):
        """
        初始化性能评估相关变量。
        """
        self._frame_count = 0  # 已接收的帧数
        self._last_frame_id = -1  # 上一帧的ID

        # 用于实时计算FPS的时间窗口
        self._time_window = 1.0  # 时间窗口大小（单位：秒）
        self._frame_times = deque()  # 记录时间窗口内每帧接收的时间戳

        # 传输质量统计
        self._latencies = deque()  # 时间窗口内每帧的延迟
        self._lost_frames = 0  # 总丢失帧数
        self._total_frames = 0  # 依据frame_id推测的总帧数

    def _update_performance_metrics(self, timestamp, frame_id, receive_time):
        """
        更新每一帧的性能指标。
        参数：
            timestamp：帧发送时的时间戳
            frame_id：帧编号
            receive_time：接收时刻的时间戳
        """
        # 计算单帧延迟
        latency = receive_time - timestamp
        self._latencies.append(latency)

        # 移除时间窗口之外的数据
        while self._latencies and self._frame_times and self._latencies[0] < receive_time - self._time_window:
            self._latencies.popleft()

        # 更新帧接收时间
        self._frame_times.append(receive_time)
        while self._frame_times and self._frame_times[0] < receive_time - self._time_window:
            self._frame_times.popleft()

        # 检测丢帧
        expected_frame_id = self._last_frame_id + 1 if self._last_frame_id != -1 else frame_id
        if frame_id != expected_frame_id:
            lost = frame_id - expected_frame_id
            if lost < 0:
                print(f"[Image Client] 收到乱序帧ID: {frame_id}")
            else:
                self._lost_frames += lost
                print(f"[Image Client] 检测到丢帧数: {lost}, 预期帧ID: {expected_frame_id}, 实际接收帧ID: {frame_id}")
        self._last_frame_id = frame_id
        self._total_frames = frame_id + 1

        self._frame_count += 1

    def _print_performance_metrics(self, receive_time):
        """
        每30帧打印一次实时性能指标。
        """
        if self._frame_count % 30 == 0:
            # 计算实时FPS
            real_time_fps = len(self._frame_times) / self._time_window if self._time_window > 0 else 0

            # 计算延迟指标
            if self._latencies:
                avg_latency = sum(self._latencies) / len(self._latencies)
                max_latency = max(self._latencies)
                min_latency = min(self._latencies)
                jitter = max_latency - min_latency
            else:
                avg_latency = max_latency = min_latency = jitter = 0

            # 计算丢帧率
            lost_frame_rate = (self._lost_frames / self._total_frames) * 100 if self._total_frames > 0 else 0

            print(f"[Image Client] 实时FPS: {real_time_fps:.2f}, 平均延迟: {avg_latency*1000:.2f} ms, 最大延迟: {max_latency*1000:.2f} ms, \
                  最小延迟: {min_latency*1000:.2f} ms, 抖动: {jitter*1000:.2f} ms, 丢帧率: {lost_frame_rate:.2f}%")
    
    def _close(self):
        """
        关闭客户端，释放资源。
        """
        self._socket.close()
        self._context.term()
        if self._image_show:
            cv2.destroyAllWindows()
        print("图像客户端已关闭。")

    def receive_process(self):
        """
        图像接收主循环。
        """
        # 创建ZMQ上下文和socket
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{self._server_address}:{self._port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print("\n图像客户端已启动, 等待接收数据...")
        try:
            while self.running:
                # 接收消息
                message = self._socket.recv()
                receive_time = time.time()

                if self._enable_performance_eval:
                    header_size = struct.calcsize('dI')
                    try:
                        # 尝试提取消息头
                        header = message[:header_size]
                        jpg_bytes = message[header_size:]
                        timestamp, frame_id = struct.unpack('dI', header)
                    except struct.error as e:
                        print(f"[Image Client] 解析头部错误: {e}, 丢弃该消息。")
                        continue
                else:
                    # 如果没有头部，整个消息是图像数据
                    jpg_bytes = message

                # 解码图像
                np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                current_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                if current_image is None:
                    print("[Image Client] 解码图像失败。")
                    continue

                # 写入共享内存
                if self.tv_enable_shm:
                    np.copyto(self.tv_img_array, np.array(current_image[:, :self.tv_img_shape[1]]))
                
                if self.wrist_enable_shm:
                    np.copyto(self.wrist_img_array, np.array(current_image[:, -self.wrist_img_shape[1]:]))
                
                # 显示图像
                if self._image_show:
                    height, width = current_image.shape[:2]
                    resized_image = cv2.resize(current_image, (width // 2, height // 2))
                    cv2.imshow('Image Client Stream', resized_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False

                # 更新性能评估
                if self._enable_performance_eval:
                    self._update_performance_metrics(timestamp, frame_id, receive_time)
                    self._print_performance_metrics(receive_time)

        except KeyboardInterrupt:
            print("用户中断图像客户端。")
        except Exception as e:
            print(f"[Image Client] 接收数据时发生错误: {e}")
        finally:
            self._close()

if __name__ == "__main__":
    # 示例1：使用共享内存接收头部相机图像
    # tv_img_shape = (480, 1280, 3)
    # img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    # img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=img_shm.buf)
    # img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = img_shm.name)
    # img_client.receive_process()

    # 示例2：打开性能评估，本地测试
    # client = ImageClient(image_show = True, server_address='127.0.0.1', Unit_Test=True) 

    # 示例3：实际部署测试
    client = ImageClient(image_show = True, server_address='192.168.123.164', Unit_Test=False)
    client.receive_process()
