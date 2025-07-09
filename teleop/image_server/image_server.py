import cv2
import zmq
import time
import struct
from collections import deque
import numpy as np
import pyrealsense2 as rs

# RealSense相机封装类
class RealSenseCamera(object):
    def __init__(self, img_shape, fps, serial_number=None, enable_depth=False) -> None:
        """
        初始化RealSense相机
        img_shape: 图像尺寸 [高度, 宽度]
        fps: 帧率
        serial_number: 相机序列号（可选）
        enable_depth: 是否开启深度图
        """
        self.img_shape = img_shape
        self.fps = fps
        self.serial_number = serial_number
        self.enable_depth = enable_depth

        align_to = rs.stream.color  # 将深度图对齐到彩色图
        self.align = rs.align(align_to)
        self.init_realsense()  # 初始化RealSense设备

    def init_realsense(self):
        """初始化RealSense设备流"""
        self.pipeline = rs.pipeline()
        config = rs.config()

        if self.serial_number is not None:
            config.enable_device(self.serial_number)  # 绑定具体序列号的相机

        config.enable_stream(rs.stream.color, self.img_shape[1], self.img_shape[0], rs.format.bgr8, self.fps)

        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.img_shape[1], self.img_shape[0], rs.format.z16, self.fps)

        profile = self.pipeline.start(config)
        self._device = profile.get_device()

        if self._device is None:
            print('[图像服务器] pipeline profile无法获取设备。')

        if self.enable_depth:
            depth_sensor = self._device.first_depth_sensor()
            self.g_depth_scale = depth_sensor.get_depth_scale()  # 深度图比例因子

        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()  # 相机内参

    def get_frame(self):
        """获取一帧图像数据"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)  # 对齐深度和彩色图
        color_frame = aligned_frames.get_color_frame()

        if self.enable_depth:
            depth_frame = aligned_frames.get_depth_frame()

        if not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())  # 彩色图转为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data()) if self.enable_depth else None
        return color_image, depth_image

    def release(self):
        """释放相机资源"""
        self.pipeline.stop()

# OpenCV相机封装类
class OpenCVCamera():
    def __init__(self, device_id, img_shape, fps):
        """
        初始化OpenCV相机
        device_id: 相机设备ID (/dev/video*或编号)
        img_shape: 图像尺寸 [高度, 宽度]
        """
        self.id = device_id
        self.fps = fps
        self.img_shape = img_shape
        self.cap = cv2.VideoCapture(self.id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))  # 设置编码格式MJPG
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.img_shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # 测试相机是否能正常读取帧
        if not self._can_read_frame():
            print(f"[图像服务器] 相机 {self.id} 错误：无法初始化或读取帧。")
            self.release()

    def _can_read_frame(self):
        """检测是否能成功读取一帧"""
        success, _ = self.cap.read()
        return success

    def release(self):
        """释放相机资源"""
        self.cap.release()

    def get_frame(self):
        """读取一帧图像"""
        ret, color_image = self.cap.read()
        if not ret:
            return None
        return color_image

# 图像服务器类，负责采集、拼接并发送图像数据
class ImageServer:
    def __init__(self, config, port=5555, Unit_Test=False):
        """
        初始化图像服务器
        config: 配置字典, 包含fps、相机类型、分辨率、设备号或序列号等
        port: 通信端口号, 默认5555
        Unit_Test: 是否开启单元测试（打印性能指标）
        """
        print(config)
        self.fps = config.get('fps', 30)
        self.head_camera_type = config.get('head_camera_type', 'opencv')
        self.head_image_shape = config.get('head_camera_image_shape', [480, 640])
        self.head_camera_id_numbers = config.get('head_camera_id_numbers', [0])

        self.wrist_camera_type = config.get('wrist_camera_type', None)
        self.wrist_image_shape = config.get('wrist_camera_image_shape', [480, 640])
        self.wrist_camera_id_numbers = config.get('wrist_camera_id_numbers', None)

        self.port = port
        self.Unit_Test = Unit_Test

        # 初始化头部相机列表
        self.head_cameras = []
        if self.head_camera_type == 'opencv':
            for device_id in self.head_camera_id_numbers:
                camera = OpenCVCamera(device_id=device_id, img_shape=self.head_image_shape, fps=self.fps)
                self.head_cameras.append(camera)
        elif self.head_camera_type == 'realsense':
            for serial_number in self.head_camera_id_numbers:
                camera = RealSenseCamera(img_shape=self.head_image_shape, fps=self.fps, serial_number=serial_number)
                self.head_cameras.append(camera)
        else:
            print(f"[图像服务器] 不支持的头部相机类型: {self.head_camera_type}")

        # 初始化腕部相机列表（可选）
        self.wrist_cameras = []
        if self.wrist_camera_type and self.wrist_camera_id_numbers:
            if self.wrist_camera_type == 'opencv':
                for device_id in self.wrist_camera_id_numbers:
                    camera = OpenCVCamera(device_id=device_id, img_shape=self.wrist_image_shape, fps=self.fps)
                    self.wrist_cameras.append(camera)
            elif self.wrist_camera_type == 'realsense':
                for serial_number in self.wrist_camera_id_numbers:
                    camera = RealSenseCamera(img_shape=self.wrist_image_shape, fps=self.fps, serial_number=serial_number)
                    self.wrist_cameras.append(camera)
            else:
                print(f"[图像服务器] 不支持的腕部相机类型: {self.wrist_camera_type}")

        # 初始化ZeroMQ通信
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)  # 使用发布者模式
        self.socket.bind(f"tcp://*:{self.port}")

        if self.Unit_Test:
            self._init_performance_metrics()

        # 打印相机初始化信息
        for cam in self.head_cameras:
            if isinstance(cam, OpenCVCamera):
                print(f"[图像服务器] 头部相机 {cam.id} 分辨率: {cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} x {cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            elif isinstance(cam, RealSenseCamera):
                print(f"[图像服务器] 头部相机 {cam.serial_number} 分辨率: {cam.img_shape[0]} x {cam.img_shape[1]}")
            else:
                print("[图像服务器] 未知头部相机类型。")

        for cam in self.wrist_cameras:
            if isinstance(cam, OpenCVCamera):
                print(f"[图像服务器] 腕部相机 {cam.id} 分辨率: {cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} x {cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            elif isinstance(cam, RealSenseCamera):
                print(f"[图像服务器] 腕部相机 {cam.serial_number} 分辨率: {cam.img_shape[0]} x {cam.img_shape[1]}")
            else:
                print("[图像服务器] 未知腕部相机类型。")

        print("[图像服务器] 图像服务器已启动，等待客户端连接...")

    def _init_performance_metrics(self):
        """初始化性能评估指标"""
        self.frame_count = 0
        self.time_window = 1.0  # 1秒的滑动窗口
        self.frame_times = deque()  # 存储最近1秒钟内的帧时间戳
        self.start_time = time.time()

    def _update_performance_metrics(self, current_time):
        """更新性能指标"""
        self.frame_times.append(current_time)
        while self.frame_times and self.frame_times[0] < current_time - self.time_window:
            self.frame_times.popleft()
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        """打印实时性能指标"""
        if self.frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            real_time_fps = len(self.frame_times) / self.time_window
            print(f"[图像服务器] 实时FPS: {real_time_fps:.2f}，发送总帧数: {self.frame_count}，已运行时间: {elapsed_time:.2f}秒")

    def _close(self):
        """释放所有资源"""
        for cam in self.head_cameras:
            cam.release()
        for cam in self.wrist_cameras:
            cam.release()
        self.socket.close()
        self.context.term()
        print("[图像服务器] 服务器已关闭。")

    def send_process(self):
        """发送图像的主循环"""
        try:
            while True:
                # 读取头部相机数据
                head_frames = []
                for cam in self.head_cameras:
                    if self.head_camera_type == 'opencv':
                        color_image = cam.get_frame()
                        if color_image is None:
                            print("[图像服务器] 头部相机读取帧失败。")
                            break
                    elif self.head_camera_type == 'realsense':
                        color_image, _ = cam.get_frame()
                        if color_image is None:
                            print("[图像服务器] 头部相机读取帧失败。")
                            break
                    head_frames.append(color_image)

                if len(head_frames) != len(self.head_cameras):
                    break

                head_color = cv2.hconcat(head_frames)  # 将所有头部图像水平拼接

                # 读取腕部相机数据
                if self.wrist_cameras:
                    wrist_frames = []
                    for cam in self.wrist_cameras:
                        if self.wrist_camera_type == 'opencv':
                            color_image = cam.get_frame()
                            if color_image is None:
                                print("[图像服务器] 腕部相机读取帧失败。")
                                break
                        elif self.wrist_camera_type == 'realsense':
                            color_image, _ = cam.get_frame()
                            if color_image is None:
                                print("[图像服务器] 腕部相机读取帧失败。")
                                break
                        wrist_frames.append(color_image)
                    wrist_color = cv2.hconcat(wrist_frames)

                    # 合并头部和腕部图像
                    full_color = cv2.hconcat([head_color, wrist_color])
                else:
                    full_color = head_color

                # 将图像编码成JPEG格式
                ret, buffer = cv2.imencode('.jpg', full_color)
                if not ret:
                    print("[图像服务器] 图像编码失败。")
                    continue

                jpg_bytes = buffer.tobytes()

                # 打包发送
                if self.Unit_Test:
                    timestamp = time.time()
                    frame_id = self.frame_count
                    header = struct.pack('dI', timestamp, frame_id)  # 打包时间戳和帧编号
                    message = header + jpg_bytes
                else:
                    message = jpg_bytes

                self.socket.send(message)

                if self.Unit_Test:
                    current_time = time.time()
                    self._update_performance_metrics(current_time)
                    self._print_performance_metrics(current_time)

        except KeyboardInterrupt:
            print("[图像服务器] 用户中断。")
        finally:
            self._close()


# 程序崩溃后 重新启动 画面重新传输
import traceback

RESTART_DELAY = 0.2  # 错误后等待秒数再重启
MAX_RESTARTS = 10  # 最大重启次数（可选，可注释）

def run_with_restart():
    restart_count = 0
    while True:
        try:
            config = {
                'fps': 30,
                'head_camera_type': 'realsense',
                'head_camera_image_shape': [1080, 1920],
                'head_camera_id_numbers': ['242222070727'],
                # 'wrist_camera_type': 'opencv',
                # 'wrist_camera_image_shape': [480, 640],
                # 'wrist_camera_id_numbers': [2, 4],
            }

            server = ImageServer(config, Unit_Test=False)
            server.send_process()  # 主逻辑启动

        except KeyboardInterrupt:
            print("[主控] 收到 Ctrl+C, 程序正常退出。")
            break
        
        except Exception as e:
            print(f"[主控] 程序运行异常：{e}")
            traceback.print_exc()

            restart_count += 1
            if restart_count >= MAX_RESTARTS:
                print("[主控] 达到最大重启次数，程序退出。")
                break

            print(f"[主控] {RESTART_DELAY} 秒后自动重启...")
            time.sleep(RESTART_DELAY)
        else:
            print("[主控] 程序正常退出。")
            break  # 正常退出就不再重启

if __name__ == "__main__":
    run_with_restart()

