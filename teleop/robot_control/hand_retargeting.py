from .dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
import yaml
from enum import Enum

# 枚举类，用于指定不同的手型配置路径
class HandType(Enum):
    INSPIRE_HAND = "../assets/inspire_hand/inspire_hand.yml"
    INSPIRE_HAND_Unit_Test = "../../assets/inspire_hand/inspire_hand.yml"
    UNITREE_DEX3 = "../assets/unitree_hand/unitree_dex3.yml"
    UNITREE_DEX3_Unit_Test = "../../assets/unitree_hand/unitree_dex3.yml"

# 手部动作映射类，用于将参考点数据映射为机械手关节角度
class HandRetargeting:
    def __init__(self, hand_type: HandType):
        # 根据手型设置默认URDF目录
        if hand_type == HandType.UNITREE_DEX3:
            RetargetingConfig.set_default_urdf_dir('../assets')
        elif hand_type == HandType.UNITREE_DEX3_Unit_Test:
            RetargetingConfig.set_default_urdf_dir('../../assets')
        elif hand_type == HandType.INSPIRE_HAND:
            RetargetingConfig.set_default_urdf_dir('../assets')
        elif hand_type == HandType.INSPIRE_HAND_Unit_Test:
            RetargetingConfig.set_default_urdf_dir('../../assets')

        # 获取配置文件路径
        config_file_path = Path(hand_type.value)

        try:
            # 读取YAML配置文件
            with config_file_path.open('r') as f:
                self.cfg = yaml.safe_load(f)
                
            # 检查配置文件是否包含 left 和 right 手的配置
            if 'left' not in self.cfg or 'right' not in self.cfg:
                raise ValueError("Configuration file must contain 'left' and 'right' keys.")

            # 初始化左右手的动作映射配置
            left_retargeting_config = RetargetingConfig.from_dict(self.cfg['left'])
            right_retargeting_config = RetargetingConfig.from_dict(self.cfg['right'])
            self.left_retargeting = left_retargeting_config.build()
            self.right_retargeting = right_retargeting_config.build()

            # 获取动作映射器的关节名称列表
            self.left_retargeting_joint_names = self.left_retargeting.joint_names
            self.right_retargeting_joint_names = self.right_retargeting.joint_names

            # 对于 Unitree DEX3 手型，定义与SDK接口的关节顺序
            if hand_type == HandType.UNITREE_DEX3 or hand_type == HandType.UNITREE_DEX3_Unit_Test:
                self.left_dex3_api_joint_names  = [
                    'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint',
                    'left_hand_middle_0_joint', 'left_hand_middle_1_joint', 
                    'left_hand_index_0_joint', 'left_hand_index_1_joint'
                ]
                self.right_dex3_api_joint_names = [
                    'right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
                    'right_hand_middle_0_joint', 'right_hand_middle_1_joint',
                    'right_hand_index_0_joint', 'right_hand_index_1_joint'
                ]

                # 建立从动作映射器索引到真实关节控制器的索引映射
                self.left_dex_retargeting_to_hardware = [
                    self.left_retargeting_joint_names.index(name) for name in self.left_dex3_api_joint_names
                ]
                self.right_dex_retargeting_to_hardware = [
                    self.right_retargeting_joint_names.index(name) for name in self.right_dex3_api_joint_names
                ]

            # 对于 Inspire 手型，定义接口的关节顺序
            elif hand_type == HandType.INSPIRE_HAND or hand_type == HandType.INSPIRE_HAND_Unit_Test:
                self.left_inspire_api_joint_names  = [
                    'L_pinky_proximal_joint', 'L_ring_proximal_joint', 'L_middle_proximal_joint',
                    'L_index_proximal_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_proximal_yaw_joint'
                ]
                self.right_inspire_api_joint_names = [
                    'R_pinky_proximal_joint', 'R_ring_proximal_joint', 'R_middle_proximal_joint',
                    'R_index_proximal_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_proximal_yaw_joint'
                ]

                # 同样建立映射索引
                self.left_dex_retargeting_to_hardware = [
                    self.left_retargeting_joint_names.index(name) for name in self.left_inspire_api_joint_names
                ]
                self.right_dex_retargeting_to_hardware = [
                    self.right_retargeting_joint_names.index(name) for name in self.right_inspire_api_joint_names
                ]

        # 处理文件读取或YAML解析异常
        except FileNotFoundError:
            print(f"Configuration file not found: {config_file_path}")
            raise
        except yaml.YAMLError as e:
            print(f"YAML error while reading {config_file_path}: {e}")
            raise
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
