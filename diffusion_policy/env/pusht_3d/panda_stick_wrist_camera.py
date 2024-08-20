import os
import sapien
import numpy as np
from mani_skill.agents.robots import PandaStick
from mani_skill.sensors.camera import CameraConfig
from mani_skill.agents.registration import register_agent
from mani_skill.agents.controllers import PDEEPoseControllerConfig,PDEEPosControllerConfig

CUSTOM_ASSET_DIR = os.environ.get("CUSTOM_ASSET_DIR", None)
if CUSTOM_ASSET_DIR is None:
    raise ValueError("Please set the CUSTOM_ASSET_DIR environment variable.")


@register_agent()
class WristCameraPandaStick(PandaStick):
    uid = "wrist_camera_panda_stick"
    urdf_path = f"{CUSTOM_ASSET_DIR}/robots/panda/wrist_camera_panda_stick.urdf"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]



    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        arm_pd_ee_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False
        )
        
        configs.update({"pd_ee_pos": dict(arm=arm_pd_ee_pos),
                        "pd_ee_pose": dict(arm=arm_pd_ee_pose)})
        return configs

        