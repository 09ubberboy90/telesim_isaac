# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from collections import defaultdict
import os
from typing import Optional, Sequence

import carb
import numpy as np
import omni
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.cortex.robot import MotionCommandedRobot, CortexGripper, DirectSubsetCommander
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.core.articulations import Articulation, ArticulationSubset
import omni.isaac.motion_generation.interface_config_loader as icl
from omni.isaac.motion_generation import ArticulationMotionPolicy, RmpFlowSmoothed
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.cortex.motion_commander import MotionCommander

def import_baxter_robot(urdf_path):
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = False
    import_config.fix_base = True
    import_config.distance_scale = 1
    # Get the urdf file path

    # Finally import the robot
    return omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=urdf_path, import_config=import_config
    )


class Baxter(Robot):
    """[summary]

    Args:
        prim_path (str): [description]
        name (str, optional): [description]. Defaults to "ur10_robot".
        usd_path (Optional[str], optional): [description]. Defaults to None.
        position (Optional[np.ndarray], optional): [description]. Defaults to None.
        orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
        attach_gripper (bool, optional): [description]. Defaults to False.
        gripper_usd (Optional[str], optional): [description]. Defaults to "default".

    Raises:
        NotImplementedError: [description]
    """

    def __init__(
        self,
        # prim_path: str,
        urdf_path: str,
        name: str = "baxter",
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        attach_gripper: bool = True,
    ) -> None:
        # prim = get_prim_at_path(prim_path)
        self._end_effector = None
        # if not prim.IsValid():
        self._attach_gripper = attach_gripper
        result, self.baxter_prim = import_baxter_robot(urdf_path)
        super().__init__(
            prim_path=self.baxter_prim,
            name=name,
            position=position,
            orientation=orientation,
            articulation_controller=None,
        )
        # home_config = [0.0, -0.55, 0.0, 0.75, 0.0, 1.26, 0.0, 0.0, -0.55, 0.0, 0.75, 0.0, 1.26, 0.0]
        # self.set_joints_default_state(positions=home_config)
        if attach_gripper:
            self._left_end_effector_prim_path = self.baxter_prim + "/" + "left_gripper"
            self._right_end_effector_prim_path = self.baxter_prim + "/" + "right_gripper"
            self._left_gripper = ParallelGripper(
                self._left_end_effector_prim_path,
                joint_prim_names=["l_gripper_l_finger_joint", "l_gripper_r_finger_joint"],
                joint_closed_positions=[0.0, 0.0],
                joint_opened_positions=[0.020833, -0.020833],
            )

            self._right_gripper = ParallelGripper(
                self._right_end_effector_prim_path,
                joint_prim_names=["r_gripper_l_finger_joint", "r_gripper_r_finger_joint"],
                joint_closed_positions=[0.0, 0.0],
                joint_opened_positions=[0.020833, -0.020833],
            )

    def initialize_gripper(self, gripper: ParallelGripper, physics_sim_view=None):
        gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )

    def initialize_grippers(self, physics_sim_view=None):

        self._right_end_effector = RigidPrim(
            prim_path=self._right_end_effector_prim_path, name=self.name + "_right_end_effector"
        )
        self._right_end_effector.initialize(physics_sim_view)
        self._left_end_effector = RigidPrim(
            prim_path=self._left_end_effector_prim_path, name=self.name + "_left_end_effector"
        )
        self._left_end_effector.initialize(physics_sim_view)

        self.initialize_gripper(self.left_gripper, physics_sim_view=physics_sim_view)
        self.initialize_gripper(self.right_gripper, physics_sim_view=physics_sim_view)

    @property
    def attach_gripper(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        return self._attach_gripper

    @property
    def end_effector(self) -> RigidPrim:
        """[summary]

        Returns:
            RigidPrim: [description]
        """
        return self._end_effector

    @property
    def gripper(self) -> ParallelGripper:
        """[summary]

        Returns:
            ParallelGripper: [description]
        """
        return self._right_gripper

    @property
    def right_gripper(self) -> ParallelGripper:
        """[summary]

        Returns:
            ParallelGripper: [description]
        """
        return self._right_gripper

    @property
    def left_gripper(self) -> ParallelGripper:
        """[summary]

        Returns:
            ParallelGripper: [description]
        """
        return self._left_gripper

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]"""
        super().initialize(physics_sim_view)
        if self._attach_gripper:
            self.initialize_grippers(physics_sim_view)
        self.disable_gravity()
        return

    def post_reset(self) -> None:
        """[summary]"""
        super().post_reset()
        self.right_gripper.post_reset()
        self.left_gripper.post_reset()
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.right_gripper.joint_dof_indicies[0], mode="position"
        )
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.right_gripper.joint_dof_indicies[1], mode="position"
        )
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.left_gripper.joint_dof_indicies[0], mode="position"
        )
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.left_gripper.joint_dof_indicies[1], mode="position"
        )


class CortexBaxter(MotionCommandedRobot):
    def __init__(
        self,
        name: str,
        urdf_path: str,
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        use_motion_commander=True,
    ):
        rmp_config_dir = os.path.join("/home/ubb/Documents/Baxter_isaac/ROS2/src/baxter_stack/baxter_joint_controller/rmpflow", "right_config.json")

        motion_policy_config = icl._process_policy_config(rmp_config_dir)
        result, self.baxter_prim = import_baxter_robot(urdf_path)
        super().__init__(
            name=name,
            prim_path=self.baxter_prim,
            motion_policy_config=motion_policy_config,
            position=position,
            orientation=orientation,
            settings=MotionCommandedRobot.Settings(
                active_commander=use_motion_commander, smoothed_rmpflow=True, smoothed_commands=True
            ),
        )
        left_rmp_config_dir = os.path.join("/home/ubb/Documents/Baxter_isaac/ROS2/src/baxter_stack/baxter_joint_controller/rmpflow", "left_config.json")

        left_motion_policy_config = icl._process_policy_config(left_rmp_config_dir)

        self.left_motion_policy = RmpFlowSmoothed(**left_motion_policy_config)
        articulation_motion_policy = ArticulationMotionPolicy(
            robot_articulation=self, motion_policy=self.left_motion_policy, default_physics_dt=self.commanders_step_dt
        )
        target_prim = VisualCuboid("/World/left_motion_commander_target", size=0.01, color=np.array([0.15, 0.15, 0.15]))
        self.left_arm_commander = MotionCommander(
            self, articulation_motion_policy, target_prim, use_smoothed_commands=True
        )

        self.add_commander("left_arm", self.left_arm_commander)

        self.right_gripper_commander = BaxterGripper(self, ["r_gripper_l_finger_joint", "r_gripper_r_finger_joint"])
        self.add_commander("gripper", self.right_gripper_commander)
        self.left_gripper_commander = BaxterGripper(self, ["l_gripper_l_finger_joint", "l_gripper_r_finger_joint"])
        self.add_commander("left_gripper", self.left_gripper_commander)

    def initialize(self, physics_sim_view: omni.physics.tensors.SimulationView = None):
        super().initialize(physics_sim_view)

        verbose = True
        kps=[536868] * (self.num_dof - 4) + [10000000] * 4,
        kds=[68710400] * (self.num_dof - 4) + [200000] * 4,
        if verbose:
            print("setting Baxter gains:")
            print("- kps: {}".format(kps))
            print("- kds: {}".format(kds))
        self.get_articulation_controller().set_gains(kps, kds)

class BaxterGripper(CortexGripper):
    def __init__(self, articulation, joints):
        super().__init__(
            articulation_subset=ArticulationSubset(articulation, joints),
            opened_width=0.041666,
            closed_width=0.00,
        )

    def joints_to_width(self, joint_positions):
        """ The width is simply the sum of the two prismatic joints.
        """
        return abs(joint_positions[0]) + abs(joint_positions[1])

    def width_to_joints(self, width):
        """ Each joint is half of the width since the width is their sum.
        """
        return np.array([width / 2, - width / 2])
