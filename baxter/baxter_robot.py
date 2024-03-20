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

class CortexBaxter(MotionCommandedRobot):
    def __init__(
        self,
        name: str,
        urdf_path: str,
        rmp_path: str,
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        use_motion_commander=True,
    ):
        if not os.path.isdir(rmp_path):
            raise FileNotFoundError("RMP path is not a directory")
        if not os.path.isfile(os.path.join(rmp_path, "right_config.json")):
            raise FileNotFoundError("RMP path does not contain right_config.json")
        if not os.path.isfile(os.path.join(rmp_path, "left_config.json")):
            raise FileNotFoundError("RMP path does not contain left_config.json")
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError("URDF path is not a file")
        
        rmp_config_dir = os.path.join(rmp_path, "right_config.json")

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
        left_rmp_config_dir = os.path.join(rmp_path, "left_config.json")

        left_motion_policy_config = icl._process_policy_config(left_rmp_config_dir)

        self.left_motion_policy = RmpFlowSmoothed(**left_motion_policy_config)
        articulation_motion_policy = ArticulationMotionPolicy(
            robot_articulation=self, motion_policy=self.left_motion_policy, default_physics_dt=self.commanders_step_dt
        )
        target_prim = VisualCuboid("/World/left_motion_commander_target", size=0.01, color=np.array([0.15, 0.15, 0.15]))
        self.left_arm_commander = MotionCommander(
            articulation_motion_policy, target_prim
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
