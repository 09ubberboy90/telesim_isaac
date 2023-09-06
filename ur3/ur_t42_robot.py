# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from collections import defaultdict
from typing import Optional

import numpy as np
import omni

from ur3.grippers import CortexT42Gripper
import os
from typing import Optional, Sequence

import carb
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.cortex.robot import MotionCommandedRobot, CortexGripper, DirectSubsetCommander
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.core.articulations import Articulation, ArticulationSubset
import omni.isaac.motion_generation.interface_config_loader as icl
from omni.isaac.motion_generation import ArticulationMotionPolicy, RmpFlowSmoothed
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.cortex.motion_commander import MotionCommander

def import_robot(urdf_path):
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


class CortexUR(MotionCommandedRobot):
    def __init__(
        self,
        name: str,
        urdf_path: str,
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        use_motion_commander=True,
    ):
        rmp_config_dir = os.path.join("/home/ubb/Documents/Baxter_isaac/ROS2/src/ur_t42/ur_isaac/config", "rmp_config.json")

        motion_policy_config = icl._process_policy_config(rmp_config_dir)
        result, self.ur_prim = import_robot(urdf_path)
        super().__init__(
            name=name,
            prim_path=self.ur_prim,
            motion_policy_config=motion_policy_config,
            position=position,
            orientation=orientation,
            settings=MotionCommandedRobot.Settings(
                active_commander=use_motion_commander, smoothed_rmpflow=True, smoothed_commands=True
            ),
        )

        self.right_gripper_commander = CortexT42Gripper(self, ["swivel_1_to_finger_1_1","finger_1_1_to_finger_1_2", "swivel_2_to_finger_2_1","finger_2_1_to_finger_2_2"])
        self.add_commander("gripper", self.right_gripper_commander)

    def initialize(self, physics_sim_view: omni.physics.tensors.SimulationView = None):
        super().initialize(physics_sim_view)

        verbose = True
        # kps=[67108] * (self.num_dof - 4) + [10000000] * 4,
        # kds=[107374] * (self.num_dof - 4) + [200000] * 4,
        kps=[34.90659] * (self.num_dof),
        kds=[349.06586] * (self.num_dof - 4) + [1] * 4 ,
        if verbose:
            print("setting UR gains:")
            print("- kps: {}".format(kps))
            print("- kds: {}".format(kds))
        self.get_articulation_controller().set_gains(kps, kds)

