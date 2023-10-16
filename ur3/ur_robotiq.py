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

from ur3.grippers import CortexRobotiqGripper
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
        rmp_path: str,
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        use_motion_commander=True,
    ):
        if not os.path.isdir(rmp_path):
            raise FileNotFoundError("RMP path is not a directory")
        if not os.path.isfile(os.path.join(rmp_path, "rmp_config.json")):
            raise FileNotFoundError("RMP path does not contain rmp_config.json")
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError("URDF path is not a file")
        rmp_config_dir = os.path.join(rmp_path, "rmp_config.json")

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

        self.right_gripper_commander = CortexRobotiqGripper(self, ["right_inner_knuckle_joint", "right_outer_knuckle_joint", "right_inner_finger_joint",
                                                                   "left_inner_knuckle_joint", "finger_joint", "left_inner_finger_joint" ])
        self.add_commander("gripper", self.right_gripper_commander)

    def initialize(self, physics_sim_view: omni.physics.tensors.SimulationView = None):
        super().initialize(physics_sim_view)

        verbose = True
        # kps=[67108] * (self.num_dof - 4) + [10000000] * 4,
        # kds=[107374] * (self.num_dof - 4) + [200000] * 4,
        kps=[15000] * (self.num_dof - 6) + [11459] * 6,
        kds=[1500] * (self.num_dof - 6) + [1145] * 6,
        
        if verbose:
            print("setting UR gains:")
            print("- kps: {}".format(kps))
            print("- kds: {}".format(kds))
        self.get_articulation_controller().set_gains(kps, kds)

