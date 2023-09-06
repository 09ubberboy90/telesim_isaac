# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import math
from typing import Callable, List

import numpy as np
import omni.kit.app
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.gripper import Gripper

from omni.isaac.cortex.robot import MotionCommandedRobot, CortexGripper, DirectSubsetCommander
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.core.articulations import Articulation, ArticulationSubset

class CortexT42Gripper(CortexGripper):
    def __init__(self, articulation, joints):
        super().__init__(
            articulation_subset=ArticulationSubset(articulation, joints),
            opened_width=1.0,
            closed_width=0.0,
        )
        self.direct_control = False
        self.last_joint_position = None
    def joints_to_width(self, joint_positions):
        """ The width is simply the sum of the all prismatic joints.
        """
        # print(f"Joint_Pos: {sum(abs(v) for v in joint_positions)}")
        return -sum(abs(v) for v in joint_positions)+3.5


    def width_to_joints(self, width):
        """ Each joint is half of the width since the width is their sum.
        """

        a = (-width+3.5) / 2
        # b = 0.0 
        # if a > 1.7:
        #     a = 1.7
        #     # b = half_width - 1.57
        # print(a)
        return np.array([a, 0.0, a, 0.0])

    def set_gripper(self, joint_positions):
        if joint_positions is None:
            return False
        if len(joint_positions) != self.articulation_subset.num_joints:
            print("[Error] joint_positions must be of length 4")
            return False
        self.direct_control = True
        self.last_joint_position = joint_positions
        return True

    def step(self, dt):
        """ Step is called every cycle as the processing engine for the commands.
        """
        if self.direct_control:
            if self.last_joint_position is not None:
                self.articulation_subset.apply_action(joint_positions=self.last_joint_position)
            self.direct_control = False
        else:
            super().step(dt)

class CortexRobotiqGripper(CortexGripper):
    def __init__(self, articulation, joints):
        super().__init__(
            articulation_subset=ArticulationSubset(articulation, joints),
            opened_width=0.0,
            closed_width=1.4,
        )
        self.direct_control = False
        self.last_joint_position = None
    def joints_to_width(self, joint_positions):
        """ The width is simply the sum of the all prismatic joints.
        """

        return abs(np.mean(joint_positions))


    def width_to_joints(self, width):
        """ Each joint is half of the width since the width is their sum.
        """
        if math.isclose(width, 0.0, abs_tol=0.1):
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        a = width / 2
        
        # Assumming R Knuckle, inner abd outer, then outer finger 
        return np.array([-a, -a, a,
                          -a, a, a])

    def set_gripper(self, joint_positions):
        if joint_positions is None:
            return False
        if len(joint_positions) != self.articulation_subset.num_joints:
            print("[Error] joint_positions must be of length 6")
            return False
        self.direct_control = True
        self.last_joint_position = joint_positions
        return True

    def step(self, dt):
        """ Step is called every cycle as the processing engine for the commands.
        """
        if self.direct_control:
            if self.last_joint_position is not None:
                self.articulation_subset.apply_action(joint_positions=self.last_joint_position)
            self.direct_control = False
        else:
            super().step(dt)
        