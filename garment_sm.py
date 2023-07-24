# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
from omni.isaac.cortex.df import DfNetwork, DfState, DfStateMachineDecider, DfStateSequence, DfTimedDeciderState, DfWaitState
from omni.isaac.cortex.dfb import DfContext
from omni.isaac.cortex.dfb import DfOpenGripper, DfCloseGripper
from omni.isaac.cortex.motion_commander import ApproachParams, PosePq
import omni.isaac.cortex.math_util as math_util

def make_target_rotation(target_p):
    return math_util.matrix_to_quat(
        math_util.make_rotation_matrix(az_dominant=np.array([0.0, 0.0, -1.0]), ax_suggestion=-target_p)
    )

class ReachState(DfState):
    def __init__(self, target_p):
        self.target_p = target_p

    def enter(self):
        target_q = make_target_rotation(self.target_p)
        self.target = PosePq(self.target_p, target_q)
        approach_params = ApproachParams(direction=np.array([0.0, 0.0, -0.1]), std_dev=0.04)
        self.context.robot.arm.send_end_effector(self.target, approach_params=approach_params)

    def step(self):
        if np.linalg.norm(self.target_p - self.context.robot.arm.get_fk_p()) < 0.01:
            return None
        return self


def make_decider_network(robot):
    p1 = np.array([0.7, -0.07, -0.2])
    p2 = np.array([0.7, -0.07, 0.3])
    root = DfStateMachineDecider(DfStateSequence([
        DfOpenGripper(),
        ReachState(p1),
        DfCloseGripper(), 
        DfWaitState(wait_time=1.5),
        ReachState(p2)
        ],loop=False))

    return DfNetwork(root, context=DfContext(robot))