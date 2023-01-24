# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Callable, List

import numpy as np
import omni.kit.app
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.gripper import Gripper

from omni.isaac.cortex.robot import MotionCommandedRobot, CortexGripper, DirectSubsetCommander
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.core.articulations import Articulation, ArticulationSubset

class T42Gripper(Gripper):

    def __init__(
        self,
        end_effector_prim_path: str,
        joint_prim_names: List[str],
        joint_opened_positions: np.ndarray,
        joint_closed_positions: np.ndarray,
        action_deltas: np.ndarray = None,
    ) -> None:
        Gripper.__init__(self, end_effector_prim_path=end_effector_prim_path)
        self._joint_prim_names = joint_prim_names
        self._joint_dof_indicies = np.array([None]*4)
        self._joint_opened_positions = joint_opened_positions
        self._joint_closed_positions = joint_closed_positions
        self._get_joint_positions_func = None
        self._set_joint_positions_func = None
        self._action_deltas = action_deltas
        self._articulation_num_dofs = None
        return

    @property
    def joint_opened_positions(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: joint positions of the finger joints when opened.
        """
        return self._joint_opened_positions

    @property
    def joint_closed_positions(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: joint positions of the finger joints when closed.
        """
        return self._joint_closed_positions

    @property
    def joint_dof_indicies(self) -> np.ndarray:
        """
        Returns:
            dict: joint dof indices in the articulation of the finger joints.
        """
        return self._joint_dof_indicies

    @property
    def joint_prim_names(self) -> List[str]:
        """
        Returns:
            List[str]: the finger joint prim names.
        """
        return self._joint_prim_names

    def initialize(
        self,
        articulation_apply_action_func: Callable,
        get_joint_positions_func: Callable,
        set_joint_positions_func: Callable,
        dof_names: List,
        physics_sim_view: omni.physics.tensors.SimulationView = None,
    ) -> None:
        """Create a physics simulation view if not passed and creates a rigid prim view using physX tensor api.
            This needs to be called after each hard reset (i.e stop + play on the timeline) before interacting with any
            of the functions of this class.

        Args:
            articulation_apply_action_func (Callable): apply_action function from the Articulation class.
            get_joint_positions_func (Callable): get_joint_positions function from the Articulation class.
            set_joint_positions_func (Callable): set_joint_positions function from the Articulation class.
            dof_names (List): dof names from the Articulation class.
            physics_sim_view (omni.physics.tensors.SimulationView, optional): current physics simulation view. Defaults to None

        Raises:
            Exception: _description_
        """
        Gripper.initialize(self, physics_sim_view=physics_sim_view)
        self._get_joint_positions_func = get_joint_positions_func
        self._articulation_num_dofs = len(dof_names)
        for index, name in enumerate(dof_names):
            i = next((i for i, t in enumerate(self.joint_prim_names) if name == t), None)
            if i is not None and i < self._joint_dof_indicies.shape[0]:
                self._joint_dof_indicies[i] = index
        # make sure that all gripper dof names were resolved
        for val in self._joint_dof_indicies:
            if val is None:
                raise Exception("Not all gripper dof names were resolved to dof handles and dof indices.")
        self._articulation_apply_action_func = articulation_apply_action_func
        current_joint_positions = get_joint_positions_func()
        if self._default_state is None:

            self._default_state = []
            for val in self._joint_dof_indicies:
                self._default_state.append(current_joint_positions[val])
            self._default_state = np.array(self._default_state)
        self._set_joint_positions_func = set_joint_positions_func
        return

    def open(self) -> None:
        """Applies actions to the articulation that opens the gripper (ex: to release an object held).
        """
        self._articulation_apply_action_func(self.forward(action="open"))
        return

    def close(self) -> None:
        """Applies actions to the articulation that closes the gripper (ex: to hold an object).
        """
        self._articulation_apply_action_func(self.forward(action="close"))
        return

    def set_action_deltas(self, value: np.ndarray) -> None:
        """
        Args:
            value (np.ndarray): deltas to apply for finger joint positions when openning or closing the gripper. 
                               [left, right]. Defaults to None.
        """
        self._action_deltas = value
        return

    def get_action_deltas(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: deltas that will be applied for finger joint positions when openning or closing the gripper. 
                        [left, right]. Defaults to None.
        """
        return self._action_deltas

    def set_default_state(self, joint_positions: np.ndarray) -> None:
        """Sets the default state of the gripper

        Args:
            joint_positions (np.ndarray): joint positions of the left finger joint and the right finger joint respectively.
        """
        self._default_state = joint_positions
        return

    def get_default_state(self) -> np.ndarray:
        """Gets the default state of the gripper

        Returns:
            np.ndarray: joint positions of the left finger joint and the right finger joint respectively.
        """
        return self._default_state

    def post_reset(self):
        Gripper.post_reset(self)
        self._set_joint_positions_func(
            positions=self._default_state, joint_indices=self._joint_dof_indicies.astype(int)
        )
        return

    def set_joint_positions(self, positions: np.ndarray) -> None:
        """
        Args:
            positions (np.ndarray): joint positions of the left finger joint and the right finger joint respectively.
        """
        self._set_joint_positions_func(
            positions=positions, joint_indices=self._joint_dof_indicies.astype(int)
        )
        return

    def get_joint_positions(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: joint positions of the left finger joint and the right finger joint respectively.
        """
        return self._get_joint_positions_func(joint_indices=self._joint_dof_indicies.astype(int))

    def forward(self, action: str) -> ArticulationAction:
        """calculates the ArticulationAction for all of the articulation joints that corresponds to "open"
           or "close" actions.

        Args:
            action (str): "open" or "close" as an abstract action.

        Raises:
            Exception: _description_

        Returns:
            ArticulationAction: articulation action to be passed to the articulation itself
                                (includes all joints of the articulation).
        """
        if action == "open":
            target_joint_positions = [None] * self._articulation_num_dofs
            if self._action_deltas is None:
                for idx, index in enumerate(self._joint_dof_indicies):
                    target_joint_positions[index] = self._joint_opened_positions[idx]
            else:
                current_joint_positions = self._get_joint_positions_func()
                for idx, index in enumerate(self._joint_dof_indicies):
                    target_joint_positions[index] = current_joint_positions[index] + self._action_deltas[idx]

        elif action == "close":
            target_joint_positions = [None] * self._articulation_num_dofs
            if self._action_deltas is None:
                for idx, index in enumerate(self._joint_dof_indicies):
                    target_joint_positions[index] = self._joint_closed_positions[idx]
            else:
                current_joint_positions = self._get_joint_positions_func()
                for idx, index in enumerate(self._joint_dof_indicies):
                    target_joint_positions[index] = current_joint_positions[index] - self._action_deltas[idx]
        else:
            raise Exception("action {} is not defined for ParallelGripper".format(action))
        return ArticulationAction(joint_positions=target_joint_positions)

    def apply_action(self, control_actions: ArticulationAction) -> None:
        """Applies actions to all the joints of an articulation that corresponds to the ArticulationAction of the finger joints only.

        Args:
            control_actions (ArticulationAction): ArticulationAction for the left finger joint and the right finger joint respectively.
        """
        joint_actions = ArticulationAction()
        if control_actions.joint_positions is not None:
            joint_actions.joint_positions = [None] * self._articulation_num_dofs
            for idx, index in enumerate(self._joint_dof_indicies):
                joint_actions.joint_positions[index] = control_actions.joint_positions[idx]
        if control_actions.joint_velocities is not None:
            joint_actions.joint_velocities = [None] * self._articulation_num_dofs
            for idx, index in enumerate(self._joint_dof_indicies):
                joint_actions.joint_velocities[index] = control_actions.joint_velocities[idx]
        if control_actions.joint_efforts is not None:
            joint_actions.joint_efforts = [None] * self._articulation_num_dofs
            for idx, index in enumerate(self._joint_dof_indicies):
                joint_actions.joint_efforts[index] = control_actions.joint_efforts[idx]
        self._articulation_apply_action_func(control_actions=joint_actions)
        return

class CortexT42Gripper(CortexGripper):
    def __init__(self, articulation, joints):
        super().__init__(
            articulation_subset=ArticulationSubset(articulation, joints),
            opened_width=4.71,
            closed_width=0.00,
        )

    def joints_to_width(self, joint_positions):
        """ The width is simply the sum of the all prismatic joints.
        """
        return sum(abs(v) for v in joint_positions)


    def width_to_joints(self, width):
        """ Each joint is half of the width since the width is their sum.
        """

        half_width = width / 2
        a = half_width
        b = 0.0 
        if a > 1.57:
            a = 1.57
            b = half_width - 1.57
        return np.array([a, b, a, b])


