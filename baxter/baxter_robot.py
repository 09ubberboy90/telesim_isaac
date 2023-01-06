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

import carb
import numpy as np
import omni
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper


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
        if urdf_path:
            status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
            import_config.merge_fixed_joints = False
            import_config.convex_decomp = False
            import_config.import_inertia_tensor = False
            import_config.fix_base = True
            import_config.distance_scale = 1
            # Get the urdf file path

            # Finally import the robot
            result, self.baxter_prim = omni.kit.commands.execute(
                "URDFParseAndImportFile", urdf_path=urdf_path, import_config=import_config
            )
        super().__init__(
            prim_path=self.baxter_prim,
            name=name,
            position=position,
            orientation=orientation,
            articulation_controller=None,
        )
        # home_config = [0.0, -0.55, 0.0, 0.75, 0.0, 1.26, 0.0, 0.0, -0.55, 0.0, 0.75, 0.0, 1.26, 0.0]
        # self.set_joints_default_state(positions=home_config)
        self._grippers_dof_indices = {}
        if attach_gripper:
            self._left_end_effector_prim_path = self.baxter_prim + "/" + "left_gripper"
            self._right_end_effector_prim_path = self.baxter_prim + "/" + "right_gripper"
            self._left_gripper = ParallelGripper(
                self._left_end_effector_prim_path,
                joint_prim_names=["l_gripper_l_finger_joint", "l_gripper_r_finger_joint"],
                joint_closed_positions=[0.0, 0.0],
                joint_opened_positions=[0.020833, -0.020833],
            )
            # self._left_gripper.initialize(self.baxter_prim, self.get_articulation_controller())

            self._right_gripper = ParallelGripper(
                self._right_end_effector_prim_path,
                joint_prim_names=["r_gripper_l_finger_joint", "r_gripper_r_finger_joint"],
                joint_closed_positions=[0.0, 0.0],
                joint_opened_positions=[0.020833, -0.020833],
            )
            # self.initialize_gripper()
        return

    def initialize_gripper(self, gripper: ParallelGripper, physics_sim_view=None, right_side=True):
        self._grippers_dof_indices[gripper] = [None] * len(gripper.joint_prim_names)
        for index in range(self.num_dof):
            dof_handle = self._dc_interface.get_articulation_dof(self._handle, index)
            dof_name = self._dc_interface.get_dof_name(dof_handle)
            for j in range(len(gripper.joint_prim_names)):
                if gripper.joint_prim_names[j] == dof_name:
                    self._grippers_dof_indices[gripper][j] = index
        # make sure that all gripper dof names were resolved
        for i in range(len(gripper.joint_prim_names)):
            if self._grippers_dof_indices[gripper][i] is None:
                raise Exception("Not all gripper dof names were resolved to dof handles and dof indices.")
        self._grippers_dof_indices[gripper] = np.array(self._grippers_dof_indices[gripper])
        gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.right_apply_action if right_side else self.left_apply_action,
            get_joint_positions_func=self.right_get_joint_positions if right_side else self.left_get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=gripper.joint_prim_names,
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

        self.initialize_gripper(self.left_gripper, physics_sim_view=physics_sim_view, right_side=False)
        self.initialize_gripper(self.right_gripper, physics_sim_view=physics_sim_view, right_side=True)

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

    def apply_action(self, control_actions: ArticulationAction) -> None:
        return self.apply_action(control_actions, self.right_gripper)

    def apply_action(self, control_actions: ArticulationAction, gripper: ParallelGripper) -> None:
        """Applies actions to all the joints of an articulation that corresponds to the ArticulationAction of the finger joints only.

        Args:
            control_actions (ArticulationAction): ArticulationAction for the left finger joint and the right finger joint respectively.
        """
        joint_actions = ArticulationAction()
        if control_actions.joint_positions is not None:
            joint_actions.joint_positions = [None] * self.num_dof
            joint_actions.joint_positions[self._grippers_dof_indices[gripper][0]] = control_actions.joint_positions[0]
            joint_actions.joint_positions[self._grippers_dof_indices[gripper][1]] = control_actions.joint_positions[1]
        if control_actions.joint_velocities is not None:
            joint_actions.joint_velocities = [None] * self.num_dof
            joint_actions.joint_velocities[self._grippers_dof_indices[gripper][0]] = control_actions.joint_velocities[0]
            joint_actions.joint_velocities[self._grippers_dof_indices[gripper][1]] = control_actions.joint_velocities[1]
        if control_actions.joint_efforts is not None:
            joint_actions.joint_efforts = [None] * self.num_dof
            joint_actions.joint_efforts[self._grippers_dof_indices[gripper][0]] = control_actions.joint_efforts[0]
            joint_actions.joint_efforts[self._grippers_dof_indices[gripper][1]] = control_actions.joint_efforts[1]
        super().apply_action(control_actions=joint_actions)
        return

    def right_apply_action(self, control_actions: ArticulationAction) -> None:
        self.apply_action(control_actions, self.right_gripper)

    def left_apply_action(self, control_actions: ArticulationAction) -> None:
        self.apply_action(control_actions, self.left_gripper)

    def right_get_joint_positions(self, joint_indices=None) -> None:
        indexes = self._grippers_dof_indices.get(self.right_gripper, [None] * len(self.right_gripper.joint_prim_names))
        if indexes[0] is None:
            return super().get_joint_positions()
        else:
            return super().get_joint_positions(indexes)

    def left_get_joint_positions(self, joint_indices=None) -> None:
        indexes = self._grippers_dof_indices.get(self.left_gripper, [None] * len(self.left_gripper.joint_prim_names))
        if indexes[0] is None:
            return super().get_joint_positions()
        else:
            return super().get_joint_positions(indexes)
