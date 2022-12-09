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
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction

from ur3.t42_gripper import T42Gripper


class UR_T42_Robot(Robot):
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
        name: str = "ur_t42",
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

            # Finally import the robot
            result, self.ur_prim = omni.kit.commands.execute(
                "URDFParseAndImportFile", urdf_path=urdf_path, import_config=import_config
            )
        super().__init__(
            prim_path=self.ur_prim, name=name, position=position, orientation=orientation, articulation_controller=None
        )

        if attach_gripper:
            self._end_effector_prim_path = self.ur_prim+"/"+"t42_base_link"

            self._gripper = T42Gripper(
                self._end_effector_prim_path,
                joint_prim_names=[                
                    "swivel_2_to_finger_2_1",
                    "finger_2_1_to_finger_2_2",
                    "swivel_1_to_finger_1_1",
                    "finger_1_1_to_finger_1_2",
                ],
                joint_closed_positions=[1.57, 0.785, 1.57, 0.785],
                joint_opened_positions=[0.0, 0.0, 0.0, 0.0],
            )
            # self.initialize_gripper()
        return

    def initialize_gripper(self, gripper: T42Gripper, physics_sim_view=None):
        self._grippers_dof_indices = [None] * len(gripper.joint_prim_names)
        for index in range(self.num_dof):
            dof_handle = self._dc_interface.get_articulation_dof(self._handle, index)
            dof_name = self._dc_interface.get_dof_name(dof_handle)
            for j in range(len(gripper.joint_prim_names)):
                if gripper.joint_prim_names[j] == dof_name:
                    self._grippers_dof_indices[j] = index
        # make sure that all gripper dof names were resolved
        for i in range(len(gripper.joint_prim_names)):
            if self._grippers_dof_indices[i] is None:
                raise Exception("Not all gripper dof names were resolved to dof handles and dof indices.")
        self._grippers_dof_indices = np.array(self._grippers_dof_indices)
        gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.gripper_get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=gripper.joint_prim_names,
        )

    def initialize_grippers(self, physics_sim_view=None):

        self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)
        
        self.initialize_gripper(self.gripper, physics_sim_view=physics_sim_view)
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
    def gripper(self) -> T42Gripper:
        """[summary]

        Returns:
            Gripper: [description]
        """
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]
        """
        super().initialize(physics_sim_view)
        if self._attach_gripper:
            self.initialize_grippers(physics_sim_view)
        self.disable_gravity()
        return

    def post_reset(self) -> None:
        """[summary]
        """
        super().post_reset()
        self.gripper.post_reset()
        for index in self._grippers_dof_indices:
            self._articulation_controller.switch_dof_control_mode(
                dof_index=index, mode="position"
            )
    def apply_action(self, control_actions: ArticulationAction) -> None:
        """Applies actions to all the joints of an articulation that corresponds to the ArticulationAction of the finger joints only.

        Args:
            control_actions (ArticulationAction): ArticulationAction for the left finger joint and the right finger joint respectively.
        """
        joint_actions = ArticulationAction()
        if control_actions.joint_positions is not None:
            joint_actions.joint_positions = [None] * self.num_dof
            for idx, index in enumerate(self._grippers_dof_indices):
                joint_actions.joint_positions[index] = control_actions.joint_positions[idx]
        if control_actions.joint_velocities is not None:
            joint_actions.joint_velocities = [None] * self.num_dof
            for idx, index in enumerate(self._grippers_dof_indices):
                joint_actions.joint_velocities[index] = control_actions.joint_velocities[idx]
        if control_actions.joint_efforts is not None:
            joint_actions.joint_efforts = [None] * self.num_dof
            for idx, index in enumerate(self._grippers_dof_indices):
                joint_actions.joint_efforts[index] = control_actions.joint_efforts[idx]
        super().apply_action(control_actions=joint_actions)
        return

    def gripper_get_joint_positions(self, joint_indices=None) -> None:
        indexes = self._grippers_dof_indices
        if indexes[0] is None:
            return super().get_joint_positions()    
        else:
            return super().get_joint_positions(indexes)    
