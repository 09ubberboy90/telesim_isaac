from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.objects.cuboid import (DynamicCuboid, FixedCuboid,
                                            VisualCuboid)
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.extensions import (disable_extension,
                                              enable_extension)
from omni.isaac.cortex.motion_commander import PosePq

disable_extension("omni.isaac.ros_bridge")
enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

import numpy as np
import pyquaternion as pyq
# Note that this is not the system level rclpy, but one compiled for omniverse
import rclpy
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from omni.isaac.core.utils.stage import add_reference_to_stage

from baxter.baxter_robot import CortexBaxter
from general_world import TeleopWorld

class Baxter_World(TeleopWorld):
    def __init__(self):
        super().__init__(simulation_app)
        self.ros_sub = self.create_subscription(Pose, "right_hand/pose", self.move_right_cube_callback, 10)
        self.trigger_sub = self.create_subscription(Bool, "right_hand/trigger", self.right_trigger_callback, 10)
        self.ros_sub_2 = self.create_subscription(Pose, "left_hand/pose", self.move_left_cube_callback, 10)
        self.trigger_sub_2 = self.create_subscription(Bool, "left_hand/trigger", self.left_trigger_callback, 10)

    def move_right_cube_callback(self, data: Pose):
        if data.position.x > 3000:
            self.tracking_enabled["right"] = False
        else:
            self.tracking_enabled["right"] = True


        q1 = pyq.Quaternion(x=data.orientation.x, y=data.orientation.y, z=data.orientation.z, w=data.orientation.w)
        mul_rot = pyq.Quaternion(w=0.0, x=0.0, y=0.707, z=0.707)  ## Handles axis correction
        offset_rot = pyq.Quaternion(w=0.5, x=0.5, y=-0.5, z=0.5)  ## Handles overhead rotation
        seminar_rot = pyq.Quaternion(w=0.707, x=0.0, y=0.0, z=-.707)  ## Handles robot specific rotation

        q1 = mul_rot * q1
        q1 *= offset_rot
        q1 = seminar_rot *q1

        self.right_cube_pose = (
            (data.position.z, data.position.x, data.position.y),
            (q1.w, q1.x, q1.y, q1.z),
        )

    def move_left_cube_callback(self, data: Pose):
        # Make sure to respect the parity (Levi-Civita symbol)
        if data.position.x > 3000:
            self.tracking_enabled["left"] = False
        else:
            self.tracking_enabled["left"] = True


        q1 = pyq.Quaternion(x=data.orientation.x, y=data.orientation.y, z=data.orientation.z, w=data.orientation.w)
        mul_rot = pyq.Quaternion(w=0.0, x=0.0, y=0.707, z=0.707)  ## Handles axis correction
        offset_rot = pyq.Quaternion(w=0.5, x=0.5, y=-0.5, z=0.5)  ## Handles axis correction

        q1 = mul_rot * q1
        q1 *= offset_rot

        self.left_cube_pose = (
            (-data.position.x, data.position.z, data.position.y),
            (q1.w, q1.x, q1.y, q1.z),
        )

    def right_trigger_callback(self, data: Bool):
        self.trigger["right"] = data.data

    def left_trigger_callback(self, data: Bool):
        self.trigger["left"] = data.data


    def robot_run_simulation(self):
        self.robot.motion_policy.update_world()

        if self.global_tracking:  ## Make sure this is not enable when working with corte
            if self.trigger["left"] and self.tracking_enabled["left"]:
                self.robot.left_gripper.close(speed= .5)
                # print("closing")
            else:
                self.robot.left_gripper.open(speed= .5)
            if self.trigger["right"] and self.tracking_enabled["right"]:
                self.robot.gripper.close(speed= .5)
            elif not self.robot.gripper.is_open():
                self.robot.gripper.open(speed= .5)

            if self.tracking_enabled["right"]:
                if self.right_cube_pose is not None:
                    self.right_cube.set_world_pose(*self.right_cube_pose)
                self.robot.arm.send_end_effector(target_pose=PosePq(*self.right_cube.get_world_pose()))
            if self.tracking_enabled["left"]:
                if self.left_cube_pose is not None:
                    self.left_cube.set_world_pose(*self.left_cube_pose)
                self.robot.left_arm.send_end_effector(target_pose=PosePq(*self.left_cube.get_world_pose()))


    def setup_robot_scene(self):

        self.urdf_path = "/home/ubb/Documents/Baxter_isaac/ROS2/src/baxter_stack/baxter_joint_controller/urdf/baxter.urdf"

        self.robot = self.ros_world.add_robot(CortexBaxter(name="baxter", urdf_path=self.urdf_path))
        
        add_reference_to_stage(
            usd_path=self.assets_root_path + "/Isaac/Environments/Office/Props/SM_TableC.usd",
            prim_path=f"/World/Tables/table",
        )
        self.table = XFormPrim(
            prim_path=f"/World/Tables/table",
            name="table",
            position=[0.58,-0.04,-0.77],
            orientation=np.array([1, 0, 0, 0]),
            scale=[1, 0.85, 1.25],
        )  # w,x,y,z
        self.cortex_table = FixedCuboid(
            "/World/Tables/cortex_table",
            position=np.array([0.58, -0.04, -0.25]),
            orientation=np.array([1, 0, 0, 0]),
            color=np.array([1, 1, 1]),
            scale=[0.55, 1.0, 0.05],

        )

        # Create a cuboid to visualize where the "panda_hand" frame is according to the kinematics"
        self.right_cube = VisualCuboid(
            "/World/Control/right_cube",
            position=np.array([0.8, -0.1, 0.1]),
            orientation=np.array([0, -1, 0, 0]),
            size=0.005,
            color=np.array([0, 0, 1]),
        )
        self.left_cube = VisualCuboid(
            "/World/Control/left_cube",
            position=np.array([-0.1, 0.5, 0.1]), ## out of the way
            orientation=np.array([0, -1, 0, 0]),
            size=0.005,
            color=np.array([0, 0, 1]),
        )

        # self.create_cortex_cubes()

    def create_cortex_cubes(self):
        name = [
            "red_block",
            "green_block",
            # "blue_block",
            "yellow_block",
        ]
        color = [
            [1, 0, 0],
            [0, 1, 0],
            # [0, 0, 1],
            [1, 1, 0],
        ]
        for i in range(len(name)):
            self.existing_cubes[name[i]] = VisualCuboid(
            f"/World/Obs/{name[i]}",
            name = name[i],
            position=np.array([0.3+((i+1)%len(name))/10, 0.0, -0.18]),
            orientation=np.array([1, 0, 0, 0]),
            size=0.04,
            color=np.array(color[i]),
            )
            obj = self.ros_world.scene.add(self.existing_cubes[name[i]])
            ## TODO: Configure for when used in trackign or in cortex
            # self.robot.register_obstacle(obj)


if __name__ == "__main__":
    rclpy.init()
    subscriber = Baxter_World()
    subscriber.run_simulation()
