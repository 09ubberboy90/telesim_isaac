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
from sensor_msgs.msg import JointState
from omni.isaac.core.utils.stage import add_reference_to_stage
from general_world import TeleopWorld
# from ur3.ur_t42_robot import CortexUR
from ur3.ur_robotiq import CortexUR

class UR_World(TeleopWorld):
    def __init__(self):
        super().__init__(simulation_app)
        self.ros_sub = self.create_subscription(Pose, "right_hand/pose", self.move_right_cube_callback, 10)
        self.gripper_bool = False
        self.controller_sub = self.create_subscription(Bool, "controller_switch", self.controller_switch, 10)
        self.trigger_sub = self.create_subscription(Bool, "right_hand/trigger", self.right_trigger_callback, 10)
        self.gripper_sub = self.create_subscription(JointState, "senseglove_motor", self.gripper_callback, 10)

    def controller_switch(self, data: Bool):
        self.gripper_bool = data.data
        self.robot.gripper.direct_control = data.data

    def set_limits(self, value):
        return min(max(value, 0.0), 115) * np.pi/180.0

    def gripper_callback(self, data: JointState):

        self.robot.gripper.set_gripper([self.set_limits(data.position[0]),0.0, self.set_limits(data.position[1]),0.0])

    def move_right_cube_callback(self, data: Pose):
        if data.position.x > 3000:
            self.tracking_enabled["right"] = False
        else:
            self.tracking_enabled["right"] = True

        q1 = pyq.Quaternion(x=data.orientation.x, y=data.orientation.y, z=data.orientation.z, w=data.orientation.w)
        mul_rot = pyq.Quaternion(w=0.0, x=0.0, y=0.707, z=0.707)  ## Handles axis correction ## Maybe needs to be removed and other adjusted
        x_rot = pyq.Quaternion(w=0.707, x=-0.707, y=-0.0, z=0.0)  ## Handles X rotation
        y_rot = pyq.Quaternion(w=0.707, x=0.0, y=-0.707, z=0.0)  ## Handles Y rotation

        offset_rot = pyq.Quaternion(w=0.0, x=0.0, y=0.0, z=1.0)  ## Handles Changing axis

        q1 = mul_rot * q1
        q1 *= x_rot
        q1 *= y_rot
        
        q1 = offset_rot * q1

        self.right_cube_pose = (
            (data.position.x, -data.position.z, data.position.y),
            (q1.w, q1.x, q1.y, q1.z),
        )


    def right_trigger_callback(self, data: Bool):
        self.trigger = data.data

    def robot_run_simulation(self):
        self.robot.motion_policy.update_world()
        if self.global_tracking:  ## Make sure this is not enable when working with corte
            if (not self.gripper_bool) and self.tracking_enabled["right"]:
                if self.trigger and self.robot.gripper.is_open():
                    ## Passing None need to make sure the self.command = None is commented out in the cortex file
                    self.robot.gripper.close(speed= None)
                    # print("Closing gripper")
                elif not (self.robot.gripper.is_open() and self.trigger):
                    ## Passing None need to make sure the self.command = None is commented out in the cortex file
                    self.robot.gripper.open(speed= None)
                    # print("Opening gripper")

            if self.tracking_enabled["right"]:
                if self.right_cube_pose is not None:
                    self.right_cube.set_world_pose(*self.right_cube_pose)
                self.robot.arm.send_end_effector(target_pose=PosePq(*self.right_cube.get_world_pose()))


    def setup_robot_scene(self):
        self.urdf_path = "/home/ubb/Documents/Baxter_isaac/ROS2/src/ur_robotiq/ur_robotiq_isaac/urdfs/ur5e_robotiq.urdf"
        # self.urdf_path = "/home/ubb/Documents/Baxter_isaac/ROS2/src/ur_t42/ur_isaac/urdfs/ur_t42.urdf"

        self.robot = self.ros_world.add_robot(CortexUR(name="ur", urdf_path=self.urdf_path))
        add_reference_to_stage(
            usd_path=self.assets_root_path + "/Isaac/Environments/Office/Props/SM_TableC.usd",
            prim_path=f"/World/Tables/table",
        )
        self.table = XFormPrim(
            prim_path=f"/World/Tables/table",
            name="table",
            position=np.array([0.0, 0.25, -0.76]),
            orientation=np.array([0.707, 0, 0, 0.707]),
            scale=[1.4, 1.3, 1.7]
        )  # w,x,y,z

        # self.table2 = VisualCuboid(
        #     "/World/Tables/table",
        #     position=np.array([0.0, 0.25, -0.4]),
        #     orientation=np.array([1, 0, 0, 0]),
        #     color=np.array([0, 0.2196, 0.3961]),
        #     size=0.8,
        #     scale=[1.8625, 1, 1]

        # )
        self.cortex_table = FixedCuboid(
            "/World/Tables/cortex_table",
            position=np.array([0.0, 0.25, -0.08]),
            orientation=np.array([1, 0, 0, 0]),
            color=np.array([0, 0.2196, 0.3961]),
            size=0.8,
            scale=[1.8625, 1, 0.1]
            

        )

        # Create a cuboid to visualize where the "panda_hand" frame is according to the kinematics"
        self.right_cube = VisualCuboid(
            "/World/Control/right_cube",
            position=np.array([0.07, 0.3, 1.02]),
            orientation=np.array([0.5,-.5,-.5,-.5]),
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
    subscriber = UR_World()
    subscriber.run_simulation()
