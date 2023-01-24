from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import sys
from collections import defaultdict

import carb
import omni
import omni.graph.core as og
from omni.isaac.core import World
from omni.isaac.core.objects.cuboid import (DynamicCuboid, FixedCuboid,
                                            VisualCuboid)
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.extensions import (disable_extension,
                                              enable_extension,
                                              get_extension_path_from_name)
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.utils.stage import (add_reference_to_stage,
                                         is_stage_loading)
from omni.isaac.core_nodes.scripts.utils import set_target_prims
from omni.usd import Gf
from omni.isaac.cortex.cortex_world import CortexWorld
from omni.isaac.cortex.motion_commander import PosePq
from omni.isaac.cortex.cortex_utils import load_behavior_module

disable_extension("omni.isaac.ros_bridge")
enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()
from threading import Event, Thread

import numpy as np
# Note that this is not the system level rclpy, but one compiled for omniverse
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from omni.isaac.core.utils.nucleus import get_assets_root_path
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import pyquaternion as pyq

from ur3.ur_t42_robot import CortexUR

import time


class Subscriber(Node):
    def __init__(self):
        super().__init__("isaac_sim_loop")
        self.tracking_enabled = True
        self.movement_sub = self.create_subscription(Bool, "activate_tracking", self.enable_tracking, 10)
        self.decider_activate = self.create_subscription(Bool, "activate_behavior", self.enable_behavior, 10)
        self.ros_sub = self.create_subscription(Pose, "right_hand/pose", self.move_right_cube_callback, 10)
        self.trigger_sub = self.create_subscription(Bool, "right_hand/trigger", self.right_trigger_callback, 10)
        self.cube_sub = self.create_subscription(PoseStamped, "detected_cubes", self.get_cubes, 1)
        self.timeline = omni.timeline.get_timeline_interface()
        self.right_cube_pose = None
        self.left_cube_pose = None
        self.trigger = False
        self.global_tracking = True
        self.robot_state = {}
        self.cubes_pose = {}
        self.existing_cubes = {}
        self.rubiks_path = "omniverse://127.0.0.1/Isaac/Props/Rubiks_Cube/rubiks_cube.usd"
        self.nvidia_cube = "omniverse://127.0.0.1/Isaac/Props/Blocks/nvidia_cube.usd"
        self.time = time.time()
        self.ros_time = time.time()
        self.setup_scene()
        self.create_action_graph()

    def enable_tracking(self, data: Bool):
        self.global_tracking = data.data

    def enable_behavior(self, data: Bool):
        self.global_tracking = False
        decider_network = load_behavior_module("/home/ubb/Documents/Baxter_isaac/ROS2/src/isaac_sim/block_stacking_behavior.py").make_decider_network(self.ur_robot)
        self.ros_world.add_decider_network(decider_network)

    def get_cubes(self, data: PoseStamped):
        self.cubes_pose[data.header.frame_id + "_block"] = (
            (data.pose.position.x, data.pose.position.y, data.pose.position.z),
            (data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z),
        )
        # print(f"Ros Loop for {data.header.frame_id + '_block'}: {time.time()-self.ros_time}")
        # self.ros_time = time.time()


    def create_cubes(self):

        for name, pose in self.cubes_pose.items():
            if name not in self.existing_cubes.keys():
                # self.existing_cubes[name] = VisualCuboid(
                #     f"/World/{name}",
                #     position=pose[0],
                #     orientation=pose[1],
                #     size=np.array([0.04, 0.04, 0.04]),
                #     color=np.array([1, 0, 0]),
                # )
                add_reference_to_stage(
                    usd_path=self.rubiks_path,
                    prim_path=f"/World/Obs/{name}",
                )
                self.existing_cubes[name] = XFormPrim(
                    prim_path=f"/World/Obs/{name}",
                    name=f"{name}",
                    position=pose[0],
                    orientation=pose[1],
                    scale=np.array([0.0056, 0.0056, 0.0056]),
                )  # w,x,y,z

            else:
                try:
                    self.existing_cubes[name].set_world_pose(*pose)
                except:
                    carb.log_warn(f"Object with name '{name}' has been ignored")
                    # self.existing_cubes.pop(name, None)


    def move_right_cube_callback(self, data: Pose):
        if data.position.x > 3000:
            self.tracking_enabled = False
        else:
            self.tracking_enabled = True

        q1 = pyq.Quaternion(x=data.orientation.x, y=data.orientation.y, z=data.orientation.z, w=data.orientation.w)
        mul_rot = pyq.Quaternion(w=0.0, x=0.0, y=0.707, z=0.707)  ## Handles axis correction
        offset_rot = pyq.Quaternion(w=0.5, x=-0.5, y=-0.5, z=0.5)  ## Handles axis correction

        q1 = mul_rot * q1
        q1 *= offset_rot

        self.right_cube_pose = (
            (-data.position.x, data.position.z, data.position.y),
            (q1.w, q1.x, q1.y, q1.z),
        )


    def right_trigger_callback(self, data: Bool):
        self.trigger = data.data

    def rclpy_spinner(self, event):
        while not event.is_set():
            rclpy.spin_once(self)

    def run_simulation(self):
        # Track any movements of the robot base
        event = Event()
        rclpy_thread = Thread(target=self.rclpy_spinner, args=[event])
        rclpy_thread.start()

        self.timeline.play()
        while simulation_app.is_running():
            self.ros_world.step(render=True)
            # rclpy.spin_once(self, timeout_sec=0.0)
            if self.ros_world.is_playing():
                if self.ros_world.current_time_step_index == 0:
                    self.ros_world.reset()

                # print(f"Main Loop: {time.time()-self.time}")
                # self.time = time.time()
                self.create_cubes()

                # # Query the current obstacle position
                if self.global_tracking:  ## Make sure this is not enable when working with corte
                    if self.trigger and self.tracking_enabled:
                        self.ur_robot.gripper.close(speed= .5)
                    elif not self.ur_robot.gripper.is_open():
                        self.ur_robot.gripper.open(speed= .5)

                    if self.tracking_enabled:
                        if self.right_cube_pose is not None:
                            self.right_cube.set_world_pose(*self.right_cube_pose)
                        self.ur_robot.arm.send_end_effector(target_pose=PosePq(*self.right_cube.get_world_pose()))


        # Cleanup
        self.timeline.stop()
        event.set()
        rclpy_thread.join()
        self.destroy_node()
        simulation_app.close()

    def setup_scene(self):
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            simulation_app.close()
            sys.exit()
        usd_path = assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd"
        omni.usd.get_context().open_stage(usd_path)
        # Wait two frames so that stage starts loading
        simulation_app.update()
        simulation_app.update()
        print("Loading stage...")

        while is_stage_loading():
            simulation_app.update()
        print("Loading Complete")
        self.ros_world = CortexWorld(stage_units_in_meters=1.0)

        self.urdf_path = "/home/ubb/Documents/Baxter_isaac/ROS2/src/ur_t42/ur_isaac/urdfs/ur_t42.urdf"

        self.ur_robot = self.ros_world.add_robot(CortexUR(name="ur", urdf_path=self.urdf_path))

        self.stage = simulation_app.context.get_stage()
        self.table = self.stage.GetPrimAtPath("/Root/table_low_327")
        self.table.GetAttribute('xformOp:translate').Set(Gf.Vec3f(0.0,0.25,-0.8))
        self.table.GetAttribute('xformOp:rotateZYX').Set(Gf.Vec3f(0,0,0))
        self.table.GetAttribute('xformOp:scale').Set(Gf.Vec3f(1,0.86,1.15))

        # Create a cuboid to visualize where the "panda_hand" frame is according to the kinematics"
        self.right_cube = VisualCuboid(
            "/World/Control/right_cube",
            position=np.array([0.3, -0.1, 0.15]),
            orientation=np.array([0, -0.707, 0, -0.707]),
            size=0.005,
            color=np.array([0, 0, 1]),
        )
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
            # add_reference_to_stage(
            #     usd_path=self.rubiks_path,
            #     prim_path=f"/cortex/belief/objects/{name[i]}",
            # )
            # self.existing_cubes[name[i]] = XFormPrim(
            #     prim_path=f"/cortex/belief/objects/{name[i]}",
            #     name=name[i],
            #     position=np.array([0.5 + ((i + 1) % 2) / 10, 0.0, 0.25]),
            #     orientation=np.array([1, 0, 0, 0]),
            #     scale=np.array([0.0056, 0.0056, 0.0056]),
            # )  # w,x,y,z
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
            # self.ur_robot.register_obstacle(obj)


        ### DO NOT DELETE THIS !!! Will throw errors about undefined
        self.ros_world.reset()
        simulation_app.update()
        self.ros_world.step()  # Step physics to trigger cortex_sim extension self.ur_robot to be created.
        self.ur_robot.initialize()


    def create_action_graph(self):
        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ReadSystemTime", "omni.isaac.core_nodes.IsaacReadSystemTime"),
                        ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                        ("PublishTF", "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
                        # ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                        # ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                        ("ReadSystemTime.outputs:systemTime", "PublishJointState.inputs:timeStamp"),
                        # ("ReadSimTime.outputs:systemTime", "PublishClock.inputs:timeStamp"),
                        ("ReadSystemTime.outputs:systemTime", "PublishTF.inputs:timeStamp"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishJointState.inputs:topicName", "joint_states_sim"),
                        ("PublishTF.inputs:topicName", "tf_sim"),
                    ],
                },
            )
        except Exception as e:
            print(e)

        # Setting the /Franka target prim to Publish JointState node
        set_target_prims(primPath="/ActionGraph/PublishJointState", targetPrimPaths=[self.ur_robot.prim_path])

        # Setting the /Franka target prim to Publish Transform Tree node
        set_target_prims(
            primPath="/ActionGraph/PublishTF",
            inputName="inputs:targetPrims",
            targetPrimPaths=[self.ur_robot.prim_path],
        )

        simulation_app.update()

if __name__ == "__main__":
    rclpy.init()
    subscriber = Subscriber()
    subscriber.run_simulation()
