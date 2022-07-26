# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os, sys
import carb
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": False})

import omni
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.urdf import _urdf
from omni.isaac.core.utils.stage import is_stage_loading
import omni.graph.core as og
from omni.isaac.core_nodes.scripts.utils import set_target_prims
# enable ROS2 bridge extension
enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

# Note that this is not the system level rclpy, but one compiled for omniverse
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
import time
from omni.isaac.core.utils.nucleus import get_assets_root_path, is_file


class Subscriber(Node):
    def __init__(self):
        super().__init__("tutorial_subscriber")

        # setting up the world with a cube

        self.timeline = omni.timeline.get_timeline_interface()
        self.setup_scene()

        # setup the ROS2 subscriber here
        # self.ros_sub = self.create_subscription(Empty, "move_cube", self.move_cube_callback, 10)
        self.ros_world.reset()

    def move_cube_callback(self, data):
        # callback function to set the cube position to a new one upon receiving a (empty) ROS2 message
        if self.ros_world.is_playing():
            self._cube_position = np.array([np.random.rand() * 0.40, np.random.rand() * 0.40, 0.10])

    def run_simulation(self):
        self.timeline.play()
        while simulation_app.is_running():
            self.ros_world.step(render=True)
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.ros_world.is_playing():
                if self.ros_world.current_time_step_index == 0:
                    self.ros_world.reset()

                # the actual setting the cube pose is done here
                # self.ros_world.scene.get_object("cube_1").set_world_pose(self._cube_position)

        # Cleanup
        self.timeline.stop()
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
        self.ros_world = World(stage_units_in_meters=1.0)
        # add a cube in the world
        cube_path = "/cube"
        self.ros_world.scene.add(
            VisualCuboid(
                prim_path=cube_path, name="cube_1", position=np.array([0, 0, 0.1]), size=np.array([1, 1, 1])*0.2
            )
        )
        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = False
        import_config.fix_base = True
        import_config.distance_scale = 1
        # Get the urdf file path

        
        file_name = "/home/ubb/Documents/isaac_sim/src/urdfs/ur_t42.urdf"
        # Finally import the robot
        result, self.ur3 = omni.kit.commands.execute( "URDFParseAndImportFile", urdf_path=file_name,
                                                      import_config=import_config,)
        try:
            stage = simulation_app.context.get_stage()

            # stage.GetPrimAtPath(self.ur3).set_world_pose(np.array([0, 0, 0.2]))
            stage.GetPrimAtPath(self.ur3).GetAttribute("xformOp:translate").Set((0, 0, 0.2))
        except Exception as e:
            print("Failed to get ur3")
            print(e)

        self.create_action_graph()

    def create_action_graph(self):
        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ReadSystemTime", "omni.isaac.core_nodes.IsaacReadSystemTime"),
                        ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                        ("SubscribeJointState", "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                        ("PublishTF", "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
                        # ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                        # ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                        ("ReadSystemTime.outputs:systemTime", "PublishJointState.inputs:timeStamp"),
                        # ("ReadSimTime.outputs:systemTime", "PublishClock.inputs:timeStamp"),
                        ("ReadSystemTime.outputs:systemTime", "PublishTF.inputs:timeStamp"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishJointState.inputs:topicName", "joint_states_sim"),
                        ("SubscribeJointState.inputs:topicName", "joint_states"),
                        ("PublishTF.inputs:topicName", "tf_sim"),
                    ],

                },
            )
        except Exception as e:
            print(e)


        # Setting the /Franka target prim to Subscribe JointState node
        set_target_prims(primPath="/ActionGraph/SubscribeJointState", targetPrimPaths=[self.ur3])

        # Setting the /Franka target prim to Publish JointState node
        set_target_prims(primPath="/ActionGraph/PublishJointState", targetPrimPaths=[self.ur3])

        # Setting the /Franka target prim to Publish Transform Tree node
        set_target_prims(primPath="/ActionGraph/PublishTF", inputName="inputs:targetPrims", targetPrimPaths=[self.ur3])

        simulation_app.update()
if __name__ == "__main__":
    rclpy.init()
    subscriber = Subscriber()
    subscriber.run_simulation()