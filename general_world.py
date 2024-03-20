import os
import sys
from collections import defaultdict

import carb
import omni
import omni.graph.core as og
from omni.isaac.core.utils.stage import (add_reference_to_stage,
                                         is_stage_loading)
from omni.isaac.core_nodes.scripts.utils import set_target_prims
from omni.isaac.cortex.cortex_utils import load_behavior_module
from omni.isaac.cortex.cortex_world import CortexWorld
from omni.isaac.core.objects.cuboid import (DynamicCuboid, FixedCuboid,
                                            VisualCuboid)
from omni.isaac.core.utils.prims import (delete_prim, get_prim_at_path,
                                         is_prim_path_valid)

import time
from threading import Event, Thread

import numpy as np
# Note that this is not the system level rclpy, but one compiled for omniverse
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from omni.isaac.core.utils.nucleus import get_assets_root_path
from rclpy.node import Node
from std_msgs.msg import Bool, Empty


class TeleopWorld(Node):
    def __init__(self, simulation_app):
        super().__init__("isaac_sim_loop")
        self.simulation_app = simulation_app
        self._robot = None
        self._robot_2 = None # TODO: This is a temporary hack 
        self.cortex_table = None
        self.tracking_enabled = defaultdict(lambda: True)
        self.movement_sub = self.create_subscription(Bool, "activate_tracking", self.enable_tracking, 10)
        self.timeline = omni.timeline.get_timeline_interface()
        self.right_cube_pose = None
        self.left_cube_pose = None
        self.trigger = defaultdict(lambda: False)
        self.global_tracking = True
        self.cubes_pose = {}
        self.existing_cubes = {}
        self.rubiks_path = "omniverse://127.0.0.1/Isaac/Props/Rubiks_Cube/rubiks_cube.usd"
        self.nvidia_cube = "omniverse://127.0.0.1/Isaac/Props/Blocks/nvidia_cube.usd"
        self.time = time.time()
        self.ros_time = time.time()
        self.setup_scene()
        self.create_action_graph()
        self.stage = omni.usd.get_context().get_stage()

    @property
    def robot(self):
        if self._robot is None:
            raise RuntimeError("Robot not set")
        return self._robot
    
    @robot.setter
    def robot(self, value):
        self._robot = value

    def robot_run_simulation(self):
        raise NotImplementedError

    def setup_robot_scene(self):
        raise NotImplementedError

    def enable_tracking(self, data: Bool):
        self.global_tracking = data.data

    def rclpy_spinner(self, event):
        while not event.is_set():
            rclpy.spin_once(self)

    def run_simulation(self):
        # Track any movements of the robot base
        event = Event()
        rclpy_thread = Thread(target=self.rclpy_spinner, args=[event])
        rclpy_thread.start()

        self.timeline.play()
        while self.simulation_app.is_running():
            self.ros_world.step(render=True)
            # rclpy.spin_once(self, timeout_sec=0.0)
            if self.ros_world.is_playing():
                if self.ros_world.current_time_step_index == 0:
                    self.ros_world.reset()

                self.robot_run_simulation()

        # Cleanup
        self.timeline.stop()
        event.set()
        rclpy_thread.join()
        self.destroy_node()
        self.simulation_app.close()
    
    def setup_scene(self):
        self.assets_root_path = get_assets_root_path()
        if self.assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            self.simulation_app.close()
            sys.exit()
        usd_path = self.assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd"
        omni.usd.get_context().open_stage(usd_path)
        # Wait two frames so that stage starts loading
        self.simulation_app.update()

        while is_stage_loading():
            self.simulation_app.update()
        print("Loading Complete")
        self.ros_world = CortexWorld(stage_units_in_meters=1.0)
        delete_prim("/Root/table_low_327")

        self.setup_robot_scene()
        ### DO NOT DELETE THIS !!! Will throw errors about undefined
        self.ros_world.reset()
        self.simulation_app.update()
        self.ros_world.step()  # Step physics to trigger cortex_sim extension self._robot to be created.
        if self._robot is not None:
            self._robot.initialize()
        else:
            carb.log_error("Robot not defined")
            self.exit()
        if self.cortex_table is not None:
            self._robot.motion_policy.add_obstacle(self.cortex_table)
            if self._robot_2 is not None:
                self._robot_2.motion_policy.add_obstacle(self.cortex_table)

        # self._robot.motion_policy.visualize_collision_spheres()
        # self._robot_2.motion_policy.visualize_collision_spheres()

    def create_action_graph(self):
        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("Delay", "omni.graph.action.Delay"),
                        ("ReadSystemTime", "omni.isaac.core_nodes.IsaacReadSystemTime"),
                        ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                        ("PublishJointState_2", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                        ("PublishTF", "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
                        ("PublishTF_2", "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
                        # ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "Delay.inputs:execIn"),
                        ("Delay.outputs:finished", "PublishJointState.inputs:execIn"),
                        ("Delay.outputs:finished", "PublishJointState_2.inputs:execIn"),
                        ("Delay.outputs:finished", "PublishTF.inputs:execIn"),
                        ("Delay.outputs:finished", "PublishTF_2.inputs:execIn"),
                        # ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                        ("ReadSystemTime.outputs:systemTime", "PublishJointState.inputs:timeStamp"),
                        ("ReadSystemTime.outputs:systemTime", "PublishJointState_2.inputs:timeStamp"),
                        # ("ReadSimTime.outputs:systemTime", "PublishClock.inputs:timeStamp"),
                        ("ReadSystemTime.outputs:systemTime", "PublishTF.inputs:timeStamp"),
                        ("ReadSystemTime.outputs:systemTime", "PublishTF_2.inputs:timeStamp"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishJointState.inputs:topicName", "joint_states_sim"),
                        ("PublishJointState_2.inputs:topicName", "joint_states_sim_right"),
                        ("PublishTF.inputs:topicName", "tf_sim"),
                        ("PublishTF_2.inputs:topicName", "tf_sim_right"),
                        ("Delay.inputs:duration", 0.1),
                    ],
                },
            )
        except Exception as e:
            print(e)

        # Setting the /Franka target prim to Publish JointState node
        set_target_prims(primPath="/ActionGraph/PublishJointState", targetPrimPaths=[self._robot.prim_path])
        # Setting the /Franka target prim to Publish Transform Tree node
        set_target_prims(
            primPath="/ActionGraph/PublishTF",
            inputName="inputs:targetPrims",
            targetPrimPaths=[self._robot.prim_path],
        )


        if self._robot_2 is not None:
            set_target_prims(primPath="/ActionGraph/PublishJointState_2", targetPrimPaths=[self._robot_2.prim_path])
            set_target_prims(
                primPath="/ActionGraph/PublishTF_2",
                inputName="inputs:targetPrims",
                targetPrimPaths=[self._robot_2.prim_path],
            )

        self.simulation_app.update()

