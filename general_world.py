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
        self.tracking_enabled = defaultdict(lambda: True)
        self.movement_sub = self.create_subscription(Bool, "activate_tracking", self.enable_tracking, 10)
        self.decider_activate = self.create_subscription(Empty, "activate_behavior", self.enable_behavior, 10)
        self.decider_clear = self.create_subscription(Empty, "clear_behavior", self.clear_behavior, 10)
        self.cube_sub = self.create_subscription(PoseStamped, "detected_cubes", self.get_cubes, 1)
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

    def clear_behavior(self, data: Empty):
        ## TODO: Find out why it doesn't work
        self.ros_world._logical_state_monitors.clear()
        self.ros_world._behaviors.clear()

    def enable_behavior(self, data: Empty):
        self.global_tracking = False
        decider_network = load_behavior_module("/home/ubb/Documents/Baxter_isaac/ROS2/src/isaac_sim/garment_sm.py").make_decider_network(self._robot)
        self.ros_world.add_decider_network(decider_network)

    def get_cubes(self, data: PoseStamped):
        self.cubes_pose[data.header.frame_id + "_block"] = (
            (data.pose.position.x, data.pose.position.y, data.pose.position.z),
            (data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z),
        )

    def create_cubes(self):
        for name, pose in self.cubes_pose.items():
            if name not in self.existing_cubes.keys():
                self.existing_cubes[name] = VisualCuboid(
                    f"/World/Obs/{name}",
                    position=pose[0],
                    orientation=pose[1],
                    size=np.array([0.04, 0.04, 0.04]),
                    color=np.array([1, 0, 0]),
                )
                # add_reference_to_stage(
                #     usd_path=self.rubiks_path,
                #     prim_path=f"/World/Obs/{name}",
                # )
                # self.existing_cubes[name] = XFormPrim(
                #     prim_path=f"/World/Obs/{name}",
                #     name=f"{name}",
                #     position=pose[0],
                #     orientation=pose[1],
                #     scale=np.array([0.0056, 0.0056, 0.0056]),
                # )  # w,x,y,z

            else:
                try:
                    self.existing_cubes[name].set_world_pose(*pose)
                except:
                    carb.log_warn(f"Object with name '{name}' has been ignored")


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

                # print(f"Main Loop: {time.time()-self.time}")
                # self.time = time.time()
                self.create_cubes()

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
        self._robot.initialize()
        self._robot.motion_policy.add_obstacle(self.cortex_table)
        # self._robot.motion_policy.visualize_collision_spheres()

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
        set_target_prims(primPath="/ActionGraph/PublishJointState", targetPrimPaths=[self._robot.prim_path])

        # Setting the /Franka target prim to Publish Transform Tree node
        set_target_prims(
            primPath="/ActionGraph/PublishTF",
            inputName="inputs:targetPrims",
            targetPrimPaths=[self._robot.prim_path],
        )

        self.simulation_app.update()

