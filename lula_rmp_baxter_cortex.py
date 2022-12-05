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
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core_nodes.scripts.utils import set_target_prims
from omni.isaac.cortex.cortex_utils import (add_cortex_attributes_to_objects,
                                            add_cortex_attributes_to_robot,
                                            make_core_objects, set_home_config,
                                            wrap_cortex_robot_or_die)
from omni.isaac.cortex.motion_commander import MotionCommander
from omni.isaac.motion_generation import (ArticulationMotionPolicy,
                                          MotionPolicyController,
                                          RmpFlowSmoothed)
from omni.isaac.motion_generation.lula import RmpFlow
from omni.usd import Gf

disable_extension("omni.isaac.ros_bridge")
enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()
from threading import Event, Thread

import numpy as np
import pyquaternion as pyq
# Note that this is not the system level rclpy, but one compiled for omniverse
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from omni.isaac.core.utils.nucleus import get_assets_root_path
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from baxter_robot import Baxter

from omni.isaac.cortex.df_behavior_watcher import DfBehaviorWatcher

class ContextTools:
    """ The tools passed in to a behavior when build_behavior(tools) is called.
    """

    def __init__(self, world, objects, obstacles, robot, commander):
        self.world = world  # The World singleton.
        self.objects = objects  # The objects under /cortex/belief/objects as core API objects.
        self.obstacles = obstacles  # Those objects marked as obstacles.
        self.robot = robot  # The belief robot.
        self.commander = commander  # The motion commander.

    def enable_obstacles(self):
        """ Ensures the obstacles are enabled. This can be called by a behavior on construction. To
        reset any previous obstacle suppression.
        """
        for _, obs in self.obstacles.items():
            self.commander.enable_obstacle(obs)


class Subscriber(Node):
    def __init__(self):
        super().__init__("isaac_sim_loop")
        self.tracking_enabled = defaultdict(lambda: True)
        self.movement_sub = self.create_subscription(Bool, "activate_tracking", self.enable_tracking, 10)
        self.ros_sub = self.create_subscription(Pose, "right_hand/pose", self.move_right_cube_callback, 10)
        self.trigger_sub = self.create_subscription(Bool, "right_hand/trigger", self.right_trigger_callback, 10)
        self.ros_sub_2 = self.create_subscription(Pose, "left_hand/pose", self.move_left_cube_callback, 10)
        self.trigger_sub_2 = self.create_subscription(Bool, "left_hand/trigger", self.left_trigger_callback, 10)
        self.robot_state_sub = self.create_subscription(JointState, "robot/joint_states", self.get_robot_state, 10)
        self.cube_sub = self.create_subscription(PoseArray, "detected_cubes", self.get_cubes, 10)
        self.timeline = omni.timeline.get_timeline_interface()
        self.right_cube_pose = None
        self.left_cube_pose = None
        self.trigger = defaultdict(lambda: False)
        self.global_tracking = True
        self.robot_state = {}
        self.cubes_pose = {}
        self.existing_cubes = {}
        self.rubiks_path = "omniverse://127.0.0.1/Isaac/Props/Rubiks_Cube/rubiks_cube.usd"
        self.nvidia_cube = "omniverse://127.0.0.1/Isaac/Props/Blocks/nvidia_cube.usd"
        self.setup_scene()
        self.setup_ik()
        self.setup_cortex()
        self.create_action_graph()


    def enable_tracking(self, data: Bool):
        self.global_tracking = data.data

    def get_cubes(self, data:PoseArray):
        for pose in data.poses:
            self.cubes_pose[data.header.frame_id+"_block"] = (
                (pose.position.x, pose.position.y, pose.position.z),
                (pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z),
            )

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
                    prim_path=f"/World/{name}",
                )
                self.existing_cubes[name] = XFormPrim(
                    prim_path=f"/World/{name}",
                    name="cube1",
                    position=pose[0],
                    orientation=pose[1],
                    scale=np.array([0.0056,0.0056,0.0056]),
                )  # w,x,y,z

            else:
                try:
                    self.existing_cubes[name].set_world_pose(*pose)
                except:
                    carb.log_warn(f"Object with name '{name}' has been ignored")
                    # self.existing_cubes.pop(name, None)



    def get_robot_state(self, data: JointState):
        for idx, el in enumerate(data.name):
            self.robot_state[el] = data.position[idx]
        ## Duplicate gripper as they're set to mimic
        try:
            self.robot_state["l_gripper_r_finger_joint"] = self.robot_state["l_gripper_l_finger_joint"]
            self.robot_state["r_gripper_r_finger_joint"] = self.robot_state["r_gripper_l_finger_joint"]
        except:
            pass

    def move_right_cube_callback(self, data: Pose):
        # Make sure to respect the parity (Levi-Civita symbol)
        if data.position.x > 3000:
            self.tracking_enabled["right"] = False
        else:
            self.tracking_enabled["right"] = True

        mul_rot = pyq.Quaternion(w=0.0, x=0.0, y=0.707, z=0.707)  ## Handles axis correction
        offset_rot = pyq.Quaternion(w=0.707, x=0.707, y=0.0, z=0.0)  ## handles sideway instead of up

        q1 = pyq.Quaternion(x=data.orientation.x, y=data.orientation.y, z=data.orientation.z, w=data.orientation.w)
        new_quat = mul_rot * q1
        new_quat *= offset_rot

        self.right_cube_pose = (
            (-data.position.x, data.position.z, data.position.y),
            (new_quat.w, new_quat.x, new_quat.y, new_quat.z),
        )

    def move_left_cube_callback(self, data: Pose):
        # Make sure to respect the parity (Levi-Civita symbol)
        if data.position.x > 3000:
            self.tracking_enabled["left"] = False
        else:
            self.tracking_enabled["left"] = True

        mul_rot = pyq.Quaternion(w=0.0, x=0.0, y=0.707, z=0.707)  ## Handles axis correction
        offset_rot = pyq.Quaternion(w=0.707, x=0.707, y=0.0, z=0.0)  ## handles sideway instead of up

        q1 = pyq.Quaternion(x=data.orientation.x, y=data.orientation.y, z=data.orientation.z, w=data.orientation.w)
        new_quat = mul_rot * q1
        new_quat *= offset_rot

        self.left_cube_pose = (
            (-data.position.x, data.position.z, data.position.y),
            (new_quat.w, new_quat.x, new_quat.y, new_quat.z),
        )

    def right_trigger_callback(self, data: Bool):
        self.trigger["right"] = data.data

    def left_trigger_callback(self, data: Bool):
        self.trigger["left"] = data.data

    def rclpy_spinner(self, event):
        while not event.is_set():
            rclpy.spin_once(self)

    def create_robot_articulation_state(self, controler: ArticulationMotionPolicy = None) -> ArticulationAction:
        position = []
        for name in self.articulation_controller._articulation_view.dof_names:
            try:
                position.append(self.robot_state[name])
            except:
                pass
        if len(position) == 0:
            return ArticulationAction(joint_positions=[0.0]*len(self.articulation_controller._articulation_view.dof_names))
        if controler is not None:
            return ArticulationAction(joint_positions=controler._active_joints_view.map_to_articulation_order(position))
        else:
            return ArticulationAction(joint_positions=position)

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

                self.df_behavior_watcher.check_reload(self.context_tools)

                try:
                    self.df_behavior_watcher.tick_behavior()
                except Exception as e:
                    print("\nProblem ticking behavior.")
                    import traceback

                    traceback.print_exc()

                self.create_cubes()

                # if self.left_cube_pose is not None:
                #     self.left_cube.set_world_pose(*self.left_cube_pose)
                # if self.right_cube_pose is not None:
                #     self.right_cube.set_world_pose(*self.right_cube_pose)

                # if self.trigger["right"]:
                #     self.baxter_robot.right_gripper.set_positions(self.baxter_robot.right_gripper.closed_position)
                # else:
                #     self.baxter_robot.right_gripper.set_positions(self.baxter_robot.right_gripper.open_position)

                # if self.trigger["left"]:
                #     self.baxter_robot.left_gripper.set_positions(self.baxter_robot.left_gripper.closed_position)
                # else:
                #     self.baxter_robot.left_gripper.set_positions(self.baxter_robot.left_gripper.open_position)
                # # Query the current obstacle position

                ## Disable the gripper as they are handled by the robot controller itself
                action = self.context_tools.commander.get_action()
                action.joint_positions[15:] = [None]*4
                self.baxter_robot.get_articulation_controller().apply_action(action)
                action = self.left_commander.get_action()
                action.joint_positions[15:] = [None]*4
                self.baxter_robot.get_articulation_controller().apply_action(action)

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
        self.ros_world = World(stage_units_in_meters=1.0)
        # print("<enabling cortex ROS-based extensions>")
        # ext_manager = omni.kit.app.get_app().get_extension_manager()

        # ext_manager.set_extension_enabled_immediate("omni.isaac.cortex", True)
        self.urdf_path = (
            "/home/ubb/Documents/Baxter_isaac/ROS1/src/baxter_joint_controller/urdf/baxter.urdf"
        )

        self.baxter_robot = Baxter(urdf_path= self.urdf_path, name="robot", attach_gripper=True)
        self.ros_world.scene.add(self.baxter_robot)

        self.stage = simulation_app.context.get_stage()
        self.table = self.stage.GetPrimAtPath("/Root/table_low_327")
        self.table.GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.6, 0.0, -0.975))
        self.table.GetAttribute("xformOp:rotateZYX").Set(Gf.Vec3f(0, 0, 90))
        self.table.GetAttribute("xformOp:scale").Set(Gf.Vec3f(0.7, 0.6, 1.15))

        # Create a cuboid to visualize where the "panda_hand" frame is according to the kinematics"
        self.right_cube = VisualCuboid(
            "/World/right_cube",
            position=np.array([0.8, -0.1, 0.1]),
            orientation=np.array([0, -1, 0, 0]),
            size=0.005,
            color=np.array([0, 0, 1]),
        )
        name = ["red_block", "green_block", "blue_block", "yellow_block", ]
        color = [[1,0,0],[0,1,0],[0,0,1],[0,1,1],]
        for i in range(4):
            self.existing_cubes[name[i]] = DynamicCuboid(
            f"/cortex/belief/objects/{name[i]}",
            position=np.array([0.5+((i+1)%2)/10, 0.0, 0.25]),
            orientation=np.array([1, 0, 0, 0]),
            size=0.04,
            color=np.array(color[i]),
        )


        self.left_cube = VisualCuboid(
            "/World/left_cube",
            position=np.array([0.8, 0.1, 0.1]),
            orientation=np.array([0, -1, 0, 0]),
            size=0.005,
            color=np.array([0, 0, 1]),
        )
        
        ### DO NOT DELETE THIS !!! Will throw errors about undefined
        self.ros_world.reset()
        simulation_app.update()
        self.ros_world.step()  # Step physics to trigger cortex_sim extension self.baxter_robot to be created.
        self.baxter_robot.initialize()

        # self.articulation_controller.set_gains(kps=134217, kds=67100)
        
    def setup_cortex(self):
        add_cortex_attributes_to_robot(self.baxter_robot, is_suppressed=False, adaptive_cycle_dt=self.ros_world.get_physics_dt())
        
        self.ros_world.step()  # Trigger extensions to configure their robots
        objects, obstacles = make_core_objects("belief")
        add_cortex_attributes_to_objects(objects)
        for name, obj in objects.items():
            self.ros_world.scene.add(obj)

        print("obstacles:")
        for i, name in enumerate(obstacles):
            print("%d) obs: %s" % (i, name))

        # Reset then step the self.ros_world to set all initial configurations of the self.baxter_robot and corresponding
        # child USD elements (e.g. cortex's eff frame which aligns with the RMPflow policy's
        # end-effector).
        self.ros_world.reset()
        self.ros_world.step(render=False)
        # self.articulation_controller.set_gains(kps=536868, kds=4294400)

        self.right_commander = MotionCommander(self.baxter_robot, self.right_motion_policy_controller, self.right_cube)
        self.left_commander = MotionCommander(self.baxter_robot, self.left_motion_policy_controller, self.left_cube)
        ## Once again as reset reset the gains
        self.context_tools = ContextTools(self.ros_world, objects, obstacles, self.baxter_robot, self.right_commander)
        self.df_behavior_watcher = DfBehaviorWatcher(verbose=True,)
        self.stage = simulation_app.context.get_stage()
        self.baxter_robot.set_joints_default_state([0.0, -0.7441, 1.1358, -0.6647, -0.6604, 0.3762, -0.6377, 0.9195, 1.0242, -0.3, 0.5024, 1.3634, 1.3482, -2.7965, 2.9428, 0.0208, -0.0208, 0.0208, -0.0208])

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
        set_target_prims(primPath="/ActionGraph/PublishJointState", targetPrimPaths=[self.baxter_robot.prim_path])

        # Setting the /Franka target prim to Publish Transform Tree node
        set_target_prims(
            primPath="/ActionGraph/PublishTF", inputName="inputs:targetPrims", targetPrimPaths=[self.baxter_robot.prim_path]
        )

        simulation_app.update()

    def setup_ik(self):

        self.articulation_controller = self.baxter_robot.get_articulation_controller()

        self.mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
        self.kinematics_config_dir = os.path.join(self.mg_extension_path, "motion_policy_configs")

        rmp_config_dir = os.path.join("/home/ubb/Documents/baxter-stack/ROS2/src/baxter_joint_controller/rmpflow")
        # Initialize an RmpFlow object
        self.right_rmpflow = RmpFlow(
            robot_description_path=os.path.join(rmp_config_dir, "right_robot_descriptor.yaml"),
            urdf_path=self.urdf_path,
            rmpflow_config_path=os.path.join(rmp_config_dir, "baxter_rmpflow_common.yaml"),
            end_effector_frame_name="right_gripper_cube_offset",  # This frame name must be present in the URDF
            evaluations_per_frame=10,
        )
        self.left_rmpflow = RmpFlow(
            robot_description_path=os.path.join(rmp_config_dir, "left_robot_descriptor.yaml"),
            urdf_path=self.urdf_path,
            rmpflow_config_path=os.path.join(rmp_config_dir, "baxter_rmpflow_common.yaml"),
            end_effector_frame_name="left_gripper",  # This frame name must be present in the URDF
            evaluations_per_frame=5,
        )
        physics_dt = 1 / 60
        self.right_articulation_rmpflow = ArticulationMotionPolicy(self.baxter_robot, self.right_rmpflow, physics_dt)
        self.left_articulation_rmpflow = ArticulationMotionPolicy(self.baxter_robot, self.left_rmpflow, physics_dt)

        # self.right_rmpflow.visualize_collision_spheres()
        # self.left_rmpflow.visualize_collision_spheres()
        self.left_motion_policy_controller = MotionPolicyController(
            name="left_rmpflow_controller",
            articulation_motion_policy=ArticulationMotionPolicy(self.baxter_robot, self.left_rmpflow, physics_dt)
        )
        self.right_motion_policy_controller = MotionPolicyController(
            name="right_rmpflow_controller",
            articulation_motion_policy=ArticulationMotionPolicy(self.baxter_robot, self.right_rmpflow, physics_dt)
        )

        self.articulation_controller = self.baxter_robot.get_articulation_controller()
        ## Do not touch the grippers
        self.articulation_controller.set_gains(kps=[536868]*(self.baxter_robot.num_dof-4)+[10000000]*4, kds=[68710400]*(self.baxter_robot.num_dof-4)+[2000000]*4)

        # print(self.articulation_controller._articulation_view.dof_names)
        # fake_table = FixedCuboid(
        #     "/World/fake_table", position=np.array([0.6, 0.0, -0.25]), size=np.array([0.64, 1.16, 0.07])
        # )
        # self.right_rmpflow.add_obstacle(fake_table)
        # self.left_rmpflow.add_obstacle(fake_table)

if __name__ == "__main__":
    rclpy.init()
    subscriber = Subscriber()
    subscriber.run_simulation()
