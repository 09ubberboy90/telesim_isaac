from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import sys
from collections import defaultdict

import carb
import omni
import omni.graph.core as og
from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationGripper
from omni.isaac.core.objects.cuboid import VisualCuboid
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.extensions import enable_extension, disable_extension, get_extension_path_from_name
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.core.utils.prims import delete_prim
from omni.isaac.core_nodes.scripts.utils import set_target_prims
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

disable_extension("omni.isaac.ros_bridge")
enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

from threading import Thread, Event

import numpy as np

# Note that this is not the system level rclpy, but one compiled for omniverse
import rclpy
from geometry_msgs.msg import Pose
from omni.isaac.core.utils.nucleus import get_assets_root_path, is_file
from omni.isaac.universal_robots.controllers import RMPFlowController
from rclpy.node import Node
from std_msgs.msg import Bool


class Subscriber(Node):
    def __init__(self):
        super().__init__("isaac_sim_loop")
        self.tracking_enabled = True
        self.setup_scene()
        self.setup_ik()
        self.movement_sub = self.create_subscription(Bool, "activate_tracking", self.enable_tracking, 10)
        # self.ros_sub = self.create_subscription(Pose, "right_hand/pose", self.move_right_cube_callback, 10)
        # self.trigger_sub = self.create_subscription(Bool, "right_hand/trigger", self.right_trigger_callback, 10)
        self.ros_sub_2 = self.create_subscription(Pose, "left_hand/pose", self.move_left_cube_callback, 10)
        self.trigger_sub_2 = self.create_subscription(Bool, "left_hand/trigger", self.left_trigger_callback, 10)
        self.timeline = omni.timeline.get_timeline_interface()
        self.right_cube_pose = None
        self.left_cube_pose = None
        self.trigger = defaultdict(lambda: False)

    def enable_tracking(self, data: Bool):
        self.tracking_enabled = data.data

    def move_right_cube_callback(self, data: Pose):
        self.right_cube_pose = (
            (-data.position.z, -data.position.x, data.position.y),
            (data.orientation.w, data.orientation.x, data.orientation.y, data.orientation.z),
        )
    def move_left_cube_callback(self, data: Pose):
        self.left_cube_pose = (
            (-data.position.z, -data.position.x, data.position.y),
            (data.orientation.w, data.orientation.x, data.orientation.y, data.orientation.z),
        )

    def right_trigger_callback(self, data: Bool):
        self.trigger["right"] = data.data
    
    def left_trigger_callback(self, data: Bool):
        self.trigger["left"] = data.data

    def rclpy_spinner(self, event):
        while not event.is_set():
            rclpy.spin_once(self)

    def run_simulation(self):
        # Track any movements of the robot base
        counter = 0
        event = Event()
        rclpy_thread = Thread(target=self.rclpy_spinner, args=[event])
        rclpy_thread.start()

        self.timeline.play()
        robot_base_translation, robot_base_orientation = self.baxter_robot.get_world_pose()
        self.right_kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        self.left_kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        while simulation_app.is_running():
            self.ros_world.step(render=True)
            # rclpy.spin_once(self, timeout_sec=0.0)
            if self.ros_world.is_playing():
                if self.ros_world.current_time_step_index == 0:
                    self.ros_world.reset()

                if self.left_cube_pose is not None:
                    self.left_cube.set_world_pose(*self.left_cube_pose)
                if self.right_cube_pose is not None:
                    self.right_cube.set_world_pose(*self.right_cube_pose)

                if self.trigger["right"]:
                    self.right_gripper.set_positions(self.right_gripper.closed_position)
                else:
                    self.right_gripper.set_positions(self.right_gripper.open_position)

                if self.trigger["left"]:
                    self.left_gripper.set_positions(self.left_gripper.closed_position)
                else:
                    self.left_gripper.set_positions(self.left_gripper.open_position)

                if self.tracking_enabled:

                    pose, orientation = self.left_cube.get_world_pose()
                    left_actions, left_success = self.left_articulation_kinematics_solver.compute_inverse_kinematics(
                        target_position=pose,
                        target_orientation=orientation,
                    )
                    if left_success:
                        self.articulation_controller.apply_action(left_actions)
                    else:
                        if counter % 15 == 0:  ## Roughly 500 ms
                            carb.log_warn("Left IK did not converge to a solution.  No action is being taken.")
                        counter += 1

                    pose, orientation = self.right_cube.get_world_pose()
                    right_actions, right_success = self.right_articulation_kinematics_solver.compute_inverse_kinematics(
                        target_position=pose,
                        target_orientation=orientation,
                    )
                    if right_success:
                        self.articulation_controller.apply_action(right_actions)
                    else:
                        if counter % 15 == 0:  ## Roughly 500 ms
                            carb.log_warn("Right IK did not converge to a solution.  No action is being taken.")
                        counter += 1

                    # action = [right_actions.joint_positions[i] if not np.isnan(right_actions.joint_positions[i]) else left_actions.joint_positions[i] for i in range(right_actions.get_length()) ]
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
        # add a cube in the world
        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = False
        import_config.fix_base = True
        import_config.distance_scale = 1
        # Get the urdf file path

        self.file_name = (
            "/home/ubb/Documents/baxter-stack/ROS2/src/baxter_common_ros2/baxter_description/urdf/baxter.urdf"
        )
        # Finally import the robot
        result, self.baxter = omni.kit.commands.execute(
            "URDFParseAndImportFile", urdf_path=self.file_name, import_config=import_config
        )

        self.baxter_robot = self.ros_world.scene.add(Robot(prim_path=self.baxter, name="baxter"))
        # my_task = FollowTarget(name="follow_target_task", ur10_prim_path=self.baxter,
        #                        ur10_robot_name="baxter", target_name="target")

        # self.ros_world.add_task(my_task)

        ### DO NOT DELETE THIS !!! Will throw errors about undefined
        self.ros_world.reset()
        self.stage = simulation_app.context.get_stage()
        table = self.stage.GetPrimAtPath("/Root/table_low_327")
        print(table)
        delete_prim("/Root/table_low_327")
        

        self.franka_table_usd = assets_root_path + "/Isaac/Environments/Simple_Room/Props/table_low.usd"
        add_reference_to_stage(
            usd_path=self.franka_table_usd,
            prim_path="/World/table",
        )
        self.table = XFormPrim(
            prim_path="/World/table",
            name="table",
            position=np.array([0.6,0.0,-1.0]),
            orientation=np.array([0.7073883, 0, 0, 0.7068252]),
            scale=np.array([0.7,0.6,1.15]),
        )  # w,x,y,z

        # self._controller = RMPFlowController(name="target_follower_controller", robot_articulation=self.baxter_robot)

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
                        ("SubscribeJointState.inputs:topicName", "robot/joint_states"),
                        ("PublishTF.inputs:topicName", "tf_sim"),
                    ],
                },
            )
        except Exception as e:
            print(e)

        # Setting the /Franka target prim to Subscribe JointState node
        set_target_prims(primPath="/ActionGraph/SubscribeJointState", targetPrimPaths=[self.baxter])

        # Setting the /Franka target prim to Publish JointState node
        set_target_prims(primPath="/ActionGraph/PublishJointState", targetPrimPaths=[self.baxter])

        # Setting the /Franka target prim to Publish Transform Tree node
        set_target_prims(
            primPath="/ActionGraph/PublishTF", inputName="inputs:targetPrims", targetPrimPaths=[self.baxter]
        )

        simulation_app.update()

    def setup_ik(self):

        self.left_gripper = ArticulationGripper(
            gripper_dof_names=["l_gripper_l_finger_joint", "l_gripper_r_finger_joint"],
            gripper_closed_position=[0.0, 0.0],
            gripper_open_position=[0.020833, -0.020833],
        )
        self.left_gripper.initialize(self.baxter, self.baxter_robot.get_articulation_controller())

        self.right_gripper = ArticulationGripper(
            gripper_dof_names=["r_gripper_l_finger_joint", "r_gripper_r_finger_joint"],
            gripper_closed_position=[0.0, 0.0],
            gripper_open_position=[0.020833, -0.020833],
        )
        self.right_gripper.initialize(self.baxter, self.baxter_robot.get_articulation_controller())

        self.mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
        self.kinematics_config_dir = os.path.join(self.mg_extension_path, "motion_policy_configs")

        self.right_kinematics_solver = LulaKinematicsSolver(
            robot_description_path="/home/ubb/Documents/baxter-stack/ROS2/src/baxter_joint_controller/rmpflow/right_robot_descriptor.yaml",
            urdf_path=self.file_name,
        )
        self.left_kinematics_solver = LulaKinematicsSolver(
            robot_description_path="/home/ubb/Documents/baxter-stack/ROS2/src/baxter_joint_controller/rmpflow/left_robot_descriptor.yaml",
            urdf_path=self.file_name,
        )

        self.end_effector_name = "right_gripper"
        self.right_articulation_kinematics_solver = ArticulationKinematicsSolver(
            self.baxter_robot, self.right_kinematics_solver, self.end_effector_name
        )
        self.end_effector_name = "left_gripper"
        self.left_articulation_kinematics_solver = ArticulationKinematicsSolver(
            self.baxter_robot, self.left_kinematics_solver, self.end_effector_name
        )
        # Query the position of the "panda_hand" frame
        ee_position, ee_rot_mat = self.right_articulation_kinematics_solver.compute_end_effector_pose()
        ee_orientation = rot_matrices_to_quats(ee_rot_mat)

        # Create a cuboid to visualize where the "panda_hand" frame is according to the kinematics"
        self.right_cube = VisualCuboid(
            "/World/right_cube",
            position=ee_position,
            orientation=ee_orientation,
            size=np.array([0.05, 0.05, 0.05]),
            color=np.array([0, 0, 1]),
        )

        ee_position, ee_rot_mat = self.left_articulation_kinematics_solver.compute_end_effector_pose()
        ee_orientation = rot_matrices_to_quats(ee_rot_mat)

        self.left_cube = VisualCuboid(
            "/World/left_cube",
            position=ee_position,
            orientation=ee_orientation,
            size=np.array([0.05, 0.05, 0.05]),
            color=np.array([0, 0, 1]),
        )

        self.articulation_controller = self.baxter_robot.get_articulation_controller()
        # print(self.articulation_controller._articulation_view.dof_names)


if __name__ == "__main__":
    rclpy.init()
    subscriber = Subscriber()
    subscriber.run_simulation()
