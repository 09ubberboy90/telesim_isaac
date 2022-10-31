from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import sys

import carb
import omni
import omni.graph.core as og
from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationGripper
from omni.isaac.core.objects.cuboid import VisualCuboid, FixedCuboid
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.extensions import disable_extension, enable_extension
from omni.isaac.core.utils.stage import is_stage_loading
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core_nodes.scripts.utils import set_target_prims
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.motion_generation.lula import RmpFlow
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.usd import Gf

disable_extension("omni.isaac.ros_bridge")
enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

from threading import Event, Thread

import numpy as np

# Note that this is not the system level rclpy, but one compiled for omniverse
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from omni.isaac.core.utils.nucleus import get_assets_root_path
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

import pyquaternion as pyq

class Subscriber(Node):
    def __init__(self):
        super().__init__("isaac_sim_loop")
        self.tracking_enabled = True
        self.setup_scene()
        self.setup_ik()
        self.movement_sub = self.create_subscription(Bool, "activate_tracking", self.enable_tracking, 10)
        self.ros_sub = self.create_subscription(Pose, "right_hand/pose", self.move_cube_callback, 10)
        self.trigger_sub = self.create_subscription(Bool, "right_hand/trigger", self.trigger_callback, 10)
        self.robot_state_sub = self.create_subscription(JointState, "robot/joint_states", self.get_robot_state, 10)
        self.cube_sub = self.create_subscription(PoseArray, "detected_cubes", self.get_cubes, 10)

        self.timeline = omni.timeline.get_timeline_interface()
        self.cube_pose = None
        self.trigger = False
        self.robot_state = {}
        self.cubes_pose = {}
        self.existing_cubes = {}

    def enable_tracking(self, data: Bool):
        self.tracking_enabled = data.data

    def get_cubes(self, data:PoseArray):
        for pose in data.poses:
            self.cubes_pose[data.header.frame_id] = (
                (pose.position.x, pose.position.y, pose.position.z),
                (pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z),
            )

    def create_cubes(self):
        for name, pose in self.cubes_pose.items():
            if name not in self.existing_cubes.keys():
                self.existing_cubes[name] = VisualCuboid(
                    f"/World/{name}",
                    position=pose[0],
                    orientation=pose[1],
                    size=np.array([0.05, 0.05, 0.05]),
                    color=np.array([1, 0, 0]),
                )
            else:
                self.existing_cubes[name].set_world_pose(*pose)



    def get_robot_state(self, data: JointState):
        for idx, el in enumerate(data.name):
            self.robot_state[el] = data.position[idx]
        ## Duplicate gripper as they're set to mimic
        try:
            self.robot_state["l_gripper_r_finger_joint"] = self.robot_state["l_gripper_l_finger_joint"]
            self.robot_state["r_gripper_r_finger_joint"] = self.robot_state["r_gripper_l_finger_joint"]
        except:
            pass


    def move_cube_callback(self, data: Pose):
        # Make sure to respect the parity (Levi-Civita symbol)
        if data.position.x > 3000:
            self.tracking_enabled = False
        else:
            self.tracking_enabled = True

        mul_rot = pyq.Quaternion(w=0.0, x=0.0, y=0.707, z=0.707)  ## Handles axis correction
        offset_rot = pyq.Quaternion(w=0.707, x=0.707, y=0.0, z=0.0)  ## handles sideway instead of up

        q1 = pyq.Quaternion(x=data.orientation.x, y=data.orientation.y, z=data.orientation.z, w=data.orientation.w)
        new_quat = mul_rot * q1
        new_quat *= offset_rot

        self.cube_pose = (
            (-data.position.x, data.position.z, data.position.y),
            (new_quat.w, new_quat.x, new_quat.y, new_quat.z),
        )


    def trigger_callback(self, data: Bool):
        self.trigger = data.data

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
            return ArticulationAction(joint_positions=position)
        if controler is not None:
            return ArticulationAction(joint_positions=controler._active_joints_view.map_to_articulation_order(position))
        else:
            return ArticulationAction(joint_positions=position)

    def run_simulation(self):
        # Track any movements of the robot base
        counter = 0
        event = Event()
        rclpy_thread = Thread(target=self.rclpy_spinner, args=[event])
        rclpy_thread.start()

        self.timeline.play()
        robot_base_translation, robot_base_orientation = self.ur_t42_robot.get_world_pose()
        while simulation_app.is_running():
            self.ros_world.step(render=True)
            # rclpy.spin_once(self, timeout_sec=0.0)
            if self.ros_world.is_playing():
                if self.ros_world.current_time_step_index == 0:
                    self.ros_world.reset()
                self.create_cubes()

                if self.cube_pose is not None:
                    self.cube.set_world_pose(*self.cube_pose)

                if self.trigger:
                    self.gripper.set_positions(self.gripper.closed_position)
                else:
                    self.gripper.set_positions(self.gripper.open_position)
                self.rmpflow.update_world()
                if self.tracking_enabled:

                    pose, orientation = self.cube.get_world_pose()
                    self.rmpflow.set_end_effector_target(
                        target_position=pose,
                        target_orientation=orientation,
                    )
                    actions = self.articulation_rmpflow.get_next_articulation_action()
                    self.articulation_controller.apply_action(actions)
                else:
                    self.articulation_controller.apply_action(self.create_robot_articulation_state())

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
        import_config.self_collision = True
        # Get the urdf file path

        self.urdf_path = "/home/ubb/Documents/UR_T42/ur_isaac/urdfs/ur_t42.urdf"
        # Finally import the robot
        result, self.ur_t42 = omni.kit.commands.execute(
            "URDFParseAndImportFile", urdf_path=self.urdf_path, import_config=import_config
        )

        self.ur_t42_robot = self.ros_world.scene.add(Robot(prim_path=self.ur_t42, name="ur_t42"))
        # my_task = FollowTarget(name="follow_target_task", ur10_prim_path=self.ur_t42,
        #                        ur10_robot_name="ur_t42", target_name="target")

        # self.ros_world.add_task(my_task)

        ### DO NOT DELETE THIS !!! Will throw errors about undefined
        self.ros_world.reset()

        self.stage = simulation_app.context.get_stage()
        table = self.stage.GetPrimAtPath("/Root/table_low_327")
        table.GetAttribute('xformOp:translate').Set(Gf.Vec3f(0.36,0.0,-1.01))
        table.GetAttribute('xformOp:rotateZYX').Set(Gf.Vec3f(0,0,90))
        table.GetAttribute('xformOp:scale').Set(Gf.Vec3f(1,0.8,1.15))


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
        set_target_prims(primPath="/ActionGraph/SubscribeJointState", targetPrimPaths=[self.ur_t42])

        # Setting the /Franka target prim to Publish JointState node
        set_target_prims(primPath="/ActionGraph/PublishJointState", targetPrimPaths=[self.ur_t42])

        # Setting the /Franka target prim to Publish Transform Tree node
        set_target_prims(
            primPath="/ActionGraph/PublishTF", inputName="inputs:targetPrims", targetPrimPaths=[self.ur_t42]
        )

        simulation_app.update()

    def setup_ik(self):

        self.gripper = ArticulationGripper(
            gripper_dof_names=[
                "swivel_2_to_finger_2_1",
                "finger_2_1_to_finger_2_2",
                "swivel_1_to_finger_1_1",
                "finger_1_1_to_finger_1_2",
            ],
            gripper_closed_position=[1.57, 0.785, 1.57, 0.785],
            gripper_open_position=[0.0, 0.0, 0.0, 0.0],
        )
        self.gripper.initialize(self.ur_t42, self.ur_t42_robot.get_articulation_controller())

        self.rmp_config_dir = "/home/ubb/Documents/UR_T42/ur_isaac/config"

        self.rmpflow = RmpFlow(
            robot_description_path=os.path.join(self.rmp_config_dir, "ur3_t42_robot_description.yaml"),
            urdf_path=self.urdf_path,
            rmpflow_config_path=os.path.join(self.rmp_config_dir, "ur3_t42_rmpflow_config.yaml"),
            end_effector_frame_name="t42_base_link",  # This frame name must be present in the URDF
            evaluations_per_frame=5,
        )

        # Uncomment this line to visualize the collision spheres in the robot_description YAML file
        # self.rmpflow.visualize_collision_spheres()
        # rmpflow.set_ignore_state_updates(True)

        physics_dt = 1 / 60.0
        self.articulation_rmpflow = ArticulationMotionPolicy(self.ur_t42_robot, self.rmpflow, physics_dt)

        self.articulation_controller = self.ur_t42_robot.get_articulation_controller()

        self.articulation_controller.set_gains(kps=1000, kds=10000)

        # Create a cuboid to visualize where the "panda_hand" frame is according to the kinematics"
        self.cube = VisualCuboid(
            "/World/cube",
            position=np.array([0.8, -0.1, 0.1]),
            orientation=np.array([0, -1, 0, 0]),
            size=np.array([0.05, 0.05, 0.05]),
            color=np.array([0, 0, 1]),
        )
        fake_table = FixedCuboid("/World/fake_table", position=np.array([0.6,0.4,-0.27]), size=np.array([0.95,1.6,0.07]))
        self.rmpflow.add_obstacle(fake_table)

        # print(self.articulation_controller._articulation_view.dof_names)


if __name__ == "__main__":
    rclpy.init()
    subscriber = Subscriber()
    subscriber.run_simulation()
