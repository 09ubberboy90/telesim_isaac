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
from omni.isaac.core.objects.cuboid import VisualCuboid, FixedCuboid
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.extensions import enable_extension, disable_extension, get_extension_path_from_name
from omni.isaac.core.utils.stage import is_stage_loading
from omni.isaac.core.utils.types import ArticulationAction 
from omni.isaac.core_nodes.scripts.utils import set_target_prims
from omni.isaac.motion_generation.lula import RmpFlow
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.usd import Gf

disable_extension("omni.isaac.ros_bridge")
enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

from threading import Thread, Event

import numpy as np

# Note that this is not the system level rclpy, but one compiled for omniverse
import rclpy
from geometry_msgs.msg import Pose
from omni.isaac.core.utils.nucleus import get_assets_root_path
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState

class Subscriber(Node):
    def __init__(self):
        super().__init__("isaac_sim_loop")
        self.tracking_enabled = defaultdict(lambda: True)
        self.setup_scene()
        self.setup_ik()
        self.movement_sub = self.create_subscription(Bool, "activate_tracking", self.enable_tracking, 10)
        self.ros_sub = self.create_subscription(Pose, "right_hand/pose", self.move_right_cube_callback, 10)
        self.trigger_sub = self.create_subscription(Bool, "right_hand/trigger", self.right_trigger_callback, 10)
        self.ros_sub_2 = self.create_subscription(Pose, "left_hand/pose", self.move_left_cube_callback, 10)
        self.trigger_sub_2 = self.create_subscription(Bool, "left_hand/trigger", self.left_trigger_callback, 10)
        self.robot_state_sub = self.create_subscription(JointState, "robot/joint_states", self.get_robot_state, 10)
        self.timeline = omni.timeline.get_timeline_interface()
        self.right_cube_pose = None
        self.left_cube_pose = None
        self.trigger = defaultdict(lambda: False)
        self.global_tracking = True
        self.robot_state = {}

    def enable_tracking(self, data: Bool):
        self.global_tracking = data.data
    
    def get_robot_state(self, data:JointState):
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

        self.right_cube_pose = (
            (-data.position.x, data.position.z, data.position.y),
            (-data.orientation.w, -data.orientation.z, data.orientation.x, -data.orientation.y),

        )
    def move_left_cube_callback(self, data: Pose):
        # Make sure to respect the parity (Levi-Civita symbol)
        if data.position.x > 3000:
            self.tracking_enabled["left"] = False
        else:
            self.tracking_enabled["left"] = True

        self.left_cube_pose = (
            (-data.position.x, data.position.z, data.position.y),
            # (data.orientation.w, -data.orientation.z, -data.orientation.x, -data.orientation.y),
            (-data.orientation.w, -data.orientation.z, data.orientation.x, -data.orientation.y),

        )

    def right_trigger_callback(self, data: Bool):
        self.trigger["right"] = data.data
    
    def left_trigger_callback(self, data: Bool):
        self.trigger["left"] = data.data

    def rclpy_spinner(self, event):
        while not event.is_set():
            rclpy.spin_once(self)

    def create_robot_articulation_state(self, controler:ArticulationMotionPolicy = None) -> ArticulationAction:
        position = []
        for name in self.articulation_controller._articulation_view.dof_names:

            position.append(self.robot_state[name])
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
                #Query the current obstacle position
                if self.global_tracking:
                    if self.tracking_enabled["right"]:

                        self.right_rmpflow.update_world()
                        pose, orientation = self.right_cube.get_world_pose()
                        self.right_rmpflow.set_end_effector_target(
                            target_position=pose,
                            target_orientation=orientation
                        )

                        actions = self.right_articulation_rmpflow.get_next_articulation_action()
                        self.articulation_controller.apply_action(actions)
                    else:
                        self.articulation_controller.apply_action(self.create_robot_articulation_state(self.right_articulation_rmpflow))
                    if self.tracking_enabled["left"]:
                        self.left_rmpflow.update_world()
                        pose, orientation = self.left_cube.get_world_pose()
                        self.left_rmpflow.set_end_effector_target(
                            target_position=pose,
                            target_orientation=orientation
                        )

                        actions = self.left_articulation_rmpflow.get_next_articulation_action()
                        self.articulation_controller.apply_action(actions)
                    else:
                        self.articulation_controller.apply_action(self.create_robot_articulation_state(self.left_articulation_rmpflow))
                else:
                    self.articulation_controller.apply_action(self.create_robot_articulation_state())
                    # self.articulation_controller.apply_action(self.create_robot_articulation_state(self.left_articulation_rmpflow))
                

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

        self.urdf_path = (
            "/home/ubb/Documents/baxter-stack/ROS2/src/baxter_common_ros2/baxter_description/urdf/baxter.urdf"
        )
        # Finally import the robot
        result, self.baxter = omni.kit.commands.execute(
            "URDFParseAndImportFile", urdf_path=self.urdf_path, import_config=import_config
        )

        self.baxter_robot = self.ros_world.scene.add(Robot(prim_path=self.baxter, name="baxter"))
        # my_task = FollowTarget(name="follow_target_task", ur10_prim_path=self.baxter,
        #                        ur10_robot_name="baxter", target_name="target")

        # self.ros_world.add_task(my_task)

        ### DO NOT DELETE THIS !!! Will throw errors about undefined
        self.ros_world.reset()

        self.stage = simulation_app.context.get_stage()
        self.table = self.stage.GetPrimAtPath("/Root/table_low_327")
        self.table.GetAttribute('xformOp:translate').Set(Gf.Vec3f(0.6,0.034,-0.975))
        self.table.GetAttribute('xformOp:rotateZYX').Set(Gf.Vec3f(0,0,90))
        self.table.GetAttribute('xformOp:scale').Set(Gf.Vec3f(0.7,0.6,1.15))

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

        rmp_config_dir = os.path.join("/home/ubb/Documents/baxter-stack/ROS2/src/baxter_joint_controller/rmpflow")

        #Initialize an RmpFlow object
        self.right_rmpflow = RmpFlow(
            robot_description_path = os.path.join(rmp_config_dir,"right_robot_descriptor.yaml"),
            urdf_path = self.urdf_path,
            rmpflow_config_path = os.path.join(rmp_config_dir,"baxter_rmpflow_common.yaml"),
            end_effector_frame_name = "right_gripper", #This frame name must be present in the URDF
            evaluations_per_frame = 5
        )
        self.left_rmpflow = RmpFlow(
            robot_description_path = os.path.join(rmp_config_dir,"left_robot_descriptor.yaml"),
            urdf_path = self.urdf_path,
            rmpflow_config_path = os.path.join(rmp_config_dir,"baxter_rmpflow_common.yaml"),
            end_effector_frame_name = "left_gripper", #This frame name must be present in the URDF
            evaluations_per_frame = 5
        )

        # Create a cuboid to visualize where the "panda_hand" frame is according to the kinematics"
        self.right_cube = VisualCuboid(
            "/World/right_cube",
            position=np.array([0.8, -0.1, 0.1]),
            orientation=np.array([0,-1, 0, 0]),
            size=np.array([0.05, 0.05, 0.05]),
            color=np.array([0, 0, 1]),
        )

        self.left_cube = VisualCuboid(
            "/World/left_cube",
            position=np.array([0.8, 0.1, 0.1]),
            orientation=np.array([0,-1,0,0]),
            size=np.array([0.05, 0.05, 0.05]),
            color=np.array([0, 0, 1]),
        )
        physics_dt = 1/60
        self.right_articulation_rmpflow = ArticulationMotionPolicy(self.baxter_robot,self.right_rmpflow, physics_dt)
        self.left_articulation_rmpflow = ArticulationMotionPolicy(self.baxter_robot,self.left_rmpflow, physics_dt)

        # self.right_rmpflow.visualize_collision_spheres()
        # self.left_rmpflow.visualize_collision_spheres()

        self.articulation_controller = self.baxter_robot.get_articulation_controller()
        self.articulation_controller.set_gains(kps=2000000, kds=400000 )
        # self.articulation_controller.set_gains(kps=4200, kds=840)
        # print(self.articulation_controller._articulation_view.dof_names)
        fake_table = FixedCuboid("/World/fake_table", position=np.array([0.6,0.0,-0.23]), size=np.array([0.64,1.16,0.07]))
        self.right_rmpflow.add_obstacle(fake_table)
        self.left_rmpflow.add_obstacle(fake_table)


if __name__ == "__main__":
    rclpy.init()
    subscriber = Subscriber()
    subscriber.run_simulation()
