from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.universal_robots.tasks import FollowTarget
from omni.isaac.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from omni.isaac.core import World
import carb
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats
import os, sys
from omni.isaac.core.utils.extensions import enable_extension
import omni
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core import World
from omni.isaac.urdf import _urdf
from omni.isaac.core.utils.stage import is_stage_loading
import omni.graph.core as og
from omni.isaac.core_nodes.scripts.utils import set_target_prims
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.robots.robot import Robot
from omni.isaac.universal_robots import UR10
from omni.isaac.core.objects.cuboid import VisualCuboid
from omni.isaac.surface_gripper import SurfaceGripper

enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

# Note that this is not the system level rclpy, but one compiled for omniverse
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from omni.isaac.core.utils.nucleus import get_assets_root_path, is_file
from omni.isaac.universal_robots.controllers import RMPFlowController


class Subscriber(Node):
    def __init__(self):
        super().__init__("tutorial_subscriber")
        self.setup_scene()
        self.setup_ik()
        self.ros_sub = self.create_subscription(Pose, "RightHand/pose", self.move_cube_callback, 10)
        self.timeline = omni.timeline.get_timeline_interface()



    def move_cube_callback(self, data:Pose):
        # callback function to set the cube position to a new one upon receiving a (empty) ROS2 message
        if self.ros_world.is_playing():
            self.stage.GetPrimAtPath("/World/TargetCube").GetAttribute("xformOp:translate").Set((data.position.z, data.position.x, data.position.y))
            print((data.position.x, data.position.y, data.position.z))
 
    def run_simulation(self):
        #Track any movements of the robot base

        self.timeline.play()
        while simulation_app.is_running():
            self.ros_world.step(render=True)
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.ros_world.is_playing():
                if self.ros_world.current_time_step_index == 0:
                    self.ros_world.reset()

                robot_base_translation,robot_base_orientation = self.ur3_robot.get_world_pose()
                self.lula_kinematics_solver.set_robot_base_pose(robot_base_translation,robot_base_orientation)


                pose, orientation = self.panda_hand_visualization.get_world_pose()
                actions, success = self.articulation_kinematics_solver.compute_inverse_kinematics(
                    target_position=pose,
                    target_orientation=orientation,
                )
                if success:
                    self.articulation_controller.apply_action(actions)
                else:
                    carb.log_warn("IK did not converge to a solution.  No action is being taken.")

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
        print(assets_root_path)
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
        import_config.import_inertia_tensor = True
        import_config.fix_base = True
        import_config.distance_scale = 1
        # Get the urdf file path

        
        self.file_name = "/home/ubb/Documents/isaac_sim/src/urdfs/ur_t42.urdf"
        # Finally import the robot
        result, self.ur3 = omni.kit.commands.execute( "URDFParseAndImportFile", urdf_path=self.file_name,
                                                      import_config=import_config)



        self.ur3_robot = UR10(prim_path=self.ur3, name="ur3", attach_gripper=False)
        gripper_usd = "/Projects/t42.usd"

        self.ros_world.scene.add(self.ur3_robot)
        # my_task = FollowTarget(name="follow_target_task", ur10_prim_path=self.ur3,
        #                        ur10_robot_name="ur3", target_name="target")

        # self.ros_world.add_task(my_task)

        ### DO NOT DELETE THIS !!! Will throw errors about undefined
        self.ros_world.reset()

        self.ur3_robot = self.ros_world.scene.get_object("ur3")

        print(self.ur3_robot)
        # self._controller = RMPFlowController(name="target_follower_controller", robot_articulation=self.ur3_robot)

        #self.create_action_graph()
    
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

    def setup_ik(self):
        self.target_name = "/World/TargetCube"
        self.mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
        self.kinematics_config_dir = os.path.join(self.mg_extension_path, "motion_policy_configs")
        self.stage = simulation_app.context.get_stage()

        self.lula_kinematics_solver = LulaKinematicsSolver(
            robot_description_path = self.kinematics_config_dir + "/ur10/rmpflow/ur10_robot_description.yaml",
            urdf_path = self.file_name
        )
        
        print("Valid frame names at which to compute kinematics:", self.lula_kinematics_solver.get_all_frame_names())

        self.end_effector_name = "tool0"
        self.articulation_kinematics_solver = ArticulationKinematicsSolver(self.ur3_robot,self.lula_kinematics_solver,self.end_effector_name)

        #Query the position of the "panda_hand" frame
        self.ee_position,ee_rot_mat = self.articulation_kinematics_solver.compute_end_effector_pose()
        self.ee_orientation = rot_matrices_to_quats(ee_rot_mat)

        #Create a cuboid to visualize where the "panda_hand" frame is according to the kinematics"
        self.panda_hand_visualization = VisualCuboid("/World/TargetCube",position=self.ee_position,orientation=self.ee_orientation,size=np.array([.05,.05,.05]),color=np.array([0,0,1]))

        self.articulation_controller = self.ur3_robot.get_articulation_controller()


if __name__ == "__main__":
    rclpy.init()
    subscriber = Subscriber()
    subscriber.run_simulation()