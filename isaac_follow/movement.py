from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.franka.tasks import FollowTarget
from omni.isaac.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from omni.isaac.core import World
import carb
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.objects.cuboid import VisualCuboid
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats
import os
from omni.isaac.core.utils.extensions import enable_extension
import omni

enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

# Note that this is not the system level rclpy, but one compiled for omniverse
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose


class Subscriber(Node):
    def __init__(self):
        super().__init__("tutorial_subscriber")
        self.my_world = World(stage_units_in_meters=1.0)
        self.my_task = FollowTarget(name="follow_target_task")
        self.my_world.add_task(self.my_task)
        self.my_world.reset()
        self.task_params = self.my_world.get_task("follow_target_task").get_params()
        self.franka_name = self.task_params["robot_name"]["value"]
        self.target_name = self.task_params["target_name"]["value"]
        self.my_franka = self.my_world.scene.get_object(self.franka_name)
        self.mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
        self.kinematics_config_dir = os.path.join(self.mg_extension_path, "motion_policy_configs")
        self.stage = simulation_app.context.get_stage()

        self.lula_kinematics_solver = LulaKinematicsSolver(
            robot_description_path = self.kinematics_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path = self.kinematics_config_dir + "/franka/lula_franka_gen.urdf"
        )
        
        print("Valid frame names at which to compute kinematics:", self.lula_kinematics_solver.get_all_frame_names())

        self.end_effector_name = "panda_hand"
        self.articulation_kinematics_solver = ArticulationKinematicsSolver(self.my_franka,self.lula_kinematics_solver,self.end_effector_name)

        #Query the position of the "panda_hand" frame
        self.ee_position,ee_rot_mat = self.articulation_kinematics_solver.compute_end_effector_pose()
        self.ee_orientation = rot_matrices_to_quats(ee_rot_mat)

        #Create a cuboid to visualize where the "panda_hand" frame is according to the kinematics"
        self.panda_hand_visualization = VisualCuboid("/panda_hand",position=self.ee_position,orientation=self.ee_orientation,size=np.array([.05,.1,.05]),color=np.array([0,0,1]))

        self.articulation_controller = self.my_franka.get_articulation_controller()

        # setting up the world with a cube
        self.timeline = omni.timeline.get_timeline_interface()
        # setup the ROS2 subscriber here
        self.ros_sub = self.create_subscription(Pose, "RightHand/pose", self.move_cube_callback, 10)

    def move_cube_callback(self, data:Pose):
        # callback function to set the cube position to a new one upon receiving a (empty) ROS2 message
        if self.my_world.is_playing():
            self.stage.GetPrimAtPath("/World/TargetCube").GetAttribute("xformOp:translate").Set((data.position.z, data.position.x, data.position.y))
            print((data.position.x, data.position.y, data.position.z))
 
    def run_simulation(self):
        #Track any movements of the robot base

        self.timeline.play()
        while simulation_app.is_running():
            self.my_world.step(render=True)
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.my_world.is_playing():
                if self.my_world.current_time_step_index == 0:
                    self.my_world.reset()

                # the actual setting the cube pose is done here
                robot_base_translation,robot_base_orientation = self.my_franka.get_world_pose()
                self.lula_kinematics_solver.set_robot_base_pose(robot_base_translation,robot_base_orientation)

                #Use forward kinematics to update the position of the panda_hand_visualization
                ee_position,ee_rot_mat = self.articulation_kinematics_solver.compute_end_effector_pose()
                ee_orientation = rot_matrices_to_quats(ee_rot_mat)
                self.panda_hand_visualization.set_world_pose(ee_position,ee_orientation)

                observations = self.my_world.get_observations()
                actions, success = self.articulation_kinematics_solver.compute_inverse_kinematics(
                    target_position=observations[self.target_name]["position"],
                    target_orientation=observations[self.target_name]["orientation"],
                )
                if success:
                    self.articulation_controller.apply_action(actions)
                else:
                    carb.log_warn("IK did not converge to a solution.  No action is being taken.")

        # Cleanup
        self.timeline.stop()
        self.destroy_node()
        simulation_app.close()


if __name__ == "__main__":
    rclpy.init()
    subscriber = Subscriber()
    subscriber.run_simulation()