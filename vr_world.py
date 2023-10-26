import omni
from pxr import UsdGeom
import pxr
from semu.xr.openxr import _openxr
from omni.isaac.sensor import Camera
from omni.isaac.core import World
import omni.isaac.core.utils.numpy.rotations as rot_utils
import numpy as np
import matplotlib.pyplot as plt
import math
from omni.isaac.core.objects.sphere import VisualSphere
from omni.isaac.core.objects.cuboid import VisualCuboid
from operator import add 
from omni.physx import get_physx_scene_query_interface
from usdrt import Gf
from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics

# get stage unit
stage = omni.usd.get_context().get_stage()
meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)

pos_offset = [2.5, 0.5, -0.75]
rot_offset = [90, 0, 0]

right_cube = VisualCuboid(
    "/right_cube",
    position=np.array([0.0, 0, 0]),
    orientation=np.array([0,1,0,0]),
    size=0.03,
    color=np.array([0, 0, 1]),
)
left_cube = VisualCuboid(
    "/left_cube",
    position=np.array([0.0, 0, 0]),
    orientation=np.array([0,1,0,0]),
    size=0.03,
    color=np.array([0, 0, 1]),
)
right_hits = []
left_hits = []

def euler_from_quaternion(w, x, y, z):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def quatdtolist(quatd):
    return [quatd.GetReal(), quatd.GetImaginary()[0], quatd.GetImaginary()[1], quatd.GetImaginary()[2]]

def left_report_hit(hit):
    # When a collision is detected, the object color changes to red.
    left_hits.append(hit.rigid_body)
    return True     

def right_report_hit(hit):
    # When a collision is detected, the object color changes to red.
    right_hits.append(hit.rigid_body)
    return True           

def correct_input_vr(inrot):
    inrot = ref_rot * inrot
    return inrot

ref_rot = pxr.Gf.Quatd(0.707, 0.707,0.0, 0.0)  ## Handles ref correction	

z_rot = pxr.Gf.Quatd(0.707, 0.0, 0.0, .707)  ## Handles ref correction
y_rot = pxr.Gf.Quatd(0.707, 0.0 , 0.707, 0.0)  ## Handles ref correction
x_rot = pxr.Gf.Quatd(0.707, 0.707,0.0, 0.0)  ## Handles ref correction

def vr_pose_handler(value, hand):
    global left_hits, orig_left_rot, orig_left_target_rot, orig_left_pose, orig_left_target_pose, right_hits, orig_right_rot, orig_right_target_rot, orig_right_pose, orig_right_target_pose

    trigger = controller_last_triggger.get(hand, None)
    rot = correct_input_vr(value[1])
    corrected = [rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2]]
    pose = list(value[0])
    pose = list(map(add, pose, pos_offset))  
    if hand == "left":
        left_cube.set_world_pose(position=pose, orientation=corrected)
    if hand=="right":
        right_cube.set_world_pose(position=pose, orientation=corrected)


    if right_target_path in left_hits:
        #xr.apply_haptic_feedback("/user/hand/left/output/haptic", {"duration": _openxr.XR_MIN_HAPTIC_DURATION})
        if trigger is not None and trigger:
            attr_pos = right_target.GetAttribute("xformOp:translate")
            attr_rot = right_target.GetAttribute("xformOp:orient")

            if orig_left_pose is None:
                orig_left_pose = attr_pos.Get()

            if orig_left_target_pose is None:
                orig_left_target_pose = pose

            hit_pose= pxr.Gf.Vec3d(*[float(orig_left_pose[i] + pose[i] - orig_left_target_pose[i]) for i in range(len(pose))] )

            ### Get original target rotation
            if orig_left_rot is None:
                orig_left_rot = attr_rot.Get() 
            
            ### Get original source rotation
            if orig_left_target_rot is None:
                orig_left_target_rot = rot.GetInverse()
            
            ### Get the current rotation of the source controller
            ### Undo the original source rotation so that all that is left is how much the source has rotated since the button has been pressed
            deltaq = rot  * orig_left_target_rot

            ### Takes in the original rotation of the target object
            ### Apply the delta rotation of the source controller
            hit_rot = deltaq * orig_left_rot
            ### Note: Use the current rotation of the target object to get a continuous rotation

            attr_pos.Set(hit_pose)
            attr_rot.Set(hit_rot.GetNormalized())
        elif not trigger:
            orig_left_rot = None
            orig_left_target_rot = None  
            orig_left_pose = None
            orig_left_target_pose = None  

    # if left_target_path in right_hits:
    #     #xr.apply_haptic_feedback("/user/hand/left/output/haptic", {"duration": _openxr.XR_MIN_HAPTIC_DURATION})
    #     if trigger is not None and trigger:
    #         attr_pos = left_target.GetAttribute("xformOp:translate")
    #         attr_rot = left_target.GetAttribute("xformOp:orient")

    #         if orig_right_pose is None:
    #             orig_right_pose = attr_pos.Get()

    #         if orig_right_target_pose is None:
    #             orig_right_target_pose = pose

    #         hit_pose= pxr.Gf.Vec3d(*[float(orig_right_pose[i] + pose[i] - orig_right_target_pose[i]) for i in range(len(pose))] )

    #         attr_pos.Set(hit_pose)
    #         attr_rot.Set(hit_rot)
    #     elif not trigger:
    #         orig_right_rot = None
    #         orig_right_target_rot = None  
    #         orig_right_pose = None
    #         orig_right_target_pose = None  

orig_right_rot = None
orig_left_rot = None
orig_right_pose = None
orig_left_pose = None
orig_left_target_pose = None
orig_right_target_pose = None
orig_left_target_rot = None
orig_right_target_rot = None

# acquire interface
xr = _openxr.acquire_openxr_interface()

# setup OpenXR application using default parameters
xr.init()
xr.create_instance()
xr.get_system()

right_target_path = "/World/Control/right_cube"
left_target_path = "/World/Control/left_cube"

right_target = stage.GetPrimAtPath(right_target_path)
left_target = stage.GetPrimAtPath(left_target_path)

controller_last_pose = {}
controller_last_triggger = {}
# action callback
def on_action_event(path, value):
    global left_hits
    # if len(left_hits) != 0:
    #     print(f"left_hits: {left_hits}")
    # process controller's trigger
    if path == "/user/hand/left/input/trigger/value":
        # modify the sphere's radius (from 1 to 10 centimeters) according to the controller's trigger position
        # apply haptic vibration when the controller's trigger is fully depressed
        controller_last_triggger["left"] = True if value > .5 else False
    # mirror the controller's pose on the sphere (cartesian position and rotation as quaternion)
    if path == "/user/hand/left/input/grip/pose":
        vr_pose_handler(value, "left")
    left_hits = []


def on_right_action_event(path, value):
    global right_hits
    # if len(right_hits) != 0:
    #     print(f"right_hits: {right_hits}")

    # process controller's trigger
    if path == "/user/hand/right/input/trigger/value":
        # modify the sphere's radius (from 1 to 10 centimeters) according to the controller's trigger position
        # apply haptic vibration when the controller's trigger is fully depressed
        controller_last_triggger["right"] = True if value > .5 else False
    # mirror the controller's pose on the sphere (cartesian position and rotation as quaternion)
    if path == "/user/hand/right/input/grip/pose":
        vr_pose_handler(value, "right")
    right_hits = []
# subscribe controller actions (haptic actions don't require callbacks) 
xr.subscribe_action_event("/user/hand/left/input/grip/pose", callback=on_action_event, reference_space=_openxr.XR_REFERENCE_SPACE_TYPE_STAGE)
xr.subscribe_action_event("/user/hand/left/input/trigger/value", callback=on_action_event)
xr.subscribe_action_event("/user/hand/left/output/haptic")

xr.subscribe_action_event("/user/hand/right/input/grip/pose", callback=on_right_action_event, reference_space=_openxr.XR_REFERENCE_SPACE_TYPE_STAGE)
xr.subscribe_action_event("/user/hand/right/input/trigger/value", callback=on_right_action_event)
xr.subscribe_action_event("/user/hand/right/output/haptic")

# create session and define interaction profiles
xr.create_session()

# setup cameras and viewports and prepare rendering using the internal callback
xr.set_meters_per_unit(meters_per_unit)
xr.setup_stereo_view()
xr.set_reference_system_pose(position=pxr.Gf.Vec3d(*pos_offset), rotation=pxr.Gf.Vec3d(*rot_offset))
xr.set_frame_transformations(flip=0, fit=True)
xr.set_stereo_rectification(y= 5 * math.pi / 180.0)

UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/right_cube"))
PhysxSchema.PhysxTriggerAPI.Apply(stage.GetPrimAtPath("/right_cube"))
UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/left_cube"))
PhysxSchema.PhysxTriggerAPI.Apply(stage.GetPrimAtPath("/left_cube"))
UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/Control/right_cube"))
UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/Control/left_cube"))
PhysxSchema.PhysxTriggerAPI.Apply(stage.GetPrimAtPath("/World/Control/right_cube"))
PhysxSchema.PhysxTriggerAPI.Apply(stage.GetPrimAtPath("/World/Control/left_cube"))

# execute action and rendering loop on each simulation step
def on_simulation_step(step):
    
    righ_pose, right_rot = right_cube.get_world_pose()
    left_pose, left_rot = left_cube.get_world_pose()

    numHits = get_physx_scene_query_interface().overlap_box([0.03]*3, left_pose, left_rot, left_report_hit, False)
    numHits = get_physx_scene_query_interface().overlap_box([0.03]*3, righ_pose, right_rot, right_report_hit, False)

    if xr.poll_events() and xr.is_session_running():
        xr.poll_actions()
        xr.render_views(_openxr.XR_REFERENCE_SPACE_TYPE_STAGE)

physx_subs = omni.physx.get_physx_interface().subscribe_physics_step_events(on_simulation_step)