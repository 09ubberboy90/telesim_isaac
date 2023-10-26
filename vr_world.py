from collections import defaultdict
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
from copy import deepcopy
from functools import partial


class Pose:
    def __init__(self, position=None, orientation=None) -> None:
        self.position = position
        self.orientation = orientation
        self.reseted = True

    def resets(self):
        self.position = None
        self.orientation = None
        self.reseted = True

    def reinit(self, position, orientation):
        self.position = position
        self.orientation = orientation
        self.reseted = False
        return self


def report_hit(hand, hit):
    hits[hand].append(hit.rigid_body)
    return True

def quatd2arr(quatd):
    return [quatd.GetReal(), quatd.GetImaginary()[0], quatd.GetImaginary()[1], quatd.GetImaginary()[2]]

def vr_pose_handler(value, hand):
    ### handle offset
    rot = ref_rot * value[1]
    corrected = quatd2arr(rot)
    pose = list(value[0])
    pose = list(map(add, pose, pos_offset))
    source[hand].set_world_pose(position=pose, orientation=corrected)
    bool = targets_path[hand] in hits[opposite[hand]] if MIRROR else targets_path[hand] in hits[hand]
    if bool:
        trigger = trigggers.get(hand, None)
        if trigger is not None and trigger:
            attr_pos = targets[hand].GetAttribute("xformOp:translate")
            attr_rot = targets[hand].GetAttribute("xformOp:orient")

            ### Get original target pose
            target_pose = poses[hand]["target"]
            if target_pose.reseted:
                target_pose.reinit(attr_pos.Get(), attr_rot.Get())
            
            ### Get original source pose
            source_pose = poses[hand]["source"]
            if source_pose.reseted:
                source_pose.reinit(pose, rot.GetInverse())

            ### Move the object only by the delta of position since the trigger has been pressed
            hit_pose = pxr.Gf.Vec3d(
                *[float(target_pose.position[i] + pose[i] - source_pose.position[i]) for i in range(len(pose))]
            )

            ### Get the current rotation of the source controller
            ### Undo the original source rotation so that all that is left is how much the source has rotated since the button has been pressed
            deltaq = rot * source_pose.orientation

            ### Takes in the original rotation of the target object
            ### Apply the delta rotation of the source controller
            hit_rot = deltaq * target_pose.orientation
            ### Note: Use the current rotation of the target object to get a continuous rotation

            attr_pos.Set(hit_pose)
            attr_rot.Set(hit_rot.GetNormalized())

        elif not trigger:
            poses[hand]["target"].resets()
            poses[hand]["source"].resets()

# action callback
def on_action_event(hand, path, value):
    if path == f"/user/hand/{hand}/input/trigger/value":
        trigggers[hand] = True if value > 0.5 else False
    if path == f"/user/hand/{hand}/input/grip/pose":
        vr_pose_handler(value, hand)
    hits[hand] = []


# get stage unit
stage = omni.usd.get_context().get_stage()
meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
pos_offset = [0.0, 0.75, -0.75]
rot_offset = [90, 0, 0]
ref_rot = pxr.Gf.Quatd(0.707, 0.707, 0.0, 0.0)  ## Handles ref correction

# acquire interface
xr = _openxr.acquire_openxr_interface()
# setup OpenXR application using default parameters
xr.init()
xr.create_instance()
xr.get_system()

# create session and define interaction profiles
source = {"left": None, "right": None}

for hand in source.keys():
    xr.subscribe_action_event(
        f"/user/hand/{hand}/input/grip/pose",
        callback=partial(on_action_event, hand),
        reference_space=_openxr.XR_REFERENCE_SPACE_TYPE_STAGE,
    )
    xr.subscribe_action_event(f"/user/hand/{hand}/input/trigger/value", callback=partial(on_action_event, hand))
    xr.subscribe_action_event(f"/user/hand/{hand}/output/haptic")
    source[hand] = VisualCuboid(
        f"/{hand}_cube",
        position=np.array([0.0, 0, 0]),
        orientation=np.array([0, 1, 0, 0]),
        size=0.03,
        color=np.array([0, 0, 1]),
    )
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(f"/{hand}_cube"))
    PhysxSchema.PhysxTriggerAPI.Apply(stage.GetPrimAtPath(f"/{hand}_cube"))
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(f"/World/Control/visual_{hand}_cube"))
    PhysxSchema.PhysxTriggerAPI.Apply(stage.GetPrimAtPath(f"/World/Control/visual_{hand}_cube"))

hits = defaultdict(list)

objs = {"target": Pose(), "source": Pose()}
poses = {"left": deepcopy(objs), "right": deepcopy(objs)}

targets_path = {"left": "/World/Control/visual_left_cube", "right": "/World/Control/visual_right_cube"}
targets = {"left": stage.GetPrimAtPath(targets_path["left"]), "right": stage.GetPrimAtPath(targets_path["right"])}
opposite = {"left": "right", "right": "left"}
MIRROR = False
trigggers = {}

xr.create_session()

# setup cameras and viewports and prepare rendering using the internal callback
xr.set_meters_per_unit(meters_per_unit)
xr.setup_stereo_view(camera_properties={"focalLength": 5})
xr.set_reference_system_pose(position=pxr.Gf.Vec3d(*pos_offset), rotation=pxr.Gf.Vec3d(*rot_offset))
xr.set_frame_transformations(flip=0, fit=True)
xr.set_stereo_rectification(y= 10 * math.pi / 180.0)

# execute action and rendering loop on each simulation step
def on_simulation_step(step):
    for hand, obj in targets.items():
        pos = list(obj.GetAttribute("xformOp:translate").Get())
        rot = quatd2arr(obj.GetAttribute("xformOp:orient").Get())
        get_physx_scene_query_interface().overlap_box([0.025] * 3, pos, rot, partial(report_hit, hand), False)

    if xr.poll_events() and xr.is_session_running():
        xr.poll_actions()
        xr.render_views(_openxr.XR_REFERENCE_SPACE_TYPE_STAGE)


physx_subs = omni.physx.get_physx_interface().subscribe_physics_step_events(on_simulation_step)
