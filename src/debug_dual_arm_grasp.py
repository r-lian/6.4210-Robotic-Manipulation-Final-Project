"""Debug dual-arm grasping positions in detail"""

import numpy as np
from pathlib import Path
from pydrake.all import (
    DiagramBuilder,
    RigidTransform,
)
from manipulation.station import LoadScenario, MakeHardwareStation
import sys
sys.path.append(str(Path(__file__).parent))
from brick_pickplace_clean import (
    write_table_sdf,
    write_brick_sdf,
    design_grasp_pose,
    design_pregrasp_pose,
)

brick_size = [0.10, 0.08, 0.04]

# Create SDFs
assets_dir = Path("assets")
assets_dir.mkdir(exist_ok=True)
table_sdf_path = assets_dir / "table.sdf"
write_table_sdf(table_sdf_path)
brick_dir = assets_dir / "brick_model"
brick_dir.mkdir(exist_ok=True)
brick_sdf_path = brick_dir / "brick.sdf"
write_brick_sdf(brick_sdf_path, brick_size)

np.random.seed(42)
brick1_pos = [-0.50, -0.20, np.random.uniform(0, np.pi)]  # Left arm's brick

scenario_yaml = f"""
directives:
- add_model:
    name: iiwa_left
    file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1.6]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa_left::iiwa_link_0
    X_PC:
        translation: [-0.5, -0.5, 0]
        rotation: !Rpy {{ deg: [0, 0, 180] }}
- add_model:
    name: wsg_left
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa_left::iiwa_link_7
    child: wsg_left::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 90]}}
- add_model:
    name: table
    file: file://{table_sdf_path.resolve()}
- add_weld:
    parent: world
    child: table::table_link
    X_PC:
        translation: [0.0, 0.0, -0.05]
        rotation: !Rpy {{ deg: [0, 0, -90] }}
- add_model:
    name: brick1
    file: file://{brick_sdf_path.resolve()}
    default_free_body_pose:
        brick_link:
            translation: [{brick1_pos[0]}, {brick1_pos[1]}, {brick_size[2]/2.0 + 0.001}]
            rotation: !Rpy {{ deg: [0, 0, {np.degrees(brick1_pos[2]):.1f}] }}
model_drivers:
    iiwa_left: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg_left
    wsg_left: !SchunkWsgDriver {{}}
"""

print("=" * 70)
print("DUAL-ARM GRASP DEBUG")
print("=" * 70)

scenario = LoadScenario(data=scenario_yaml)
station = MakeHardwareStation(scenario)
builder = DiagramBuilder()
builder.AddSystem(station)
diagram = builder.Build()

context = diagram.CreateDefaultContext()
diagram.ForcedPublish(context)
plant = station.plant()
plant_context = diagram.GetSubsystemContext(plant, context)

# Get actual brick pose from plant (ground truth)
model_brick1 = plant.GetModelInstanceByName("brick1")
frame_brick1 = plant.GetFrameByName("brick_link", model_instance=model_brick1)
brick1_pose = plant.CalcRelativeTransform(plant_context, plant.world_frame(), frame_brick1)

print(f"\n1. BRICK GROUND TRUTH:")
print(f"   Position: {brick1_pose.translation()}")
print(f"   X: {brick1_pose.translation()[0]:.4f}m")
print(f"   Y: {brick1_pose.translation()[1]:.4f}m")
print(f"   Z: {brick1_pose.translation()[2]:.4f}m")

# Get gripper initial pose
gripper_model = plant.GetModelInstanceByName("wsg_left")
gripper_body = plant.GetBodyByName("body", gripper_model)
X_WGinitial = plant.EvalBodyPoseInWorld(plant_context, gripper_body)

print(f"\n2. GRIPPER INITIAL POSE:")
print(f"   Position: {X_WGinitial.translation()}")
print(f"   X: {X_WGinitial.translation()[0]:.4f}m")
print(f"   Y: {X_WGinitial.translation()[1]:.4f}m")
print(f"   Z: {X_WGinitial.translation()[2]:.4f}m")

# Calculate grasp pose using design_grasp_pose (same as dual_arm_pickplace.py)
X_OG, X_WGpick = design_grasp_pose(brick1_pose)

print(f"\n3. GRASP POSE CALCULATION:")
print(f"   Grasp offset (X_OG): {X_OG.translation()}")
print(f"   Pick pose (X_WGpick): {X_WGpick.translation()}")
print(f"     X: {X_WGpick.translation()[0]:.4f}m")
print(f"     Y: {X_WGpick.translation()[1]:.4f}m")
print(f"     Z: {X_WGpick.translation()[2]:.4f}m")

# Pre-grasp
X_WGprepick = design_pregrasp_pose(X_WGpick)

print(f"\n4. PRE-GRASP POSE:")
print(f"   Position: {X_WGprepick.translation()}")
print(f"     X: {X_WGprepick.translation()[0]:.4f}m")
print(f"     Y: {X_WGprepick.translation()[1]:.4f}m")
print(f"     Z: {X_WGprepick.translation()[2]:.4f}m")

# Analysis
print(f"\n5. ALIGNMENT CHECK:")
brick_xy = brick1_pose.translation()[:2]
pick_xy = X_WGpick.translation()[:2]
prepick_xy = X_WGprepick.translation()[:2]

print(f"   Brick XY:    [{brick_xy[0]:.4f}, {brick_xy[1]:.4f}]")
print(f"   Pick XY:     [{pick_xy[0]:.4f}, {pick_xy[1]:.4f}]")
print(f"   Pre-pick XY: [{prepick_xy[0]:.4f}, {prepick_xy[1]:.4f}]")

xy_error_pick = np.linalg.norm(pick_xy - brick_xy)
xy_error_prepick = np.linalg.norm(prepick_xy - brick_xy)

print(f"\n   XY error at pick: {xy_error_pick*1000:.1f}mm")
print(f"   XY error at pre-pick: {xy_error_prepick*1000:.1f}mm")

if xy_error_pick > 0.001:
    print(f"\n   ❌ PROBLEM: Pick pose is {xy_error_pick*1000:.1f}mm off center!")
    print(f"      The gripper won't align with the brick center")
else:
    print(f"\n   ✓ Pick pose is centered on brick")

# Check robot base vs brick
robot_model = plant.GetModelInstanceByName("iiwa_left")
robot_base = plant.GetBodyByName("iiwa_link_0", robot_model)
X_WR = plant.EvalBodyPoseInWorld(plant_context, robot_base)

print(f"\n6. ROBOT BASE CHECK:")
print(f"   Robot base: {X_WR.translation()}")
print(f"   Brick:      {brick1_pose.translation()}")
print(f"   Distance Y: {abs(brick1_pose.translation()[1] - X_WR.translation()[1]):.3f}m (forward from robot)")

print("=" * 70)
