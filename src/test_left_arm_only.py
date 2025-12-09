"""Test just the left arm pick and place (no cameras, simplified)"""

import numpy as np
from pathlib import Path
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    StartMeshcat,
    RigidTransform,
)
from manipulation.station import LoadScenario, MakeHardwareStation
import sys
sys.path.append(str(Path(__file__).parent))
from brick_pickplace_clean import (
    write_table_sdf,
    write_brick_sdf,
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

# Left arm at [-0.5, -0.5], brick at [-0.5, -0.2]
brick_pos = [-0.50, -0.20, 0.0]

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
            translation: [{brick_pos[0]}, {brick_pos[1]}, {brick_size[2]/2.0 + 0.001}]
            rotation: !Rpy {{ deg: [0, 0, 0] }}
model_drivers:
    iiwa_left: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg_left
    wsg_left: !SchunkWsgDriver {{}}
"""

print("=== LEFT ARM ONLY TEST ===\n")
print(f"Left robot base: [-0.5, -0.5, 0]")
print(f"Brick position: [{brick_pos[0]:.2f}, {brick_pos[1]:.2f}, {brick_size[2]/2.0 + 0.001:.4f}]")
print(f"Distance forward from robot: {abs(brick_pos[1] - (-0.5)):.2f}m\n")

meshcat = StartMeshcat()
scenario = LoadScenario(data=scenario_yaml)
station = MakeHardwareStation(scenario, meshcat)

builder = DiagramBuilder()
builder.AddSystem(station)
diagram = builder.Build()

context = diagram.CreateDefaultContext()
diagram.ForcedPublish(context)
plant = station.plant()
plant_context = diagram.GetSubsystemContext(plant, context)

# Get positions
robot_model = plant.GetModelInstanceByName("iiwa_left")
robot_base = plant.GetBodyByName("iiwa_link_0", robot_model)
X_WR = plant.EvalBodyPoseInWorld(plant_context, robot_base)

gripper_model = plant.GetModelInstanceByName("wsg_left")
gripper_body = plant.GetBodyByName("body", gripper_model)
X_WG = plant.EvalBodyPoseInWorld(plant_context, gripper_body)

brick_model = plant.GetModelInstanceByName("brick1")
brick_body = plant.GetBodyByName("brick_link", brick_model)
X_WB = plant.EvalBodyPoseInWorld(plant_context, brick_body)

print("=" * 70)
print("POSITION CHECK")
print("=" * 70)
print(f"Robot base (iiwa_link_0): {X_WR.translation()}")
print(f"Gripper (wsg body):       {X_WG.translation()}")
print(f"Brick:                    {X_WB.translation()}")
print()
print(f"Distance from gripper to brick: {np.linalg.norm(X_WG.translation() - X_WB.translation()):.3f}m")
print()

# Test design_grasp_pose
from brick_pickplace_clean import design_grasp_pose
X_OG, X_WGpick = design_grasp_pose(X_WB)

print("Grasp pose calculation:")
print(f"  Brick pose (X_WB):      {X_WB.translation()}")
print(f"  Grasp offset (X_OG):    {X_OG.translation()}")
print(f"  Pick pose (X_WGpick):   {X_WGpick.translation()}")
print(f"  Gripper Z above table:  {X_WGpick.translation()[2]*1000:.1f}mm")
print(f"  Brick top surface:      {(X_WB.translation()[2] + brick_size[2]/2)*1000:.1f}mm")
print(f"  Gap:                    {(X_WGpick.translation()[2] - (X_WB.translation()[2] + brick_size[2]/2))*1000:.1f}mm")

if X_WGpick.translation()[2] > 0.10:
    print("\n❌ PROBLEM: Gripper will be >100mm above table - too high!")
elif X_WGpick.translation()[2] < 0.02:
    print("\n⚠ WARNING: Gripper very low")
else:
    print("\n✓ Grasp height looks reasonable")

print("=" * 70)
print("\nCheck Meshcat at http://localhost:7000")
print("Scene loaded successfully!")
