"""Test just the left arm to debug positioning"""

import numpy as np
from pathlib import Path
from pydrake.all import StartMeshcat
from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import DiagramBuilder, Simulator

# Setup
assets_dir = Path("assets")
brick_size = [0.10, 0.08, 0.04]

table_sdf = assets_dir / "table.sdf"
brick_sdf = assets_dir / "brick_model" / "brick.sdf"

# Test: robot at [-0.5, -0.5], brick at [-0.5, -0.2]
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
    file: file://{table_sdf.resolve()}
- add_weld:
    parent: world
    child: table::table_link
    X_PC:
        translation: [0.0, 0.0, -0.05]
        rotation: !Rpy {{ deg: [0, 0, -90] }}
- add_model:
    name: brick1
    file: file://{brick_sdf.resolve()}
    default_free_body_pose:
        brick_link:
            translation: [-0.50, -0.20, {brick_size[2]/2.0 + 0.001}]
            rotation: !Rpy {{ deg: [0, 0, 0] }}
model_drivers:
    iiwa_left: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg_left
    wsg_left: !SchunkWsgDriver {{}}
"""

scenario = LoadScenario(data=scenario_yaml)
meshcat = StartMeshcat()
station = MakeHardwareStation(scenario, meshcat)

builder = DiagramBuilder()
builder.AddSystem(station)
diagram = builder.Build()

context = diagram.CreateDefaultContext()
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
print("Robot rotation (Z-axis in radians):")
print(f"  Yaw: {X_WR.rotation().ToRollPitchYaw().yaw_angle():.2f} rad ({np.degrees(X_WR.rotation().ToRollPitchYaw().yaw_angle()):.1f}°)")
print()
print("Expected: Brick should be ~0.3m in front of gripper")
print("=" * 70)

# Force publish to visualize
diagram.ForcedPublish(context)
print("\nCheck Meshcat at http://localhost:7000")
print("Press Ctrl+C to exit")

try:
    while True:
        pass
except KeyboardInterrupt:
    print("\nExiting...")
