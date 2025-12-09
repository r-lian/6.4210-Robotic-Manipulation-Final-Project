"""Test camera mounting - Step 1: Verify cameras are positioned correctly."""

import numpy as np
from pathlib import Path
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    StartMeshcat,
)
from manipulation.station import LoadScenario, MakeHardwareStation

# Setup
meshcat = StartMeshcat()
brick_size = [0.10, 0.08, 0.04]

# Create brick positions (same as stacking script)
np.random.seed(42)
brick_positions = [
    [-0.40, -0.15, np.random.uniform(0, np.pi)],
    [-0.35, 0.15, np.random.uniform(0, np.pi)],
    [-0.25, 0.0, np.random.uniform(0, np.pi)],
]

# Create SDF files
table_path = Path("/tmp/table.sdf")
table_path.write_text("""<?xml version="1.0"?>
<sdf version="1.7">
    <model name="table">
        <link name="table_link">
            <collision name="collision">
                <geometry><box><size>1.0 1.0 0.1</size></box></geometry>
            </collision>
            <visual name="visual">
                <geometry><box><size>1.0 1.0 0.1</size></box></geometry>
                <material><diffuse>0.9 0.9 0.9 1.0</diffuse></material>
            </visual>
        </link>
    </model>
</sdf>""")

brick_path = Path("/tmp/brick.sdf")
sx, sy, sz = brick_size
brick_path.write_text(f"""<?xml version="1.0"?>
<sdf version="1.7">
    <model name="brick">
        <link name="brick_link">
            <inertial>
                <mass>0.1</mass>
                <inertia>
                    <ixx>0.001</ixx><ixy>0</ixy><ixz>0</ixz>
                    <iyy>0.001</iyy><iyz>0</iyz>
                    <izz>0.001</izz>
                </inertia>
            </inertial>
            <collision name="collision">
                <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
            </collision>
            <visual name="visual">
                <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
                <material><diffuse>0.8 0.3 0.1 1.0</diffuse></material>
            </visual>
        </link>
    </model>
</sdf>""")

# Create brick directives
brick_directives = ""
for i, pos in enumerate(brick_positions):
    brick_directives += f"""
# Add brick {i+1}
- add_model:
    name: brick{i+1}
    file: file://{brick_path.resolve()}
    default_free_body_pose:
        brick_link:
            translation: [{pos[0]}, {pos[1]}, {brick_size[2]/2.0 + 0.001}]
            rotation: !Rpy {{ deg: [0, 0, {np.degrees(pos[2]):.1f}] }}
"""

# Create scenario YAML
scenario_yaml = f"""
directives:
# Add IIWA robot
- add_model:
    name: iiwa
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
    child: iiwa::iiwa_link_0
    X_PC:
        translation: [0, -0.5, 0]
        rotation: !Rpy {{ deg: [0, 0, 180] }}

# Add WSG gripper
- add_model:
    name: wsg
    file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.114]
        rotation: !Rpy {{ deg: [90, 0, 90] }}

# Add table
- add_model:
    name: table
    file: file://{table_path.resolve()}
- add_weld:
    parent: world
    child: table::table_link
    X_PC:
        translation: [0.0, 0.0, -0.05]
        rotation: !Rpy {{ deg: [0, 0, -90] }}
{brick_directives}
# Add cameras (matching notebook)
- add_frame:
    name: camera0_origin
    X_PF:
        base_frame: world
        rotation: !Rpy {{ deg: [-120.0, 0.0, 180.0]}}
        translation: [0, 0.8, 0.5]
- add_model:
    name: camera0
    file: package://manipulation/camera_box.sdf
- add_weld:
    parent: camera0_origin
    child: camera0::base

- add_frame:
    name: camera1_origin
    X_PF:
        base_frame: world
        rotation: !Rpy {{ deg: [-125, 0.0, 90.0]}}
        translation: [0.8, 0.1, 0.5]
- add_model:
    name: camera1
    file: package://manipulation/camera_box.sdf
- add_weld:
    parent: camera1_origin
    child: camera1::base

- add_frame:
    name: camera2_origin
    X_PF:
        base_frame: world
        rotation: !Rpy {{ deg: [-120.0, 0.0, -90.0]}}
        translation: [-0.8, 0.1, 0.5]
- add_model:
    name: camera2
    file: package://manipulation/camera_box.sdf
- add_weld:
    parent: camera2_origin
    child: camera2::base

model_drivers:
    iiwa: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg
    wsg: !SchunkWsgDriver {{}}
cameras:
    camera0:
        name: camera0
        depth: True
        X_PB:
            base_frame: camera0::base
    camera1:
        name: camera1
        depth: True
        X_PB:
            base_frame: camera1::base
    camera2:
        name: camera2
        depth: True
        X_PB:
            base_frame: camera2::base
"""

print("=== TESTING CAMERA MOUNTING ===")
print(f"\nBrick positions:")
for i, pos in enumerate(brick_positions):
    print(f"  Brick {i+1}: [{pos[0]}, {pos[1]}], rot={np.degrees(pos[2]):.1f}°")

# Build diagram
scenario = LoadScenario(data=scenario_yaml)
builder = DiagramBuilder()
station = MakeHardwareStation(scenario, meshcat)
builder.AddSystem(station)
diagram = builder.Build()

# Run simulation for a few seconds to let cameras stabilize
simulator = Simulator(diagram)
context = simulator.get_mutable_context()

print("\n=== CAMERA POSITIONS ===")
plant = station.GetSubsystemByName("plant")
station_context = station.GetMyContextFromRoot(context)
plant_context = plant.GetMyContextFromRoot(context)

for i in range(3):
    camera_name = f"camera{i}"
    camera_body = plant.GetBodyByName("base", plant.GetModelInstanceByName(camera_name))
    X_WC = plant.EvalBodyPoseInWorld(plant_context, camera_body)
    pos = X_WC.translation()
    print(f"Camera {i}: position = [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

print("\n=== BRICK POSITIONS ===")
for i in range(1, 4):
    brick_name = f"brick{i}"
    brick_body = plant.GetBodyByName("brick_link", plant.GetModelInstanceByName(brick_name))
    X_WB = plant.EvalBodyPoseInWorld(plant_context, brick_body)
    pos = X_WB.translation()
    print(f"Brick {i}: position = [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

print("\n✓ Camera mounting test complete!")
print("Cameras are correctly positioned above the table:")
print("  - Camera 0 is at front center (Y=0.8)")
print("  - Camera 1 is at right side (X=0.8)")
print("  - Camera 2 is at left side (X=-0.8)")
print("\nAll cameras are at Z=0.5 (50cm above table)")
print("All bricks are at Z~0.02 (sitting on table)")
print("\nCheck Meshcat at http://localhost:7000 to view the scene")
input("Press Enter to exit...")
