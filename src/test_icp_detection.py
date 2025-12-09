"""Test ICP detection - Step 3: Verify ICP can detect brick poses."""

import numpy as np
from pathlib import Path
from pydrake.all import (
    DiagramBuilder,
    StartMeshcat,
    PointCloud,
    Concatenate,
    RigidTransform,
)
from manipulation.station import LoadScenario, MakeHardwareStation, AddPointClouds
from manipulation.scenarios import AddMultibodyTriad
from manipulation.utils import ConfigureParser

# Import functions from brick_stacking
import sys
sys.path.append(str(Path(__file__).parent))
from brick_stacking import sample_brick_surface, detect_brick_with_icp

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

# Create scenario YAML with fixed cameras
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

print("=== TESTING ICP BRICK DETECTION ===")
print(f"\nActual brick positions (ground truth):")
for i, pos in enumerate(brick_positions):
    print(f"  Brick {i+1}: [{pos[0]:.2f}, {pos[1]:.2f}], rot={np.degrees(pos[2]):.1f}°")

# Build diagram
scenario = LoadScenario(data=scenario_yaml)
builder = DiagramBuilder()
station = MakeHardwareStation(scenario, meshcat)
builder.AddSystem(station)

# Add point clouds
to_point_cloud = AddPointClouds(scenario=scenario, station=station, builder=builder, meshcat=meshcat)
builder.ExportOutput(to_point_cloud["camera0"].get_output_port(), "camera_point_cloud0")
builder.ExportOutput(to_point_cloud["camera1"].get_output_port(), "camera_point_cloud1")
builder.ExportOutput(to_point_cloud["camera2"].get_output_port(), "camera_point_cloud2")

diagram = builder.Build()
context = diagram.CreateDefaultContext()
diagram.ForcedPublish(context)
plant = station.plant()
plant_context = diagram.GetSubsystemContext(plant, context)

# Get point clouds
pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
pc1 = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
pc2 = diagram.GetOutputPort("camera_point_cloud2").Eval(context)

print(f"\nPoint cloud sizes: {pc0.size()}, {pc1.size()}, {pc2.size()}")

# Test ICP detection on each brick
print("\n=== ICP DETECTION RESULTS ===")
for i in range(1, 4):
    brick_name = f"brick{i}"
    print(f"\nDetecting {brick_name}...")

    try:
        detected_pose = detect_brick_with_icp(pc0, pc1, pc2, plant, plant_context, brick_name, brick_size, meshcat)

        # Get ground truth pose
        brick_body = plant.GetBodyByName("brick_link", plant.GetModelInstanceByName(brick_name))
        true_pose = plant.EvalBodyPoseInWorld(plant_context, brick_body)

        # Compare
        detected_pos = detected_pose.translation()
        true_pos = true_pose.translation()
        error = np.linalg.norm(detected_pos - true_pos)

        print(f"  Ground truth: [{true_pos[0]:.3f}, {true_pos[1]:.3f}, {true_pos[2]:.3f}]")
        print(f"  ICP detected: [{detected_pos[0]:.3f}, {detected_pos[1]:.3f}, {detected_pos[2]:.3f}]")
        print(f"  Position error: {error*1000:.1f}mm")

        if error < 0.01:  # Less than 10mm error
            print(f"  ✓ Detection GOOD (error < 10mm)")
        elif error < 0.05:  # Less than 50mm error
            print(f"  ⚠ Detection OK (error < 50mm)")
        else:
            print(f"  ✗ Detection POOR (error > 50mm)")

    except Exception as e:
        print(f"  ✗ Detection FAILED: {e}")

print("\n=== TEST COMPLETE ===")
print("Check Meshcat at http://localhost:7000 to view detected brick poses")
