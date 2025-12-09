"""Test point cloud capture - Step 2: Verify cameras can see bricks."""

import numpy as np
from pathlib import Path
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    StartMeshcat,
    PointCloud,
)
from manipulation.station import LoadScenario, MakeHardwareStation, AddPointClouds

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

print("=== TESTING POINT CLOUD CAPTURE ===")

# Build diagram with point cloud systems
scenario = LoadScenario(data=scenario_yaml)
builder = DiagramBuilder()
station = MakeHardwareStation(scenario, meshcat)
builder.AddSystem(station)

# Add point cloud aggregation
to_point_cloud = AddPointClouds(scenario=scenario, station=station, builder=builder, meshcat=meshcat)
builder.ExportOutput(to_point_cloud["camera0"].get_output_port(), "camera_point_cloud0")
builder.ExportOutput(to_point_cloud["camera1"].get_output_port(), "camera_point_cloud1")
builder.ExportOutput(to_point_cloud["camera2"].get_output_port(), "camera_point_cloud2")

diagram = builder.Build()

# Create context and force publish to generate point clouds
context = diagram.CreateDefaultContext()
diagram.ForcedPublish(context)

# Get point clouds from each camera
print("\n=== POINT CLOUD STATISTICS ===")
for i in range(3):
    camera_name = f"camera{i}"
    port_name = f"camera_point_cloud{i}"

    # Get the point cloud from the diagram output port
    try:
        port = diagram.GetOutputPort(port_name)
        pc = port.Eval(context)

        num_points = pc.size()
        print(f"\nCamera {i} ({camera_name}):")
        print(f"  Total points: {num_points}")

        if num_points > 0:
            # Get XYZ coordinates
            xyz = pc.xyzs()
            x_range = [xyz[0, :].min(), xyz[0, :].max()]
            y_range = [xyz[1, :].min(), xyz[1, :].max()]
            z_range = [xyz[2, :].min(), xyz[2, :].max()]

            print(f"  X range: [{x_range[0]:.3f}, {x_range[1]:.3f}]")
            print(f"  Y range: [{y_range[0]:.3f}, {y_range[1]:.3f}]")
            print(f"  Z range: [{z_range[0]:.3f}, {z_range[1]:.3f}]")

            # Check if any points are near brick locations
            brick_points = 0
            for brick_pos in brick_positions:
                # Count points within 0.15m of each brick
                distances = np.sqrt(
                    (xyz[0, :] - brick_pos[0])**2 +
                    (xyz[1, :] - brick_pos[1])**2
                )
                brick_points += np.sum(distances < 0.15)

            print(f"  Points near bricks: {brick_points} (~{100*brick_points/num_points:.1f}%)")
        else:
            print("  ⚠ WARNING: No points captured!")

    except Exception as e:
        print(f"\nCamera {i}: Error - {e}")

print("\n=== MERGED POINT CLOUD ===")
try:
    from pydrake.all import Concatenate
    pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
    pc1 = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
    pc2 = diagram.GetOutputPort("camera_point_cloud2").Eval(context)
    merged_pc = Concatenate([pc0, pc1, pc2])
    print(f"Merged point cloud: {merged_pc.size()} total points")

    if merged_pc.size() > 0:
        print("✓ Point cloud capture is working!")
    else:
        print("✗ Point cloud capture failed - merged cloud is empty")
except Exception as e:
    print(f"Error getting merged point cloud: {e}")

print("\nCheck Meshcat at http://localhost:7000 to view point clouds")
print("Point clouds should be visible as colored points around the bricks")
