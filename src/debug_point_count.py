"""
Debug why brick_stacking.py gets fewer points than verify_icp_centered_bricks.py
"""

import numpy as np
from pathlib import Path
from pydrake.all import (
    DiagramBuilder,
    StartMeshcat,
    PointCloud,
    Concatenate,
)
from manipulation.station import LoadScenario, MakeHardwareStation, AddPointClouds

def remove_table_points(pc, table_height=0.0):
    xyz = pc.xyzs()
    mask = xyz[2, :] > table_height
    filtered = xyz[:, mask]
    if filtered.shape[1] == 0:
        return PointCloud(0)
    new_pc = PointCloud(filtered.shape[1])
    new_pc.mutable_xyzs()[:] = filtered
    return new_pc

meshcat = StartMeshcat()
brick_size = [0.10, 0.08, 0.04]

np.random.seed(42)
brick_positions = [
    [-0.30, -0.20, np.random.uniform(0, np.pi)],
    [-0.30, 0.20, np.random.uniform(0, np.pi)],
    [-0.20, 0.0, np.random.uniform(0, np.pi)],
]

# Create SDFs
table_path = Path("/tmp/table_debug.sdf")
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

brick_path = Path("/tmp/brick_debug.sdf")
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

brick_directives = ""
for i, pos in enumerate(brick_positions):
    brick_directives += f"""
- add_model:
    name: brick{i+1}
    file: file://{brick_path.resolve()}
    default_free_body_pose:
        brick_link:
            translation: [{pos[0]}, {pos[1]}, {brick_size[2]/2.0 + 0.001}]
            rotation: !Rpy {{ deg: [0, 0, {np.degrees(pos[2]):.1f}] }}
"""

scenario_yaml = f"""
directives:
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
- add_model:
    name: wsg
    file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.114]
        rotation: !Rpy {{ deg: [90, 0, 90] }}
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
- add_frame:
    name: camera3_origin
    X_PF:
        base_frame: world
        rotation: !Rpy {{ deg: [-120.0, 0.0, 0.0]}}
        translation: [0, -0.8, 0.5]
- add_model:
    name: camera3
    file: package://manipulation/camera_box.sdf
- add_weld:
    parent: camera3_origin
    child: camera3::base
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
    camera3:
        name: camera3
        depth: True
        X_PB:
            base_frame: camera3::base
"""

scenario = LoadScenario(data=scenario_yaml)
builder = DiagramBuilder()
station = MakeHardwareStation(scenario, meshcat)
builder.AddSystem(station)

to_point_cloud = AddPointClouds(scenario=scenario, station=station, builder=builder, meshcat=meshcat)
builder.ExportOutput(to_point_cloud["camera0"].get_output_port(), "camera_point_cloud0")
builder.ExportOutput(to_point_cloud["camera1"].get_output_port(), "camera_point_cloud1")
builder.ExportOutput(to_point_cloud["camera2"].get_output_port(), "camera_point_cloud2")
builder.ExportOutput(to_point_cloud["camera3"].get_output_port(), "camera_point_cloud3")

diagram = builder.Build()
context = diagram.CreateDefaultContext()
diagram.ForcedPublish(context)
plant = station.plant()
plant_context = diagram.GetSubsystemContext(plant, context)

pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
pc1 = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
pc2 = diagram.GetOutputPort("camera_point_cloud2").Eval(context)
pc3 = diagram.GetOutputPort("camera_point_cloud3").Eval(context)

print("=" * 70)
print("POINT COUNT DEBUGGING - brick1")
print("=" * 70)

brick_name = "brick1"
model_brick = plant.GetModelInstanceByName(brick_name)
frame_brick = plant.GetFrameByName("brick_link", model_instance=model_brick)
X_PC_brick = plant.CalcRelativeTransform(plant_context, plant.world_frame(), frame_brick)

brick_lower = X_PC_brick.translation() + np.array([-0.20, -0.20, -0.20])
brick_upper = X_PC_brick.translation() + np.array([0.20, 0.20, 0.20])

camera0_crop = pc0.Crop(brick_lower, brick_upper)
camera1_crop = pc1.Crop(brick_lower, brick_upper)
camera2_crop = pc2.Crop(brick_lower, brick_upper)
camera3_crop = pc3.Crop(brick_lower, brick_upper)

print(f"\n1. Individual camera crops:")
print(f"   Camera 0: {camera0_crop.size()} points")
print(f"   Camera 1: {camera1_crop.size()} points")
print(f"   Camera 2: {camera2_crop.size()} points")
print(f"   Camera 3: {camera3_crop.size()} points")

combined = Concatenate([camera0_crop, camera1_crop, camera2_crop, camera3_crop])
print(f"\n2. Combined (before any processing): {combined.size()} points")

# METHOD A: brick_stacking.py approach (downsample THEN remove table)
downsampled_first = combined.VoxelizedDownSample(0.005)
print(f"\n3A. After downsampling (0.005m): {downsampled_first.size()} points")
brick_cloud_A = remove_table_points(downsampled_first)
print(f"4A. After table removal: {brick_cloud_A.size()} points")
print(f"    ⚠ BRICK_STACKING.PY METHOD: {brick_cloud_A.size()} points")

# METHOD B: verify_icp_centered_bricks.py approach (remove table THEN downsample)
no_table_first = remove_table_points(combined)
print(f"\n3B. After table removal: {no_table_first.size()} points")
downsampled_B = no_table_first.VoxelizedDownSample(0.005)
print(f"4B. After downsampling (0.005m): {downsampled_B.size()} points")
print(f"    ✓ VERIFY_ICP METHOD: {downsampled_B.size()} points")

print(f"\n{'=' * 70}")
print(f"DIAGNOSIS:")
print(f"{'=' * 70}")
print(f"Method A (downsample→table): {brick_cloud_A.size()} points")
print(f"Method B (table→downsample): {downsampled_B.size()} points")
print(f"\nRatio: {downsampled_B.size() / brick_cloud_A.size():.1f}x more points with Method B")
print(f"\nWhy? Downsampling includes table points, which wastes voxel budget.")
print(f"When table is removed first, more voxels are allocated to brick points.")
print(f"\nFIX: Change brick_stacking.py to use Method B (remove table FIRST)")
print(f"{'=' * 70}")
