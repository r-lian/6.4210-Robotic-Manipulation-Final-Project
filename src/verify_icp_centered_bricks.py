"""
Test ICP with bricks positioned in the center of all 4 camera views.

Key insight: Current brick positions at X∈[-0.40, -0.25] may be outside
optimal camera FOV. Let's test with bricks closer to workspace center.

Camera positions:
- Camera 0: [0, 0.8, 0.5] pointing toward origin
- Camera 1: [0.8, 0.1, 0.5] pointing toward origin
- Camera 2: [-0.8, 0.1, 0.5] pointing toward origin
- Camera 3: [0, -0.8, 0.5] pointing toward origin

Optimal brick workspace: centered around [-0.3, 0.0] with better spread
"""

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
from manipulation.icp import IterativeClosestPoint

meshcat = StartMeshcat()
brick_size = [0.10, 0.08, 0.04]

# NEW: Position bricks in better view of all 4 cameras
# Previous: X∈[-0.40, -0.25], Y∈[-0.15, 0.15] (too far left, poor camera coverage)
# New: X∈[-0.35, -0.20], Y∈[-0.20, 0.20] (more centered, better spread)
np.random.seed(42)
brick_positions = [
    [-0.30, -0.20, np.random.uniform(0, np.pi)],  # Bottom left
    [-0.30, 0.20, np.random.uniform(0, np.pi)],   # Top left
    [-0.20, 0.0, np.random.uniform(0, np.pi)],    # Center right
]

print("=== TESTING CENTERED BRICK POSITIONS ===")
print("\nOLD positions (poor coverage):")
print("  Brick 1: [-0.40, -0.15]")
print("  Brick 2: [-0.35, 0.15]")
print("  Brick 3: [-0.25, 0.0]")
print("\nNEW positions (better coverage):")
for i, pos in enumerate(brick_positions, 1):
    print(f"  Brick {i}: [{pos[0]:.2f}, {pos[1]:.2f}]")
print("\nGoal: Improve average error from 97mm to <50mm\n")

def sample_brick_surface(size_xyz, num_samples=1500):
    sx, sy, sz = size_xyz
    areas = np.array([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy])
    probs = areas / np.sum(areas)
    samples_per_face = np.random.multinomial(num_samples, probs)
    points = []

    for face_idx, n_samples in enumerate(samples_per_face):
        if n_samples == 0:
            continue
        if face_idx == 0:
            x = np.full(n_samples, -sx/2)
            y = np.random.uniform(-sy/2, sy/2, n_samples)
            z = np.random.uniform(-sz/2, sz/2, n_samples)
        elif face_idx == 1:
            x = np.full(n_samples, sx/2)
            y = np.random.uniform(-sy/2, sy/2, n_samples)
            z = np.random.uniform(-sz/2, sz/2, n_samples)
        elif face_idx == 2:
            x = np.random.uniform(-sx/2, sx/2, n_samples)
            y = np.full(n_samples, -sy/2)
            z = np.random.uniform(-sz/2, sz/2, n_samples)
        elif face_idx == 3:
            x = np.random.uniform(-sx/2, sx/2, n_samples)
            y = np.full(n_samples, sy/2)
            z = np.random.uniform(-sz/2, sz/2, n_samples)
        elif face_idx == 4:
            x = np.random.uniform(-sx/2, sx/2, n_samples)
            y = np.random.uniform(-sy/2, sy/2, n_samples)
            z = np.full(n_samples, -sz/2)
        else:
            x = np.random.uniform(-sx/2, sx/2, n_samples)
            y = np.random.uniform(-sy/2, sy/2, n_samples)
            z = np.full(n_samples, sz/2)
        points.append(np.vstack([x, y, z]))

    all_points = np.hstack(points)
    cloud = PointCloud(all_points.shape[1])
    cloud.mutable_xyzs()[:] = all_points
    return cloud

def remove_table_points(pc, table_height=0.0):
    xyz = pc.xyzs()
    mask = xyz[2, :] > table_height
    filtered = xyz[:, mask]
    if filtered.shape[1] == 0:
        return PointCloud(0)
    new_pc = PointCloud(filtered.shape[1])
    new_pc.mutable_xyzs()[:] = filtered
    return new_pc

# Create SDFs
table_path = Path("/tmp/table_centered.sdf")
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

brick_path = Path("/tmp/brick_centered.sdf")
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
# 4 cameras for full coverage
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
print("ICP RESULTS (CENTERED BRICK POSITIONS)")
print("=" * 70)

errors = []
for i in range(1, 4):
    brick_name = f"brick{i}"
    print(f"\n{brick_name.upper()}:")

    model_brick = plant.GetModelInstanceByName(brick_name)
    frame_brick = plant.GetFrameByName("brick_link", model_instance=model_brick)
    X_PC_brick = plant.CalcRelativeTransform(plant_context, plant.world_frame(), frame_brick)
    true_pos = X_PC_brick.translation()

    # Crop around brick
    brick_lower = X_PC_brick.translation() + np.array([-0.20, -0.20, -0.20])
    brick_upper = X_PC_brick.translation() + np.array([0.20, 0.20, 0.20])

    combined = Concatenate([
        pc0.Crop(brick_lower, brick_upper),
        pc1.Crop(brick_lower, brick_upper),
        pc2.Crop(brick_lower, brick_upper),
        pc3.Crop(brick_lower, brick_upper)
    ])

    no_table = remove_table_points(combined)
    downsampled = no_table.VoxelizedDownSample(0.005)

    print(f"  Scene points: {downsampled.size()}")

    # ICP
    model_cloud = sample_brick_surface(brick_size, num_samples=2000)
    scene_points = downsampled.xyzs()
    X_init = X_PC_brick

    detected_pose, _ = IterativeClosestPoint(
        p_Om=model_cloud.xyzs(),
        p_Ws=scene_points,
        X_Ohat=X_init,
        max_iterations=60
    )

    detected_pos = detected_pose.translation()
    error_vec = detected_pos - true_pos
    error_total = np.linalg.norm(error_vec)

    print(f"  Ground truth: [{true_pos[0]:.4f}, {true_pos[1]:.4f}, {true_pos[2]:.4f}]")
    print(f"  ICP detected: [{detected_pos[0]:.4f}, {detected_pos[1]:.4f}, {detected_pos[2]:.4f}]")
    print(f"  Error: X={error_vec[0]*1000:+.1f}mm, Y={error_vec[1]*1000:+.1f}mm, Z={error_vec[2]*1000:+.1f}mm")
    print(f"  Total: {error_total*1000:.1f}mm", end="")

    errors.append({
        'total': error_total*1000,
        'x': abs(error_vec[0])*1000,
        'y': abs(error_vec[1])*1000
    })

    if error_total < 0.015:
        print(" ✓✓✓ EXCELLENT")
    elif error_total < 0.030:
        print(" ✓✓ VERY GOOD")
    elif error_total < 0.050:
        print(" ✓ GRASPABLE")
    else:
        print(" ⚠ POOR")

avg_error = np.mean([e['total'] for e in errors])
excellent = sum(1 for e in errors if e['x'] < 20 and e['y'] < 20)
very_good = sum(1 for e in errors if e['x'] < 30 and e['y'] < 30)
graspable = sum(1 for e in errors if e['x'] < 50 and e['y'] < 50)

print(f"\n{'=' * 70}")
print(f"SUMMARY")
print(f"{'=' * 70}")
print(f"Average error: {avg_error:.1f}mm")
print(f"\nGraspability breakdown:")
print(f"  Excellent (<20mm): {excellent}/3 bricks")
print(f"  Very good (<30mm): {very_good}/3 bricks")
print(f"  Graspable (<50mm): {graspable}/3 bricks")

if avg_error < 20:
    print(f"\n✓✓✓✓ OUTSTANDING! Centered positions work great!")
elif avg_error < 30:
    print(f"\n✓✓✓ EXCELLENT! Major improvement!")
elif avg_error < 50:
    print(f"\n✓✓ GOOD! Significant improvement!")
else:
    print(f"\n✓ Positioning didn't help much")

print(f"\nComparison to old positions:")
print(f"  Old ([-0.40,-0.25] X range): ~97mm average error")
print(f"  New ([-0.30,-0.20] X range): {avg_error:.1f}mm average error")
print(f"  Improvement: {97 - avg_error:.1f}mm ({(97 - avg_error)/97 * 100:.1f}%)")
print(f"{'=' * 70}")
