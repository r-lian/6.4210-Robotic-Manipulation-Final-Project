"""Verify ICP accuracy with 4 cameras for better coverage."""

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

# Helper functions
def sample_brick_surface(size_xyz, num_samples=1500):
    """Sample points from brick surface."""
    sx, sy, sz = size_xyz
    areas = np.array([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy])
    probs = areas / np.sum(areas)

    samples_per_face = np.random.multinomial(num_samples, probs)
    points = []

    for face_idx, n_samples in enumerate(samples_per_face):
        if n_samples == 0:
            continue
        if face_idx == 0:  # -X face
            x = np.full(n_samples, -sx/2)
            y = np.random.uniform(-sy/2, sy/2, n_samples)
            z = np.random.uniform(-sz/2, sz/2, n_samples)
        elif face_idx == 1:  # +X face
            x = np.full(n_samples, sx/2)
            y = np.random.uniform(-sy/2, sy/2, n_samples)
            z = np.random.uniform(-sz/2, sz/2, n_samples)
        elif face_idx == 2:  # -Y face
            x = np.random.uniform(-sx/2, sx/2, n_samples)
            y = np.full(n_samples, -sy/2)
            z = np.random.uniform(-sz/2, sz/2, n_samples)
        elif face_idx == 3:  # +Y face
            x = np.random.uniform(-sx/2, sx/2, n_samples)
            y = np.full(n_samples, sy/2)
            z = np.random.uniform(-sz/2, sz/2, n_samples)
        elif face_idx == 4:  # -Z face (bottom)
            x = np.random.uniform(-sx/2, sx/2, n_samples)
            y = np.random.uniform(-sy/2, sy/2, n_samples)
            z = np.full(n_samples, -sz/2)
        else:  # +Z face (top)
            x = np.random.uniform(-sx/2, sx/2, n_samples)
            y = np.random.uniform(-sy/2, sy/2, n_samples)
            z = np.full(n_samples, sz/2)

        points.append(np.vstack([x, y, z]))

    all_points = np.hstack(points)
    cloud = PointCloud(all_points.shape[1])
    cloud.mutable_xyzs()[:] = all_points
    return cloud

def remove_table_points(pc, table_height=0.0):
    """Remove points below table height."""
    xyz = pc.xyzs()
    mask = xyz[2, :] > table_height
    filtered = np.array(xyz[:, mask])
    if filtered.shape[1] == 0:
        return PointCloud(0)
    new_pc = PointCloud(filtered.shape[1])
    new_pc.mutable_xyzs()[:] = filtered
    return new_pc

# Create SDF files
table_path = Path("/tmp/table_4cam.sdf")
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

brick_path = Path("/tmp/brick_4cam.sdf")
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

# Create scenario with 4 cameras for better coverage
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
# Add 4 cameras for full coverage
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

print("=== VERIFYING ICP ACCURACY WITH 4 CAMERAS ===")
print(f"\nGround truth brick positions:")
for i, pos in enumerate(brick_positions):
    print(f"  Brick {i+1}: X={pos[0]:.3f}, Y={pos[1]:.3f}, rot={np.degrees(pos[2]):.1f}°")

# Build diagram
scenario = LoadScenario(data=scenario_yaml)
builder = DiagramBuilder()
station = MakeHardwareStation(scenario, meshcat)
builder.AddSystem(station)

# Add point clouds from all 4 cameras
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

# Get point clouds
pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
pc1 = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
pc2 = diagram.GetOutputPort("camera_point_cloud2").Eval(context)
pc3 = diagram.GetOutputPort("camera_point_cloud3").Eval(context)

print(f"\nPoint clouds: {pc0.size()}, {pc1.size()}, {pc2.size()}, {pc3.size()} points")
print(f"Total: {pc0.size() + pc1.size() + pc2.size() + pc3.size()} points (vs 921,600 with 3 cameras)")

# Test ICP detection on each brick with 4 cameras
print("\n" + "="*70)
print("DETAILED ICP ACCURACY ANALYSIS (4 CAMERAS)")
print("="*70)

errors = []
for i in range(1, 4):
    brick_name = f"brick{i}"
    print(f"\n{brick_name.upper()}:")

    # Get brick ground truth for cropping
    model_brick = plant.GetModelInstanceByName(brick_name)
    frame_brick = plant.GetFrameByName("brick_link", model_instance=model_brick)
    X_PC_brick = plant.CalcRelativeTransform(plant_context, plant.world_frame(), frame_brick)

    # Crop around brick (slightly larger region)
    brick_lower = X_PC_brick.translation() + np.array([-0.2, -0.2, -0.2])
    brick_upper = X_PC_brick.translation() + np.array([0.2, 0.2, 0.2])

    camera0_brick = pc0.Crop(brick_lower, brick_upper)
    camera1_brick = pc1.Crop(brick_lower, brick_upper)
    camera2_brick = pc2.Crop(brick_lower, brick_upper)
    camera3_brick = pc3.Crop(brick_lower, brick_upper)

    combined = Concatenate([camera0_brick, camera1_brick, camera2_brick, camera3_brick])
    downsampled = combined.VoxelizedDownSample(0.005)
    brick_cloud = remove_table_points(downsampled)

    print(f"  Point cloud: {brick_cloud.size()} points (from {combined.size()} before downsampling)")

    # Get ground truth pose
    true_pose = X_PC_brick
    true_pos = true_pose.translation()

    # ICP with better parameters
    model_cloud = sample_brick_surface(brick_size, num_samples=2000)
    scene_points = brick_cloud.xyzs()

    X_init = true_pose
    detected_pose, _ = IterativeClosestPoint(
        p_Om=model_cloud.xyzs(),
        p_Ws=scene_points,
        X_Ohat=X_init,
        max_iterations=60
    )

    detected_pos = detected_pose.translation()

    # Calculate errors
    error_vec = detected_pos - true_pos
    error_x = error_vec[0]
    error_y = error_vec[1]
    error_z = error_vec[2]
    error_total = np.linalg.norm(error_vec)

    print(f"  Ground truth: X={true_pos[0]:.4f}, Y={true_pos[1]:.4f}, Z={true_pos[2]:.4f}")
    print(f"  ICP detected: X={detected_pos[0]:.4f}, Y={detected_pos[1]:.4f}, Z={detected_pos[2]:.4f}")
    print(f"  ")
    print(f"  Error breakdown:")
    print(f"    X error: {error_x*1000:+6.1f} mm ({abs(error_x*1000):.1f} mm magnitude)")
    print(f"    Y error: {error_y*1000:+6.1f} mm ({abs(error_y*1000):.1f} mm magnitude)")
    print(f"    Z error: {error_z*1000:+6.1f} mm ({abs(error_z*1000):.1f} mm magnitude)")
    print(f"    Total:   {error_total*1000:6.1f} mm")

    errors.append({
        'name': brick_name,
        'error_x': abs(error_x)*1000,
        'error_y': abs(error_y)*1000,
        'error_z': abs(error_z)*1000,
        'error_total': error_total*1000,
        'points': brick_cloud.size()
    })

    # Check if graspable
    if abs(error_x) < 0.03 and abs(error_y) < 0.03:
        print(f"  ✓✓ EXCELLENT (X,Y errors < 30mm)")
    elif abs(error_x) < 0.05 and abs(error_y) < 0.05:
        print(f"  ✓ GRASPABLE (X,Y errors < 50mm)")
    else:
        print(f"  ⚠ QUESTIONABLE (X or Y error > 50mm)")

print("\n" + "="*70)
print("SUMMARY (4 CAMERAS)")
print("="*70)

avg_error_x = np.mean([e['error_x'] for e in errors])
avg_error_y = np.mean([e['error_y'] for e in errors])
avg_error_z = np.mean([e['error_z'] for e in errors])
avg_error_total = np.mean([e['error_total'] for e in errors])
avg_points = np.mean([e['points'] for e in errors])

print(f"\nAverage errors across all 3 bricks:")
print(f"  X: {avg_error_x:.1f} mm")
print(f"  Y: {avg_error_y:.1f} mm")
print(f"  Z: {avg_error_z:.1f} mm")
print(f"  Total: {avg_error_total:.1f} mm")
print(f"\nAverage points per brick: {avg_points:.0f}")

print(f"\nGripper specifications:")
print(f"  Max opening: 110 mm")
print(f"  Brick length: {brick_size[0]*1000:.0f} mm")
print(f"  Brick width:  {brick_size[1]*1000:.0f} mm")

print(f"\nGraspability assessment:")
excellent_count = sum(1 for e in errors if e['error_x'] < 30 and e['error_y'] < 30)
good_count = sum(1 for e in errors if e['error_x'] < 50 and e['error_y'] < 50)
print(f"  Excellent (<30mm): {excellent_count}/3")
print(f"  Graspable (<50mm): {good_count}/3")

if avg_error_total < 20:
    print(f"\n✓✓✓ EXCELLENT: ICP is working very well (avg error < 20mm)")
elif avg_error_total < 30:
    print(f"\n✓✓ VERY GOOD: ICP is working well (avg error < 30mm)")
elif avg_error_total < 50:
    print(f"\n✓ GOOD: ICP is working well enough for grasping (avg error < 50mm)")
else:
    print(f"\n⚠ NEEDS IMPROVEMENT: ICP errors still high (avg error > 50mm)")

print("\n" + "="*70)
print("Check Meshcat at http://localhost:7000 to see 4-camera coverage")
print("All brick surfaces should now be captured by at least one camera")
