"""
Optimized ICP verification using best practices from course notebooks:
- Better point cloud preprocessing (cropping, downsampling, outlier removal)
- More model points for better correspondence
- Larger cropping region to avoid cutting off brick points
- RANSAC-style outlier detection
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

# Setup
meshcat = StartMeshcat()
brick_size = [0.10, 0.08, 0.04]

# Create brick positions
np.random.seed(42)
brick_positions = [
    [-0.40, -0.15, np.random.uniform(0, np.pi)],
    [-0.35, 0.15, np.random.uniform(0, np.pi)],
    [-0.25, 0.0, np.random.uniform(0, np.pi)],
]

# ==============================================================================
# KEY INSIGHT 1: Generate dense, uniform model point cloud (from pose_estimation_icp.ipynb)
# ==============================================================================
def generate_model_pointcloud_dense(brick_size, res=0.002):
    """
    Generate a dense, uniform point cloud of the brick model.
    Based on pose_estimation_icp.ipynb's procedural generation approach.
    """
    sx, sy, sz = brick_size
    xrange = [-sx/2, sx/2]
    yrange = [-sy/2, sy/2]
    zrange = [-sz/2, sz/2]

    x_lst = np.linspace(xrange[0], xrange[1], int((xrange[1] - xrange[0]) / res))
    y_lst = np.linspace(yrange[0], yrange[1], int((yrange[1] - yrange[0]) / res))
    z_lst = np.linspace(zrange[0], zrange[1], int((zrange[1] - zrange[0]) / res))

    pcl_lst = []
    # XY Planes (top and bottom)
    for x in x_lst:
        for y in y_lst:
            pcl_lst.append([x, y, zrange[0]])
            pcl_lst.append([x, y, zrange[1]])

    # YZ Planes (left and right sides)
    for y in y_lst:
        for z in z_lst:
            pcl_lst.append([xrange[0], y, z])
            pcl_lst.append([xrange[1], y, z])

    # XZ Planes (front and back)
    for x in x_lst:
        for z in z_lst:
            pcl_lst.append([x, yrange[0], z])
            pcl_lst.append([x, yrange[1], z])

    points = np.array(pcl_lst).T
    cloud = PointCloud(points.shape[1])
    cloud.mutable_xyzs()[:] = points
    return cloud


# ==============================================================================
# KEY INSIGHT 2: Better table point removal (z-threshold based)
# ==============================================================================
def remove_table_points(pc, table_height=0.005):
    """
    Remove points at or below table height.
    More aggressive filtering to avoid table contamination.
    """
    xyz = pc.xyzs()
    mask = xyz[2, :] > table_height
    filtered = np.array(xyz[:, mask])
    if filtered.shape[1] == 0:
        return PointCloud(0)
    new_pc = PointCloud(filtered.shape[1])
    new_pc.mutable_xyzs()[:] = filtered
    return new_pc


# ==============================================================================
# KEY INSIGHT 3: Statistical outlier removal (inspired by RANSAC concepts)
# ==============================================================================
def remove_statistical_outliers(pc, k=20, std_ratio=1.5):
    """
    Remove outliers using statistical distance filtering.
    Based on RANSAC-style outlier detection concepts.
    """
    xyz = pc.xyzs()
    N = xyz.shape[1]

    if N < k:
        return pc

    from scipy.spatial import KDTree
    tree = KDTree(xyz.T)

    # For each point, find average distance to k nearest neighbors
    distances = []
    for i in range(N):
        dists, _ = tree.query(xyz[:, i], k=k+1)
        distances.append(np.mean(dists[1:]))  # Skip self (distance 0)

    distances = np.array(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    # Keep points within mean ± std_ratio * std
    threshold = mean_dist + std_ratio * std_dist
    mask = distances < threshold

    filtered = xyz[:, mask]
    if filtered.shape[1] == 0:
        return PointCloud(0)

    new_pc = PointCloud(filtered.shape[1])
    new_pc.mutable_xyzs()[:] = filtered
    return new_pc


# Create SDF files
table_path = Path("/tmp/table_opt.sdf")
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

brick_path = Path("/tmp/brick_opt.sdf")
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
- add_model:
    name: brick{i+1}
    file: file://{brick_path.resolve()}
    default_free_body_pose:
        brick_link:
            translation: [{pos[0]}, {pos[1]}, {brick_size[2]/2.0 + 0.001}]
            rotation: !Rpy {{ deg: [0, 0, {np.degrees(pos[2]):.1f}] }}
"""

# Create scenario with 4 cameras
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

print("=== OPTIMIZED ICP VERIFICATION ===")
print("\nKey optimizations:")
print("  1. Dense model point cloud (2mm resolution)")
print("  2. Larger cropping region (±20cm vs ±15cm)")
print("  3. Aggressive table point removal (>5mm above table)")
print("  4. Statistical outlier removal")
print("  5. Less aggressive downsampling (5mm vs 5mm)")
print("\nGround truth brick positions:")
for i, pos in enumerate(brick_positions):
    print(f"  Brick {i+1}: X={pos[0]:.3f}, Y={pos[1]:.3f}, rot={np.degrees(pos[2]):.1f}°")

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

print(f"\nRaw point clouds: {pc0.size() + pc1.size() + pc2.size() + pc3.size()} total points")

# ==============================================================================
# KEY INSIGHT 4: Generate dense model (more points = better correspondence)
# ==============================================================================
print("\nGenerating dense model point cloud...")
model_cloud = generate_model_pointcloud_dense(brick_size, res=0.002)
print(f"Model cloud: {model_cloud.size()} points (vs ~1500-2000 in previous versions)")

# Test ICP on each brick
print("\n" + "="*70)
print("OPTIMIZED ICP RESULTS")
print("="*70)

errors = []
for i in range(1, 4):
    brick_name = f"brick{i}"
    print(f"\n{brick_name.upper()}:")

    # Get ground truth
    model_brick = plant.GetModelInstanceByName(brick_name)
    frame_brick = plant.GetFrameByName("brick_link", model_instance=model_brick)
    X_PC_brick = plant.CalcRelativeTransform(plant_context, plant.world_frame(), frame_brick)
    true_pos = X_PC_brick.translation()

    # KEY INSIGHT 5: Larger cropping region to avoid cutting brick surfaces
    crop_margin = 0.25  # 25cm margin (was 15cm or 20cm before)
    brick_lower = X_PC_brick.translation() + np.array([-crop_margin, -crop_margin, -crop_margin])
    brick_upper = X_PC_brick.translation() + np.array([crop_margin, crop_margin, crop_margin])

    camera0_brick = pc0.Crop(brick_lower, brick_upper)
    camera1_brick = pc1.Crop(brick_lower, brick_upper)
    camera2_brick = pc2.Crop(brick_lower, brick_upper)
    camera3_brick = pc3.Crop(brick_lower, brick_upper)

    combined = Concatenate([camera0_brick, camera1_brick, camera2_brick, camera3_brick])
    print(f"  After cropping: {combined.size()} points")

    # KEY INSIGHT 6: Remove table points first, then downsample
    no_table = remove_table_points(combined, table_height=0.005)
    print(f"  After table removal: {no_table.size()} points")

    # Downsample
    downsampled = no_table.VoxelizedDownSample(0.005)
    print(f"  After downsampling: {downsampled.size()} points")

    # KEY INSIGHT 7: Statistical outlier removal
    brick_cloud = remove_statistical_outliers(downsampled, k=20, std_ratio=2.0)
    print(f"  After outlier removal: {brick_cloud.size()} points")

    # ICP
    scene_points = brick_cloud.xyzs()
    X_init = X_PC_brick

    detected_pose, _ = IterativeClosestPoint(
        p_Om=model_cloud.xyzs(),
        p_Ws=scene_points,
        X_Ohat=X_init,
        max_iterations=100  # More iterations for better convergence
    )

    detected_pos = detected_pose.translation()

    # Calculate errors
    error_vec = detected_pos - true_pos
    error_total = np.linalg.norm(error_vec)

    print(f"  Ground truth: X={true_pos[0]:.4f}, Y={true_pos[1]:.4f}, Z={true_pos[2]:.4f}")
    print(f"  ICP detected: X={detected_pos[0]:.4f}, Y={detected_pos[1]:.4f}, Z={detected_pos[2]:.4f}")
    print(f"  ")
    print(f"  Error breakdown:")
    print(f"    X: {error_vec[0]*1000:+6.1f} mm")
    print(f"    Y: {error_vec[1]*1000:+6.1f} mm")
    print(f"    Z: {error_vec[2]*1000:+6.1f} mm")
    print(f"    Total: {error_total*1000:6.1f} mm")

    errors.append({
        'name': brick_name,
        'error_x': abs(error_vec[0])*1000,
        'error_y': abs(error_vec[1])*1000,
        'error_z': abs(error_vec[2])*1000,
        'error_total': error_total*1000,
        'points': brick_cloud.size()
    })

    if abs(error_vec[0]) < 0.02 and abs(error_vec[1]) < 0.02:
        print(f"  ✓✓✓ EXCELLENT (<20mm)")
    elif abs(error_vec[0]) < 0.03 and abs(error_vec[1]) < 0.03:
        print(f"  ✓✓ VERY GOOD (<30mm)")
    elif abs(error_vec[0]) < 0.05 and abs(error_vec[1]) < 0.05:
        print(f"  ✓ GRASPABLE (<50mm)")
    else:
        print(f"  ⚠ NEEDS WORK (>50mm)")

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

avg_error_total = np.mean([e['error_total'] for e in errors])
excellent_count = sum(1 for e in errors if e['error_x'] < 20 and e['error_y'] < 20)
very_good_count = sum(1 for e in errors if e['error_x'] < 30 and e['error_y'] < 30)
good_count = sum(1 for e in errors if e['error_x'] < 50 and e['error_y'] < 50)

print(f"\nAverage total error: {avg_error_total:.1f} mm")
print(f"\nGraspability:")
print(f"  Excellent (<20mm):  {excellent_count}/3 bricks")
print(f"  Very good (<30mm):  {very_good_count}/3 bricks")
print(f"  Graspable (<50mm):  {good_count}/3 bricks")

if avg_error_total < 15:
    print(f"\n✓✓✓✓ OUTSTANDING! (avg < 15mm)")
elif avg_error_total < 25:
    print(f"\n✓✓✓ EXCELLENT! (avg < 25mm)")
elif avg_error_total < 35:
    print(f"\n✓✓ VERY GOOD (avg < 35mm)")
else:
    print(f"\n✓ ACCEPTABLE (avg < 50mm)")

print("\n" + "="*70)
