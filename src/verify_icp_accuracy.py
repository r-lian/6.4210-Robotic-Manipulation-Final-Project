"""Verify ICP accuracy - detailed analysis of errors."""

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
import sys
sys.path.append(str(Path(__file__).parent))
from brick_stacking import sample_brick_surface, detect_brick_with_icp, create_scenario_yaml_multi_brick, write_table_sdf, write_brick_sdf

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

# Create SDFs
assets_dir = Path("assets")
assets_dir.mkdir(exist_ok=True)
table_sdf_path = assets_dir / "table.sdf"
write_table_sdf(table_sdf_path)
brick_dir = assets_dir / "brick_model"
brick_dir.mkdir(exist_ok=True)
brick_sdf_path = brick_dir / "brick.sdf"
write_brick_sdf(brick_sdf_path, brick_size)

# Create scenario
scenario_yaml = create_scenario_yaml_multi_brick(table_sdf_path, brick_sdf_path, brick_size, brick_positions)
scenario = LoadScenario(data=scenario_yaml)

print("=== VERIFYING ICP ACCURACY ===")
print(f"\nGround truth brick positions:")
for i, pos in enumerate(brick_positions):
    print(f"  Brick {i+1}: X={pos[0]:.3f}, Y={pos[1]:.3f}, rot={np.degrees(pos[2]):.1f}°")

# Build diagram
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

print(f"\nPoint clouds: {pc0.size()}, {pc1.size()}, {pc2.size()} points")

# Test ICP detection on each brick
print("\n" + "="*70)
print("DETAILED ICP ACCURACY ANALYSIS")
print("="*70)

errors = []
for i in range(1, 4):
    brick_name = f"brick{i}"
    print(f"\n{brick_name.upper()}:")

    # Get ground truth pose
    brick_body = plant.GetBodyByName("brick_link", plant.GetModelInstanceByName(brick_name))
    true_pose = plant.EvalBodyPoseInWorld(plant_context, brick_body)
    true_pos = true_pose.translation()

    # Detect with ICP
    detected_pose = detect_brick_with_icp(pc0, pc1, pc2, plant, plant_context, brick_name, brick_size, meshcat)
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
        'error_total': error_total*1000
    })

    # Check if graspable
    # Gripper is 110mm wide, so we need to be within ~50mm in X and Y
    if abs(error_x) < 0.05 and abs(error_y) < 0.05:
        print(f"  ✓ GRASPABLE (X,Y errors < 50mm)")
    else:
        print(f"  ⚠ QUESTIONABLE (X or Y error > 50mm)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

avg_error_x = np.mean([e['error_x'] for e in errors])
avg_error_y = np.mean([e['error_y'] for e in errors])
avg_error_z = np.mean([e['error_z'] for e in errors])
avg_error_total = np.mean([e['error_total'] for e in errors])

print(f"\nAverage errors across all 3 bricks:")
print(f"  X: {avg_error_x:.1f} mm")
print(f"  Y: {avg_error_y:.1f} mm")
print(f"  Z: {avg_error_z:.1f} mm")
print(f"  Total: {avg_error_total:.1f} mm")

print(f"\nGripper specifications:")
print(f"  Max opening: 110 mm")
print(f"  Brick length: {brick_size[0]*1000:.0f} mm")
print(f"  Brick width:  {brick_size[1]*1000:.0f} mm")

print(f"\nGraspability assessment:")
graspable_count = sum(1 for e in errors if e['error_x'] < 50 and e['error_y'] < 50)
print(f"  Bricks likely graspable: {graspable_count}/3")

if avg_error_total < 30:
    print(f"\n✓✓ EXCELLENT: ICP is working very well (avg error < 30mm)")
elif avg_error_total < 50:
    print(f"\n✓ GOOD: ICP is working well enough for grasping (avg error < 50mm)")
elif avg_error_total < 70:
    print(f"\n⚠ ACCEPTABLE: ICP errors are high but might still work (avg error < 70mm)")
else:
    print(f"\n✗ POOR: ICP errors too large, grasping will likely fail (avg error > 70mm)")

print("\n" + "="*70)
