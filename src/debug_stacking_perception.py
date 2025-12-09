"""
Debug the exact perception phase from brick_stacking.py to see why point counts are low.
"""

import numpy as np
from pathlib import Path
from pydrake.all import (
    Concatenate,
    DiagramBuilder,
    PointCloud,
)
from manipulation.station import LoadScenario, MakeHardwareStation, AddPointClouds
from manipulation.icp import IterativeClosestPoint
import sys
sys.path.append(str(Path(__file__).parent))
from brick_pickplace_clean import (
    write_table_sdf,
    write_brick_sdf,
    sample_brick_surface,
    remove_table_points,
)
from pydrake.all import StartMeshcat

def create_scenario_yaml_multi_brick(table_sdf_path: Path, brick_sdf_path: Path,
                                      brick_size: list[float], brick_positions: list) -> str:
    """Create scenario YAML with multiple bricks."""
    brick_directives = ""
    for i, pos in enumerate(brick_positions):
        brick_directives += f"""
# Add brick {i+1}
- add_model:
    name: brick{i+1}
    file: file://{brick_sdf_path.resolve()}
    default_free_body_pose:
        brick_link:
            translation: [{pos[0]}, {pos[1]}, {brick_size[2]/2.0 + 0.001}]
            rotation: !Rpy {{ deg: [0, 0, {np.degrees(pos[2]):.1f}] }}
"""

    return f"""
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

# Add gripper
- add_model:
    name: wsg
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 90]}}

# Add table
- add_model:
    name: table
    file: file://{table_sdf_path.resolve()}
- add_weld:
    parent: world
    child: table::table_link
    X_PC:
        translation: [0.0, 0.0, -0.05]
        rotation: !Rpy {{ deg: [0, 0, -90] }}
{brick_directives}
# Add cameras
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

# Setup (exactly as in brick_stacking.py)
assets_dir = Path("assets")
assets_dir.mkdir(exist_ok=True)
brick_size = [0.10, 0.08, 0.04]

table_sdf_path = assets_dir / "table.sdf"
write_table_sdf(table_sdf_path)
brick_dir = assets_dir / "brick_model"
brick_dir.mkdir(exist_ok=True)
brick_sdf_path = brick_dir / "brick.sdf"
write_brick_sdf(brick_sdf_path, brick_size)

np.random.seed(42)
brick_positions = [
    [-0.30, -0.20, np.random.uniform(0, np.pi)],
    [-0.30, 0.20, np.random.uniform(0, np.pi)],
    [-0.20, 0.0, np.random.uniform(0, np.pi)],
]

scenario_yaml = create_scenario_yaml_multi_brick(table_sdf_path, brick_sdf_path, brick_size, brick_positions)
scenario = LoadScenario(data=scenario_yaml)

meshcat = StartMeshcat()

# EXACT COPY of perception phase from brick_stacking.py
station = MakeHardwareStation(scenario, meshcat)
builder = DiagramBuilder()
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

# Get point clouds
pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
pc1 = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
pc2 = diagram.GetOutputPort("camera_point_cloud2").Eval(context)
pc3 = diagram.GetOutputPort("camera_point_cloud3").Eval(context)

print("=" * 70)
print("DEBUGGING BRICK_STACKING.PY PERCEPTION PHASE")
print("=" * 70)

print(f"\nTotal raw points:")
print(f"  Camera 0: {pc0.size()}")
print(f"  Camera 1: {pc1.size()}")
print(f"  Camera 2: {pc2.size()}")
print(f"  Camera 3: {pc3.size()}")
print(f"  Total: {pc0.size() + pc1.size() + pc2.size() + pc3.size()}")

# Test brick1 processing
brick_name = "brick1"
model_brick = plant.GetModelInstanceByName(brick_name)
frame_brick = plant.GetFrameByName("brick_link", model_instance=model_brick)
X_PC_brick = plant.CalcRelativeTransform(plant_context, plant.world_frame(), frame_brick)

brick_lower = X_PC_brick.translation() + np.array([-0.20, -0.20, -0.20])
brick_upper = X_PC_brick.translation() + np.array([0.20, 0.20, 0.20])

camera0_brick = pc0.Crop(brick_lower, brick_upper)
camera1_brick = pc1.Crop(brick_lower, brick_upper)
camera2_brick = pc2.Crop(brick_lower, brick_upper)
camera3_brick = pc3.Crop(brick_lower, brick_upper)

print(f"\nBrick1 cropped points:")
print(f"  Camera 0: {camera0_brick.size()}")
print(f"  Camera 1: {camera1_brick.size()}")
print(f"  Camera 2: {camera2_brick.size()}")
print(f"  Camera 3: {camera3_brick.size()}")

combined = Concatenate([camera0_brick, camera1_brick, camera2_brick, camera3_brick])
print(f"  Combined: {combined.size()}")

# BRICK_STACKING.PY METHOD
downsampled = combined.VoxelizedDownSample(0.005)
print(f"  After downsample: {downsampled.size()}")
brick_cloud = remove_table_points(downsampled)
print(f"  After table removal: {brick_cloud.size()}")

print(f"\n{'=' * 70}")
print(f"COMPARISON:")
print(f"{'=' * 70}")
print(f"Expected (from verify_icp_centered_bricks.py): 7532 points")
print(f"Actual (from brick_stacking.py recreation): {brick_cloud.size()} points")
print(f"\nIf these match, the issue is somewhere else!")
print(f"{'=' * 70}")
