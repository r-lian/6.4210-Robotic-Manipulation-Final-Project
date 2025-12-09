"""Debug Jacobian indexing with kV vs kQDot"""
import numpy as np
from pathlib import Path
from pydrake.all import (
    DiagramBuilder,
    RigidTransform,
    JacobianWrtVariable,
)
from manipulation.station import LoadScenario, MakeHardwareStation
import sys
sys.path.append(str(Path(__file__).parent))
from brick_pickplace_clean import write_table_sdf, write_brick_sdf

brick_size = [0.10, 0.08, 0.04]

# Create SDFs
assets_dir = Path("assets")
assets_dir.mkdir(exist_ok=True)
table_sdf_path = assets_dir / "table.sdf"
write_table_sdf(table_sdf_path)
brick_dir = assets_dir / "brick_model"
brick_dir.mkdir(exist_ok=True)
brick_sdf_path = brick_dir / "brick.sdf"
write_brick_sdf(brick_sdf_path, brick_size)

scenario_yaml = f"""
directives:
- add_model:
    name: iiwa_left
    file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
- add_weld:
    parent: world
    child: iiwa_left::iiwa_link_0
    X_PC:
        translation: [-0.5, -0.5, 0]
- add_model:
    name: wsg_left
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa_left::iiwa_link_7
    child: wsg_left::body
    X_PC:
        translation: [0, 0, 0.09]
- add_model:
    name: iiwa_right
    file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
- add_weld:
    parent: world
    child: iiwa_right::iiwa_link_0
    X_PC:
        translation: [0.5, -0.5, 0]
- add_model:
    name: wsg_right
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa_right::iiwa_link_7
    child: wsg_right::body
    X_PC:
        translation: [0, 0, 0.09]
model_drivers:
    iiwa_left: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg_left
    wsg_left: !SchunkWsgDriver {{}}
    iiwa_right: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg_right
    wsg_right: !SchunkWsgDriver {{}}
"""

scenario = LoadScenario(data=scenario_yaml)
station = MakeHardwareStation(scenario)
builder = DiagramBuilder()
builder.AddSystem(station)
diagram = builder.Build()

context = diagram.CreateDefaultContext()
plant = station.plant()
plant_context = diagram.GetSubsystemContext(plant, context)

# Check left arm
iiwa_left = plant.GetModelInstanceByName("iiwa_left")
wsg_left = plant.GetModelInstanceByName("wsg_left")
G_left = plant.GetBodyByName("body", wsg_left).body_frame()
W = plant.world_frame()

joint1_left = plant.GetJointByName("iiwa_joint_1", iiwa_left)
joint7_left = plant.GetJointByName("iiwa_joint_7", iiwa_left)

print("=" * 70)
print("LEFT ARM JOINT INDICES")
print("=" * 70)
print(f"iiwa_joint_1 velocity_start: {joint1_left.velocity_start()}")
print(f"iiwa_joint_7 velocity_start: {joint7_left.velocity_start()}")
print(f"Extracting columns: [{joint1_left.velocity_start()}:{joint7_left.velocity_start()+1}]")

# Check right arm
iiwa_right = plant.GetModelInstanceByName("iiwa_right")
wsg_right = plant.GetModelInstanceByName("wsg_right")
G_right = plant.GetBodyByName("body", wsg_right).body_frame()

joint1_right = plant.GetJointByName("iiwa_joint_1", iiwa_right)
joint7_right = plant.GetJointByName("iiwa_joint_7", iiwa_right)

print("\n" + "=" * 70)
print("RIGHT ARM JOINT INDICES")
print("=" * 70)
print(f"iiwa_joint_1 velocity_start: {joint1_right.velocity_start()}")
print(f"iiwa_joint_7 velocity_start: {joint7_right.velocity_start()}")
print(f"Extracting columns: [{joint1_right.velocity_start()}:{joint7_right.velocity_start()+1}]")

# Test Jacobian with kV
print("\n" + "=" * 70)
print("JACOBIAN WITH kV (generalized velocities)")
print("=" * 70)

J_G_left_V = plant.CalcJacobianSpatialVelocity(
    plant_context,
    JacobianWrtVariable.kV,
    G_left,
    [0, 0, 0],
    W,
    W
)
print(f"Left arm Jacobian shape: {J_G_left_V.shape}")

J_G_right_V = plant.CalcJacobianSpatialVelocity(
    plant_context,
    JacobianWrtVariable.kV,
    G_right,
    [0, 0, 0],
    W,
    W
)
print(f"Right arm Jacobian shape: {J_G_right_V.shape}")

# Test Jacobian with kQDot
print("\n" + "=" * 70)
print("JACOBIAN WITH kQDot (coordinate time derivatives)")
print("=" * 70)

J_G_left_QDot = plant.CalcJacobianSpatialVelocity(
    plant_context,
    JacobianWrtVariable.kQDot,
    G_left,
    [0, 0, 0],
    W,
    W
)
print(f"Left arm Jacobian shape: {J_G_left_QDot.shape}")

J_G_right_QDot = plant.CalcJacobianSpatialVelocity(
    plant_context,
    JacobianWrtVariable.kQDot,
    G_right,
    [0, 0, 0],
    W,
    W
)
print(f"Right arm Jacobian shape: {J_G_right_QDot.shape}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"Total DOF (nv): {plant.num_velocities()}")
print(f"Total positions (nq): {plant.num_positions()}")
print("If kV and kQDot give same shape → they're equivalent for this system")
print("If shapes differ → velocity indexing needs adjustment")
print("=" * 70)
