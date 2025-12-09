"""
Simplified Dual-Arm Pick and Place (no cameras)
Test left arm grasping with ground truth poses only
"""

import numpy as np
from pathlib import Path
from pydrake.all import (
    DiagramBuilder,
    Integrator,
    Simulator,
    StartMeshcat,
    TrajectorySource,
)
from manipulation.station import LoadScenario, MakeHardwareStation
import sys
sys.path.append(str(Path(__file__).parent))
from dual_arm_pickplace import DualArmController, create_dual_arm_scenario, create_pickplace_trajectory
from brick_pickplace_clean import write_table_sdf, write_brick_sdf

print("=== SIMPLIFIED DUAL-ARM TEST (LEFT ARM ONLY, NO CAMERAS) ===\n")

# Setup
assets_dir = Path("assets")
assets_dir.mkdir(exist_ok=True)
brick_size = [0.10, 0.08, 0.04]

# Create SDFs
table_sdf_path = assets_dir / "table.sdf"
write_table_sdf(table_sdf_path)
brick_dir = assets_dir / "brick_model"
brick_dir.mkdir(exist_ok=True)
brick_sdf_path = brick_dir / "brick.sdf"
write_brick_sdf(brick_sdf_path, brick_size)

np.random.seed(42)
brick1_pos = [-0.50, -0.20, np.random.uniform(0, np.pi)]
brick2_pos = [0.50, -0.20, np.random.uniform(0, np.pi)]
goal1_pos = [-0.50, 0.20]
goal2_pos = [0.50, 0.20]

print(f"Left arm: Brick at [{brick1_pos[0]:.2f}, {brick1_pos[1]:.2f}], goal at [{goal1_pos[0]:.2f}, {goal1_pos[1]:.2f}]")
print(f"Right arm: Brick at [{brick2_pos[0]:.2f}, {brick2_pos[1]:.2f}], goal at [{goal2_pos[0]:.2f}, {goal2_pos[1]:.2f}]\n")

# Create scenario WITHOUT cameras
scenario_yaml = f"""
directives:
- add_model:
    name: iiwa_left
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
    child: iiwa_left::iiwa_link_0
    X_PC:
        translation: [-0.5, -0.5, 0]
        rotation: !Rpy {{ deg: [0, 0, 180] }}
- add_model:
    name: wsg_left
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa_left::iiwa_link_7
    child: wsg_left::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 90]}}
- add_model:
    name: iiwa_right
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
    child: iiwa_right::iiwa_link_0
    X_PC:
        translation: [0.5, -0.5, 0]
        rotation: !Rpy {{ deg: [0, 0, 180] }}
- add_model:
    name: wsg_right
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa_right::iiwa_link_7
    child: wsg_right::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 90]}}
- add_model:
    name: table
    file: file://{table_sdf_path.resolve()}
- add_weld:
    parent: world
    child: table::table_link
    X_PC:
        translation: [0.0, 0.0, -0.05]
        rotation: !Rpy {{ deg: [0, 0, -90] }}
- add_model:
    name: brick1
    file: file://{brick_sdf_path.resolve()}
    default_free_body_pose:
        brick_link:
            translation: [{brick1_pos[0]}, {brick1_pos[1]}, {brick_size[2]/2.0 + 0.001}]
            rotation: !Rpy {{ deg: [0, 0, {np.degrees(brick1_pos[2]):.1f}] }}
- add_model:
    name: brick2
    file: file://{brick_sdf_path.resolve()}
    default_free_body_pose:
        brick_link:
            translation: [{brick2_pos[0]}, {brick2_pos[1]}, {brick_size[2]/2.0 + 0.001}]
            rotation: !Rpy {{ deg: [0, 0, {np.degrees(brick2_pos[2]):.1f}] }}
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
meshcat = StartMeshcat()

# Get ground truth brick poses
print("Getting ground truth brick poses...")
station = MakeHardwareStation(scenario, meshcat)
builder = DiagramBuilder()
builder.AddSystem(station)
diagram = builder.Build()
context = diagram.CreateDefaultContext()
plant = station.plant()
plant_context = diagram.GetSubsystemContext(plant, context)

model_brick1 = plant.GetModelInstanceByName("brick1")
frame_brick1 = plant.GetFrameByName("brick_link", model_instance=model_brick1)
brick1_pose = plant.CalcRelativeTransform(plant_context, plant.world_frame(), frame_brick1)

model_brick2 = plant.GetModelInstanceByName("brick2")
frame_brick2 = plant.GetFrameByName("brick_link", model_instance=model_brick2)
brick2_pose = plant.CalcRelativeTransform(plant_context, plant.world_frame(), frame_brick2)

print(f"brick1 pose: {brick1_pose.translation()}")
print(f"brick2 pose: {brick2_pose.translation()}\n")

# Create control diagram
print("Creating control system...")
builder2 = DiagramBuilder()
station2 = MakeHardwareStation(scenario, meshcat)
builder2.AddSystem(station2)
plant2 = station2.plant()
station_context = station2.CreateDefaultContext()

# Create trajectories
traj_V_G_left, traj_wsg_left = create_pickplace_trajectory(
    brick1_pose, goal1_pos, brick_size, plant2, station_context, "wsg_left"
)
traj_V_G_right, traj_wsg_right = create_pickplace_trajectory(
    brick2_pose, goal2_pos, brick_size, plant2, station_context, "wsg_right"
)

# Wire up controllers
V_src_left = builder2.AddSystem(TrajectorySource(traj_V_G_left))
controller_left = builder2.AddSystem(DualArmController(plant2, "iiwa_left", "wsg_left"))
integrator_left = builder2.AddSystem(Integrator(7))
wsg_src_left = builder2.AddSystem(TrajectorySource(traj_wsg_left))

builder2.Connect(V_src_left.get_output_port(), controller_left.get_input_port(0))
builder2.Connect(controller_left.get_output_port(), integrator_left.get_input_port())
builder2.Connect(integrator_left.get_output_port(), station2.GetInputPort("iiwa_left.position"))
builder2.Connect(station2.GetOutputPort("iiwa_left.position_measured"), controller_left.get_input_port(1))
builder2.Connect(wsg_src_left.get_output_port(), station2.GetInputPort("wsg_left.position"))

V_src_right = builder2.AddSystem(TrajectorySource(traj_V_G_right))
controller_right = builder2.AddSystem(DualArmController(plant2, "iiwa_right", "wsg_right"))
integrator_right = builder2.AddSystem(Integrator(7))
wsg_src_right = builder2.AddSystem(TrajectorySource(traj_wsg_right))

builder2.Connect(V_src_right.get_output_port(), controller_right.get_input_port(0))
builder2.Connect(controller_right.get_output_port(), integrator_right.get_input_port())
builder2.Connect(integrator_right.get_output_port(), station2.GetInputPort("iiwa_right.position"))
builder2.Connect(station2.GetOutputPort("iiwa_right.position_measured"), controller_right.get_input_port(1))
builder2.Connect(wsg_src_right.get_output_port(), station2.GetInputPort("wsg_right.position"))

diagram2 = builder2.Build()
simulator = Simulator(diagram2)
simulator.set_target_realtime_rate(1.0)

print("Running simulation...")
print("Check Meshcat at http://localhost:7000\n")

simulator.AdvanceTo(27.0)  # 9 waypoints * 3 seconds

print("\n=== SIMULATION COMPLETE ===")
print("Dual-arm pick and place finished!")
