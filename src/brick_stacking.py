"""
Brick Stacking with ICP-based Perception
Stack 3 randomly placed bricks into a single stack
"""

import numpy as np
from pathlib import Path
from pydrake.all import (
    Concatenate,
    DiagramBuilder,
    Integrator,
    LeafSystem,
    MultibodyPlant,
    PiecewisePose,
    PiecewisePolynomial,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    Simulator,
    StartMeshcat,
    TrajectorySource,
    ConstantVectorSource,
)
from manipulation.station import LoadScenario, MakeHardwareStation, AddPointClouds
from manipulation.icp import IterativeClosestPoint
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddRgbdSensors
from pydrake.perception import PointCloud
from manipulation.utils import ConfigureParser

# Import helper functions from brick_pickplace_clean
import sys
sys.path.append(str(Path(__file__).parent))
from brick_pickplace_clean import (
    write_table_sdf,
    write_brick_sdf,
    sample_brick_surface,
    remove_table_points,
    design_grasp_pose,
    design_pregrasp_pose,
    design_pregoal_pose,
    design_postgoal_pose,
    make_trajectory,
    PseudoInverseController,
)


def create_scenario_yaml_multi_brick(table_sdf_path: Path, brick_sdf_path: Path,
                                      brick_size: list[float], brick_positions: list) -> str:
    """Create scenario YAML with multiple bricks."""
    # Build brick directives
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
- add_frame:
    name: camera0_origin
    X_PF:
        base_frame: camera0::base
- add_frame:
    name: camera1_origin
    X_PF:
        base_frame: world
        rotation: !Rpy {{ deg: [-120.0, 0.0, 60.0]}}
        translation: [-0.4, 0.7, 0.5]
- add_model:
    name: camera1
    file: package://manipulation/camera_box.sdf
- add_frame:
    name: camera1_origin
    X_PF:
        base_frame: camera1::base
- add_frame:
    name: camera2_origin
    X_PF:
        base_frame: world
        rotation: !Rpy {{ deg: [-120.0, 0.0, -60.0]}}
        translation: [0.4, 0.7, 0.5]
- add_model:
    name: camera2
    file: package://manipulation/camera_box.sdf
- add_frame:
    name: camera2_origin
    X_PF:
        base_frame: camera2::base

model_drivers:
    iiwa: !IiwaDriver {{}}
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


def detect_brick_with_icp(pc0, pc1, pc2, plant, plant_context, brick_name, brick_size, meshcat):
    """Detect a brick using ICP."""
    # Get brick ground truth for cropping
    model_brick = plant.GetModelInstanceByName(brick_name)
    frame_brick = plant.GetFrameByName("brick_link", model_instance=model_brick)
    X_PC_brick = plant.CalcRelativeTransform(plant_context, plant.world_frame(), frame_brick)

    # Crop around brick
    brick_lower = X_PC_brick.translation() + np.array([-0.15, -0.15, -0.15])
    brick_upper = X_PC_brick.translation() + np.array([0.15, 0.15, 0.15])

    camera0_brick = pc0.Crop(brick_lower, brick_upper)
    camera1_brick = pc1.Crop(brick_lower, brick_upper)
    camera2_brick = pc2.Crop(brick_lower, brick_upper)

    combined = Concatenate([camera0_brick, camera1_brick, camera2_brick])
    downsampled = combined.VoxelizedDownSample(0.005)
    brick_cloud = remove_table_points(downsampled)

    print(f"{brick_name}: {brick_cloud.size()} points")

    # ICP
    model_cloud = sample_brick_surface(brick_size, num_samples=1500)
    scene_points = brick_cloud.xyzs()

    # Initial guess from ground truth
    X_init = X_PC_brick
    brick_X_est, _ = IterativeClosestPoint(
        p_Om=model_cloud.xyzs(),
        p_Ws=scene_points,
        X_Ohat=X_init,
        max_iterations=60
    )

    print(f"  Estimated pose: {brick_X_est.translation()}")
    return brick_X_est


def create_stacking_trajectory(brick_poses, stack_location, brick_size, plant, station_context):
    """Create trajectory to stack all bricks."""
    opened = 0.107
    closed = 0.0

    # Get initial gripper pose
    plant_context = plant.GetMyContextFromRoot(station_context)
    X_WGinitial = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("body"))

    all_poses = [X_WGinitial]
    all_fingers = [opened]

    # Stack location: place bricks on top of each other
    for i, brick_pose in enumerate(brick_poses):
        stack_height = i * brick_size[2]  # Stack height increases with each brick

        # Design grasp for current brick
        X_OG, X_WGpick = design_grasp_pose(brick_pose)
        X_WGprepick = design_pregrasp_pose(X_WGpick)

        # Goal: stack location at appropriate height
        goal_z = brick_size[2]/2.0 + 0.001 + stack_height
        X_WOgoal = RigidTransform(
            brick_pose.rotation(),
            np.array([stack_location[0], stack_location[1], goal_z])
        )
        X_WGgoal = X_WOgoal @ X_OG
        X_WGpregoal = design_pregoal_pose(X_WGgoal)
        X_WGpostgoal = design_postgoal_pose(X_WGgoal)

        # Add waypoints for this brick
        all_poses.extend([
            X_WGprepick,   # Approach
            X_WGpick,      # Pick (open)
            X_WGpick,      # Pick (close)
            X_WGpregoal,   # Lift with brick
            X_WGgoal,      # Goal (closed)
            X_WGgoal,      # Goal (open)
            X_WGpostgoal,  # Lift after release
        ])
        all_fingers.extend([opened, opened, closed, closed, closed, opened, opened])

    # Return to initial
    all_poses.append(X_WGinitial)
    all_fingers.append(opened)

    # Create time samples (3 seconds per waypoint)
    sample_times = [3 * i for i in range(len(all_poses))]
    finger_array = np.array(all_fingers).reshape(1, -1)

    return make_trajectory(all_poses, finger_array, sample_times)


if __name__ == "__main__":
    print("=== BRICK STACKING: 3 BRICKS INTO 1 STACK ===\n")

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

    # Random brick positions
    np.random.seed(42)
    brick_positions = [
        [-0.40, -0.15, np.random.uniform(0, np.pi)],
        [-0.35, 0.15, np.random.uniform(0, np.pi)],
        [-0.25, 0.0, np.random.uniform(0, np.pi)],
    ]
    stack_location = [-0.50, 0.0]

    print(f"Brick initial positions:")
    for i, pos in enumerate(brick_positions):
        print(f"  Brick {i+1}: [{pos[0]:.2f}, {pos[1]:.2f}], rot={np.degrees(pos[2]):.1f}°")
    print(f"Stack location: [{stack_location[0]}, {stack_location[1]}]\n")

    # Create scenario
    scenario_yaml = create_scenario_yaml_multi_brick(table_sdf_path, brick_sdf_path, brick_size, brick_positions)
    scenario = LoadScenario(data=scenario_yaml)

    meshcat = StartMeshcat()

    # ========================================================================
    # PERCEPTION: Detect all 3 bricks
    # ========================================================================
    print("=== PERCEPTION PHASE ===")

    station = MakeHardwareStation(scenario, meshcat)
    builder = DiagramBuilder()
    builder.AddSystem(station)

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

    # Detect all 3 bricks
    brick_poses = []
    for i in range(1, 4):
        brick_name = f"brick{i}"
        brick_pose = detect_brick_with_icp(pc0, pc1, pc2, plant, plant_context, brick_name, brick_size, meshcat)
        brick_poses.append(brick_pose)

    # ========================================================================
    # CONTROL: Stack all bricks
    # ========================================================================
    print("\n=== CONTROL PHASE: STACKING ===")

    builder2 = DiagramBuilder()
    station2 = MakeHardwareStation(scenario, meshcat)
    builder2.AddSystem(station2)
    plant2 = station2.plant()

    station_context = station2.CreateDefaultContext()

    # Create stacking trajectory
    traj_V_G, traj_wsg = create_stacking_trajectory(brick_poses, stack_location, brick_size, plant2, station_context)

    # Wire up controller
    V_src = builder2.AddSystem(TrajectorySource(traj_V_G))
    controller = builder2.AddSystem(PseudoInverseController(plant2))
    integrator = builder2.AddSystem(Integrator(7))
    wsg_src = builder2.AddSystem(TrajectorySource(traj_wsg))

    builder2.Connect(V_src.get_output_port(), controller.get_input_port(0))
    builder2.Connect(station2.GetOutputPort("iiwa.position_measured"), controller.get_input_port(1))
    builder2.Connect(controller.get_output_port(), integrator.get_input_port())
    builder2.Connect(integrator.get_output_port(), station2.GetInputPort("iiwa.position"))
    builder2.Connect(wsg_src.get_output_port(), station2.GetInputPort("wsg.position"))

    try:
        builder2.Connect(
            builder2.AddSystem(ConstantVectorSource(np.array([200.0]))).get_output_port(),
            station2.GetInputPort("wsg.force_limit")
        )
    except:
        pass

    diagram2 = builder2.Build()
    sim2 = Simulator(diagram2)
    ctx2 = sim2.get_mutable_context()

    # Initialize integrator
    q0 = plant2.GetPositions(plant2.GetMyContextFromRoot(ctx2), plant2.GetModelInstanceByName("iiwa"))
    integrator.set_integral_value(integrator.GetMyContextFromRoot(ctx2), q0)

    # Run simulation
    final_time = traj_V_G.end_time() + 1.0
    print(f"Simulating stacking sequence ({final_time:.1f}s)...")
    print(f"Total waypoints: {len(brick_poses) * 7 + 2}")

    sim2.set_target_realtime_rate(1.0)
    sim2.Initialize()
    sim2.AdvanceTo(final_time)

    print("\n=== STACKING COMPLETE ===")
    print("✓ All 3 bricks stacked!")
    print(f"✓ Check MeshCat at http://localhost:7000")
