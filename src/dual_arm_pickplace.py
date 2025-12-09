"""
Dual-Arm Pick and Place
Two IIWA robots picking and placing bricks simultaneously
"""

import numpy as np
from pathlib import Path
from pydrake.all import (
    Concatenate,
    DiagramBuilder,
    Integrator,
    JacobianWrtVariable,
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
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddRgbdSensors
from pydrake.perception import PointCloud

# Import helper functions from brick_pickplace_clean
import sys
sys.path.append(str(Path(__file__).parent))
from brick_pickplace_clean import (
    write_table_sdf,
    write_brick_sdf,
    design_grasp_pose,
    design_pregrasp_pose,
    design_pregoal_pose,
    design_postgoal_pose,
    make_trajectory,
)


class DualArmController(LeafSystem):
    """Pseudoinverse controller for dual-arm setup."""
    def __init__(self, plant: MultibodyPlant, iiwa_name: str, gripper_name: str):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName(iiwa_name)
        self._gripper_model = plant.GetModelInstanceByName(gripper_name)
        self._G = plant.GetBodyByName("body", self._gripper_model).body_frame()
        self._W = plant.world_frame()

        # Get the velocity start index for this robot's joints
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1", self._iiwa).velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7", self._iiwa).velocity_start()

        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("iiwa.position", 7)
        self.DeclareVectorOutputPort("iiwa.velocity", 7, self.CalcOutput)

    def CalcOutput(self, context, output):
        V_G_desired = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)

        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._G,
            [0, 0, 0],
            self._W,
            self._W
        )
        # Extract only the columns for this robot's joints
        J_G_robot = J_G[:, self.iiwa_start:self.iiwa_end+1]
        v = np.linalg.pinv(J_G_robot).dot(V_G_desired)
        output.SetFromVector(v)


def create_dual_arm_scenario(table_sdf_path: Path, brick_sdf_path: Path,
                               brick_size: list[float],
                               brick1_pos: list, brick2_pos: list,
                               goal1_pos: list, goal2_pos: list) -> str:
    """Create scenario YAML with two IIWA arms and two bricks."""
    return f"""
directives:
# Add IIWA robot 1 (left arm)
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

# Add gripper for left arm
- add_model:
    name: wsg_left
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa_left::iiwa_link_7
    child: wsg_left::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 90]}}

# Add IIWA robot 2 (right arm)
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

# Add gripper for right arm
- add_model:
    name: wsg_right
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa_right::iiwa_link_7
    child: wsg_right::body
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

# Add brick 1 (for left arm)
- add_model:
    name: brick1
    file: file://{brick_sdf_path.resolve()}
    default_free_body_pose:
        brick_link:
            translation: [{brick1_pos[0]}, {brick1_pos[1]}, {brick_size[2]/2.0 + 0.001}]
            rotation: !Rpy {{ deg: [0, 0, {np.degrees(brick1_pos[2]):.1f}] }}

# Add brick 2 (for right arm)
- add_model:
    name: brick2
    file: file://{brick_sdf_path.resolve()}
    default_free_body_pose:
        brick_link:
            translation: [{brick2_pos[0]}, {brick2_pos[1]}, {brick_size[2]/2.0 + 0.001}]
            rotation: !Rpy {{ deg: [0, 0, {np.degrees(brick2_pos[2]):.1f}] }}

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

model_drivers:
    iiwa_left: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg_left
    wsg_left: !SchunkWsgDriver {{}}
    iiwa_right: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg_right
    wsg_right: !SchunkWsgDriver {{}}
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
"""


def create_pickplace_trajectory(brick_pose, goal_pos, brick_size, plant, station_context, gripper_model_name):
    """Create trajectory to pick and place one brick."""
    opened = 0.107
    closed = 0.0

    # Get initial gripper pose
    plant_context = plant.GetMyContextFromRoot(station_context)
    gripper_model = plant.GetModelInstanceByName(gripper_model_name)
    X_WGinitial = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("body", gripper_model))

    # Design grasp for brick
    X_OG, X_WGpick = design_grasp_pose(brick_pose)
    X_WGprepick = design_pregrasp_pose(X_WGpick)

    # Goal pose
    goal_z = brick_size[2]/2.0 + 0.001
    X_WOgoal = RigidTransform(
        brick_pose.rotation(),
        np.array([goal_pos[0], goal_pos[1], goal_z])
    )
    X_WGgoal = X_WOgoal @ X_OG
    X_WGpregoal = design_pregoal_pose(X_WGgoal)
    X_WGpostgoal = design_postgoal_pose(X_WGgoal)

    # Waypoints
    all_poses = [
        X_WGinitial,    # Start
        X_WGprepick,    # Approach
        X_WGpick,       # Pick (open)
        X_WGpick,       # Pick (close)
        X_WGpregoal,    # Lift with brick
        X_WGgoal,       # Goal (closed)
        X_WGgoal,       # Goal (open)
        X_WGpostgoal,   # Lift after release
        X_WGinitial,    # Return
    ]
    all_fingers = [opened, opened, opened, closed, closed, closed, opened, opened, opened]

    # Create time samples (3 seconds per waypoint)
    sample_times = [3 * i for i in range(len(all_poses))]
    finger_array = np.array(all_fingers).reshape(1, -1)

    return make_trajectory(all_poses, finger_array, sample_times)


if __name__ == "__main__":
    print("=== DUAL-ARM PICK AND PLACE ===\n")

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

    # Brick positions (in front of each robot)
    # Left robot at [-0.5, -0.5], right robot at [0.5, -0.5]
    # Bricks should be forward (positive Y direction) from robot base
    # Optimal workspace: 0.4-0.65m from base (avoid too close and too far)
    np.random.seed(42)
    brick1_pos = [-0.50, -0.10, np.random.uniform(0, np.pi)]  # Left arm's brick (0.4m forward)
    brick2_pos = [0.50, -0.10, np.random.uniform(0, np.pi)]   # Right arm's brick (0.4m forward)

    # Goal positions (0.65m from base - good reach without full extension)
    goal1_pos = [-0.50, 0.15]  # Left arm goal (0.65m from base)
    goal2_pos = [0.50, 0.15]   # Right arm goal (0.65m from base)

    print(f"Left arm:")
    print(f"  Brick 1 initial: [{brick1_pos[0]:.2f}, {brick1_pos[1]:.2f}], rot={np.degrees(brick1_pos[2]):.1f}°")
    print(f"  Goal 1: [{goal1_pos[0]:.2f}, {goal1_pos[1]:.2f}]")
    print(f"\nRight arm:")
    print(f"  Brick 2 initial: [{brick2_pos[0]:.2f}, {brick2_pos[1]:.2f}], rot={np.degrees(brick2_pos[2]):.1f}°")
    print(f"  Goal 2: [{goal2_pos[0]:.2f}, {goal2_pos[1]:.2f}]\n")

    # Create scenario
    scenario_yaml = create_dual_arm_scenario(table_sdf_path, brick_sdf_path, brick_size,
                                              brick1_pos, brick2_pos, goal1_pos, goal2_pos)
    scenario = LoadScenario(data=scenario_yaml)

    meshcat = StartMeshcat()

    # ========================================================================
    # PERCEPTION: Get ground truth brick poses
    # ========================================================================
    print("=== PERCEPTION PHASE ===")

    station = MakeHardwareStation(scenario, meshcat)
    builder = DiagramBuilder()
    builder.AddSystem(station)

    to_point_cloud = AddPointClouds(scenario=scenario, station=station, builder=builder, meshcat=meshcat)
    builder.ExportOutput(to_point_cloud["camera0"].get_output_port(), "camera_point_cloud0")
    builder.ExportOutput(to_point_cloud["camera1"].get_output_port(), "camera_point_cloud1")

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)
    plant = station.plant()
    plant_context = diagram.GetSubsystemContext(plant, context)

    # Get point clouds (for visualization)
    pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
    pc1 = diagram.GetOutputPort("camera_point_cloud1").Eval(context)

    print(f"Point clouds captured: {pc0.size() + pc1.size()} total points")

    # Get ground truth brick poses
    model_brick1 = plant.GetModelInstanceByName("brick1")
    frame_brick1 = plant.GetFrameByName("brick_link", model_instance=model_brick1)
    brick1_pose = plant.CalcRelativeTransform(plant_context, plant.world_frame(), frame_brick1)

    model_brick2 = plant.GetModelInstanceByName("brick2")
    frame_brick2 = plant.GetFrameByName("brick_link", model_instance=model_brick2)
    brick2_pose = plant.CalcRelativeTransform(plant_context, plant.world_frame(), frame_brick2)

    print(f"brick1: Ground truth pose: {brick1_pose.translation()}")
    print(f"brick2: Ground truth pose: {brick2_pose.translation()}")

    # ========================================================================
    # CONTROL: Simultaneous pick and place for both arms
    # ========================================================================
    print("\n=== CONTROL PHASE: DUAL-ARM PICK AND PLACE ===")

    builder2 = DiagramBuilder()
    station2 = MakeHardwareStation(scenario, meshcat)
    builder2.AddSystem(station2)
    plant2 = station2.plant()

    station_context = station2.CreateDefaultContext()

    # Create trajectories for both arms
    traj_V_G_left, traj_wsg_left = create_pickplace_trajectory(
        brick1_pose, goal1_pos, brick_size, plant2, station_context, "wsg_left"
    )

    traj_V_G_right, traj_wsg_right = create_pickplace_trajectory(
        brick2_pose, goal2_pos, brick_size, plant2, station_context, "wsg_right"
    )

    # Wire up LEFT arm controller
    V_src_left = builder2.AddSystem(TrajectorySource(traj_V_G_left))
    controller_left = builder2.AddSystem(DualArmController(plant2, "iiwa_left", "wsg_left"))
    integrator_left = builder2.AddSystem(Integrator(7))
    wsg_src_left = builder2.AddSystem(TrajectorySource(traj_wsg_left))

    builder2.Connect(V_src_left.get_output_port(), controller_left.get_input_port(0))
    builder2.Connect(controller_left.get_output_port(), integrator_left.get_input_port())
    builder2.Connect(integrator_left.get_output_port(), station2.GetInputPort("iiwa_left.position"))
    builder2.Connect(station2.GetOutputPort("iiwa_left.position_measured"), controller_left.get_input_port(1))
    builder2.Connect(wsg_src_left.get_output_port(), station2.GetInputPort("wsg_left.position"))

    # Wire up RIGHT arm controller
    V_src_right = builder2.AddSystem(TrajectorySource(traj_V_G_right))
    controller_right = builder2.AddSystem(DualArmController(plant2, "iiwa_right", "wsg_right"))
    integrator_right = builder2.AddSystem(Integrator(7))
    wsg_src_right = builder2.AddSystem(TrajectorySource(traj_wsg_right))

    builder2.Connect(V_src_right.get_output_port(), controller_right.get_input_port(0))
    builder2.Connect(controller_right.get_output_port(), integrator_right.get_input_port())
    builder2.Connect(integrator_right.get_output_port(), station2.GetInputPort("iiwa_right.position"))
    builder2.Connect(station2.GetOutputPort("iiwa_right.position_measured"), controller_right.get_input_port(1))
    builder2.Connect(wsg_src_right.get_output_port(), station2.GetInputPort("wsg_right.position"))

    # Add force limits for grippers
    try:
        builder2.Connect(
            builder2.AddSystem(ConstantVectorSource(np.array([200.0]))).get_output_port(),
            station2.GetInputPort("wsg_left.force_limit")
        )
        builder2.Connect(
            builder2.AddSystem(ConstantVectorSource(np.array([200.0]))).get_output_port(),
            station2.GetInputPort("wsg_right.force_limit")
        )
    except:
        pass

    diagram2 = builder2.Build()
    sim2 = Simulator(diagram2)
    ctx2 = sim2.get_mutable_context()

    # Initialize integrators
    q0_left = plant2.GetPositions(plant2.GetMyContextFromRoot(ctx2), plant2.GetModelInstanceByName("iiwa_left"))
    integrator_left.set_integral_value(integrator_left.GetMyContextFromRoot(ctx2), q0_left)

    q0_right = plant2.GetPositions(plant2.GetMyContextFromRoot(ctx2), plant2.GetModelInstanceByName("iiwa_right"))
    integrator_right.set_integral_value(integrator_right.GetMyContextFromRoot(ctx2), q0_right)

    # Run simulation
    final_time = max(traj_V_G_left.end_time(), traj_V_G_right.end_time()) + 1.0
    print(f"Simulating dual-arm pick and place ({final_time:.1f}s)...")
    print(f"Both arms operating simultaneously!")

    sim2.set_target_realtime_rate(1.0)
    sim2.Initialize()
    sim2.AdvanceTo(final_time)

    print("\n=== DUAL-ARM PICK AND PLACE COMPLETE ===")
    print("✓ Both arms completed their tasks simultaneously!")
    print(f"✓ Check MeshCat at http://localhost:7000")
