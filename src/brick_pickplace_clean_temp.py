"""
Brick Pick and Place with Geometry
Based on 11_pickplace_initials_with_geometry.ipynb
"""
import numpy as np
from pathlib import Path
from pydrake.all import (
    AddFrameTriadIllustration,
    BasicVector,
    Concatenate,
    Context,
    DiagramBuilder,
    Integrator,
    JacobianWrtVariable,
    LeafSystem,
    MultibodyPlant,
    PiecewisePolynomial,
    PiecewisePose,
    PointCloud,
    Rgba,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    TrajectorySource,
)
from manipulation.station import LoadScenario, MakeHardwareStation, AddPointClouds
from manipulation.icp import IterativeClosestPoint
from manipulation.meshcat_utils import AddMeshcatTriad


# ============================================================================
# Part 1: Setup - Create SDFs and Scenario
# ============================================================================

def write_table_sdf(table_path: Path) -> None:
    """Create table SDF matching the notebook."""
    table_path.write_text(
        """<?xml version="1.0"?>
<sdf version="1.7">
    <model name="table">
        <pose>0 0 0 0 0 0</pose>
        <link name="table_link">
            <inertial>
                <mass>1.0</mass>
                <inertia>
                    <ixx>0.005833</ixx><ixy>0.0</ixy><ixz>0.0</ixz>
                    <iyy>0.005833</iyy><iyz>0.0</iyz>
                    <izz>0.005</izz>
                </inertia>
            </inertial>
            <collision name="collision">
                <geometry><box><size>2 2 0.1</size></box></geometry>
            </collision>
            <visual name="visual">
                <geometry><box><size>2 2 0.1</size></box></geometry>
                <material><diffuse>1.0 1.0 1.0 1.0</diffuse></material>
            </visual>
        </link>
    </model>
</sdf>
"""
    )


def write_brick_sdf(brick_path: Path, size_xyz: list[float]) -> None:
    """Create brick SDF with high friction like the notebook's letters."""
    sx, sy, sz = size_xyz
    brick_path.write_text(
        f"""<?xml version="1.0"?>
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
</sdf>
"""
    )


def create_scenario_yaml(table_sdf_path: Path, brick_sdf_path: Path, brick_size: list[float]) -> str:
    """Create complete scenario YAML matching notebook structure."""
    return f"""
directives:
# Add IIWA robot (matching notebook positioning)
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

# Add brick (matching letter position in notebook: [-0.35, 0, 0])
- add_model:
    name: brick
    file: file://{brick_sdf_path.resolve()}
    default_free_body_pose:
        brick_link:
            translation: [-0.35, 0, {brick_size[2]/2.0 + 0.01}]
            rotation: !Rpy {{ deg: [0, 0, 0] }}

# Add cameras (matching notebook)
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

model_drivers:
    iiwa: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg
    wsg: !SchunkWsgDriver {{}}
"""


# ============================================================================
# Part 2: Perception - Get Model Point Cloud and Process Scene
# ============================================================================

def sample_brick_surface(size_xyz, num_samples=1500) -> PointCloud:
    """Sample points from brick surface (similar to trimesh.sample in notebook)."""
    sx, sy, sz = size_xyz
    areas = np.array([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy])
    probs = areas / np.sum(areas)
    face_ids = np.random.choice(6, size=num_samples, p=probs)
    u = np.random.uniform(-0.5, 0.5, size=num_samples)
    v = np.random.uniform(-0.5, 0.5, size=num_samples)
    pts = np.zeros((3, num_samples))

    m = face_ids == 0
    pts[:, m] = np.vstack([np.full(np.sum(m), +sx / 2), sy * u[m], sz * v[m]])
    m = face_ids == 1
    pts[:, m] = np.vstack([np.full(np.sum(m), -sx / 2), sy * u[m], sz * v[m]])
    m = face_ids == 2
    pts[:, m] = np.vstack([sx * u[m], np.full(np.sum(m), +sy / 2), sz * v[m]])
    m = face_ids == 3
    pts[:, m] = np.vstack([sx * u[m], np.full(np.sum(m), -sy / 2), sz * v[m]])
    m = face_ids == 4
    pts[:, m] = np.vstack([sx * u[m], sy * v[m], np.full(np.sum(m), +sz / 2)])
    m = face_ids == 5
    pts[:, m] = np.vstack([sx * u[m], sy * v[m], np.full(np.sum(m), -sz / 2)])

    pc = PointCloud(num_samples)
    pc.mutable_xyzs()[:] = pts
    return pc


def remove_table_points(point_cloud: PointCloud, z_threshold=0.01) -> PointCloud:
    """Remove table points (matching notebook's approach)."""
    xyz = point_cloud.xyzs()
    keep = xyz[2, :] >= z_threshold
    cloud = PointCloud(int(np.sum(keep)))
    cloud.mutable_xyzs()[:] = xyz[:, keep]
    return cloud


# ============================================================================
# Part 3: Grasp Design (adapted from notebook for brick)
# ============================================================================

def design_grasp_pose(X_WO: RigidTransform) -> tuple[RigidTransform, RigidTransform]:
    """Design grasp pose - EXACTLY matching notebook's approach."""
    # Notebook uses: RollPitchYaw(0, 0, π) @ RollPitchYaw(-π/2, 0, 0)
    R_OG = (
        RollPitchYaw(0, 0, np.pi).ToRotationMatrix()
        @ RollPitchYaw(-np.pi / 2, 0, 0).ToRotationMatrix()
    )
    # Notebook offset for letter 'A': [0.07, 0.06, 0.12]
    # Adjust for brick (smaller, simpler shape)
    p_OG = [0.04, 0.0, 0.10]
    X_OG = RigidTransform(R_OG, p_OG)
    X_WG = X_WO @ X_OG
    return X_OG, X_WG


def design_pregrasp_pose(X_WG: RigidTransform) -> RigidTransform:
    """Pre-grasp approach (matching notebook)."""
    X_GGApproach = RigidTransform([0.0, -0.2, 0.0])
    return X_WG @ X_GGApproach


def design_pregoal_pose(X_WG: RigidTransform) -> RigidTransform:
    """Pre-goal hover (matching notebook)."""
    X_GGApproach = RigidTransform([0.0, 0.0, -0.2])
    return X_WG @ X_GGApproach


def design_goal_poses(X_WO: RigidTransform, X_OG: RigidTransform) -> RigidTransform:
    """Goal pose - move brick to new location."""
    # Move brick forward and to the side (away from robot)
    X_WOgoal = X_WO @ RigidTransform(
        R=RotationMatrix.Identity(),
        p=np.array([-0.2, 0.3, 0.0])
    )
    X_WGgoal = X_WOgoal @ X_OG
    return X_WGgoal


def design_postgoal_pose(X_WG: RigidTransform) -> RigidTransform:
    """Post-goal lift (matching notebook)."""
    X_GGApproach = RigidTransform([0.0, 0.0, -0.2])
    return X_WG @ X_GGApproach


def make_trajectory(X_Gs: list[RigidTransform], finger_values: np.ndarray,
                    sample_times: list[float]) -> tuple:
    """Create trajectory (matching notebook)."""
    robot_position_trajectory = PiecewisePose.MakeLinear(sample_times, X_Gs)
    robot_velocity_trajectory = robot_position_trajectory.MakeDerivative()
    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(sample_times, finger_values)
    return robot_velocity_trajectory, traj_wsg_command


# ============================================================================
# Part 4: Controller (copied from notebook)
# ============================================================================

class PseudoInverseController(LeafSystem):
    """Pseudoinverse controller from notebook."""
    def __init__(self, plant: MultibodyPlant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("iiwa.position", 7)
        self.DeclareVectorOutputPort("iiwa.velocity", 7, self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context: Context, output: BasicVector):
        V_G = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kV,
            self._G,
            [0, 0, 0],
            self._W,
            self._W
        )
        J_G = J_G[:, self.iiwa_start:self.iiwa_end + 1]
        v = np.linalg.pinv(J_G) @ V_G
        output.SetFromVector(v)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=== BRICK PICK AND PLACE WITH GEOMETRY ===\n")

    # Setup paths and parameters
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    brick_size = [0.16, 0.08, 0.04]  # Standard brick dimensions

    # Create SDFs
    table_sdf_path = assets_dir / "table.sdf"
    write_table_sdf(table_sdf_path)
    brick_dir = assets_dir / "brick_model"
    brick_dir.mkdir(exist_ok=True)
    brick_sdf_path = brick_dir / "brick.sdf"
    write_brick_sdf(brick_sdf_path, brick_size)

    # Create scenario
    scenario_yaml = create_scenario_yaml(table_sdf_path, brick_sdf_path, brick_size)
    scenario = LoadScenario(data=scenario_yaml)

    # Start meshcat
    print("Starting MeshCat...")
    meshcat = StartMeshcat()

    # ========================================================================
    # PERCEPTION PHASE
    # ========================================================================
    print("\n=== PERCEPTION PHASE ===")

    # Build hardware station with cameras
    station = MakeHardwareStation(scenario, meshcat)
    builder = DiagramBuilder()
    builder.AddSystem(station)

    # Add point clouds
    to_point_cloud = AddPointClouds(scenario=scenario, station=station,
                                   builder=builder, meshcat=meshcat)
    builder.ExportOutput(to_point_cloud["camera0"].get_output_port(), "camera_point_cloud0")
    builder.ExportOutput(to_point_cloud["camera1"].get_output_port(), "camera_point_cloud1")
    builder.ExportOutput(to_point_cloud["camera2"].get_output_port(), "camera_point_cloud2")

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)

    # Get point clouds from cameras
    pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
    pc1 = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
    pc2 = diagram.GetOutputPort("camera_point_cloud2").Eval(context)

    # Concatenate and process (matching notebook)
    combined_pc = Concatenate([pc0, pc1, pc2])
    print(f"Merged point cloud: {combined_pc.size()} points")

    # Crop to brick region (using cheat port for bounding box like notebook)
    plant = station.plant()
    plant_context = diagram.GetSubsystemContext(plant, context)
    world_frame = plant.world_frame()
    model_brick = plant.GetModelInstanceByName("brick")
    frame_brick = plant.GetFrameByName("brick_link", model_instance=model_brick)
    X_PC_brick = plant.CalcRelativeTransform(plant_context, world_frame, frame_brick)

    # Crop around brick (matching notebook's approach)
    brick_lower = X_PC_brick.translation() + np.array([-0.15, -0.15, -0.15])
    brick_upper = X_PC_brick.translation() + np.array([0.15, 0.15, 0.15])

    camera0_brick_pc = pc0.Crop(brick_lower, brick_upper)
    camera1_brick_pc = pc1.Crop(brick_lower, brick_upper)
    camera2_brick_pc = pc2.Crop(brick_lower, brick_upper)

    combined_brick = Concatenate([camera0_brick_pc, camera1_brick_pc, camera2_brick_pc])
    downsampled = combined_brick.VoxelizedDownSample(0.005)

    # Remove table points
    brick_point_cloud = remove_table_points(downsampled)
    print(f"Brick point cloud: {brick_point_cloud.size()} points")

    # Visualize
    meshcat.SetObject("brick_point_cloud", brick_point_cloud,
                     point_size=0.05, rgba=Rgba(1, 0, 0))

    # Get model point cloud
    model_cloud = sample_brick_surface(brick_size, num_samples=1500)
    model_points = model_cloud.xyzs()
    scene_points = brick_point_cloud.xyzs()

    # Run ICP (matching notebook)
    print("\nRunning ICP...")
    X_brick_initial = X_PC_brick  # Use cheat port as initial guess like notebook
    brick_X_Ohat, brick_error = IterativeClosestPoint(
        p_Om=model_points,
        p_Ws=scene_points,
        X_Ohat=X_brick_initial,
        meshcat=meshcat,
        max_iterations=70
    )

    print(f"ICP complete")
    print(f"Brick pose: {brick_X_Ohat.translation()}")

    # ========================================================================
    # CONTROL PHASE
    # ========================================================================
    print("\n=== CONTROL PHASE ===")

    # Rebuild diagram for control
    builder = DiagramBuilder()
    station = MakeHardwareStation(scenario, meshcat)
    builder.AddSystem(station)
    plant = station.GetSubsystemByName("plant")
    builder.AddSystem(station)
    plant = station.GetSubsystemByName("plant")

    station_context = station.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(station_context)

    # Get initial gripper pose
    X_WGinitial = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("body"))

    # Design trajectory (matching notebook pattern)
    X_WOinitial = brick_X_Ohat
    X_OG, X_WGpick = design_grasp_pose(X_WOinitial)
    X_WGprepick = design_pregrasp_pose(X_WGpick)
    X_WGgoal = design_goal_poses(X_WOinitial, X_OG)
    X_WGpregoal = design_pregoal_pose(X_WGgoal)
    X_WGpostgoal = design_postgoal_pose(X_WGgoal)

    # Build keyframes
    opened = 0.107
    closed = 0.0

    keyframes = [
        ("X_WGinitial", X_WGinitial, opened),
        ("X_WGprepick", X_WGprepick, opened),
        ("X_WGpick", X_WGpick, opened),
        ("X_WGpick", X_WGpick, closed),
        ("X_WGpregoal", X_WGpregoal, closed),
        ("X_WGgoal", X_WGgoal, closed),
        ("X_WGgoal", X_WGgoal, opened),
        ("X_WGpostgoal", X_WGpostgoal, opened),
        ("X_WGinitial", X_WGinitial, opened),
    ]

    gripper_poses = [kf[1] for kf in keyframes]
    finger_states = np.asarray([kf[2] for kf in keyframes]).reshape(1, -1)
    sample_times = [3 * i for i in range(len(gripper_poses))]
    traj_V_G, traj_wsg_command = make_trajectory(gripper_poses, finger_states, sample_times)

    # Add systems
    V_G_source = builder.AddSystem(TrajectorySource(traj_V_G))
    controller = builder.AddSystem(PseudoInverseController(plant))
    integrator = builder.AddSystem(Integrator(7))
    wsg_source = builder.AddSystem(TrajectorySource(traj_wsg_command))

    # Connect (matching notebook)
    builder.Connect(V_G_source.get_output_port(), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(), integrator.get_input_port())
    builder.Connect(integrator.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(station.GetOutputPort("iiwa.position_measured"), controller.get_input_port(1))
    builder.Connect(wsg_source.get_output_port(), station.GetInputPort("wsg.position"))

    # Visualize
    scenegraph = station.GetSubsystemByName("scene_graph")
    AddFrameTriadIllustration(scene_graph=scenegraph, body=plant.GetBodyByName("brick_link"), length=0.1)
    AddFrameTriadIllustration(scene_graph=scenegraph, body=plant.GetBodyByName("body"), length=0.1)

    diagram = builder.Build()

    # Run simulation
    print("\nRunning simulation...")
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    station_context = station.GetMyContextFromRoot(context)
    integrator.set_integral_value(
        integrator.GetMyContextFromRoot(context),
        plant.GetPositions(
            plant.GetMyContextFromRoot(context),
            plant.GetModelInstanceByName("iiwa"),
        ),
    )

    diagram.ForcedPublish(context)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(traj_V_G.end_time())

    print("\n=== COMPLETE ===")
    print("Check MeshCat visualization!")
    builder = DiagramBuilder()
    station = MakeHardwareStation(scenario, meshcat)
    builder.AddSystem(station)
    plant = station.GetSubsystemByName("plant")

    station_context = station.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(station_context)
