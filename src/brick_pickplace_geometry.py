from pydrake.all import (
    InverseKinematics,
    Solve,
    StartMeshcat,
    DiagramBuilder,
    Simulator,
    RigidTransform,
    RotationMatrix,
    PointCloud,
    Concatenate,
    LeafSystem,
    MultibodyPlant,
    BasicVector,
    Context,
    JacobianWrtVariable,
    PiecewisePose,
    PiecewisePolynomial,
    TrajectorySource,
    Integrator,
    ConstantVectorSource,
    Rgba,
    Box,
    RollPitchYaw,
)
from manipulation.station import LoadScenario, MakeHardwareStation, AddPointClouds
from manipulation.icp import IterativeClosestPoint
from manipulation.meshcat_utils import AddMeshcatTriad
from pathlib import Path
import numpy as np
import os
import time


def write_table_sdf(table_path: Path) -> None:
    table_path.write_text(
        """<?xml version="1.0"?>
<sdf xmlns:drake="drake.mit.edu" version="1.7">
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
        <geometry><box><size>5 5 0.1</size></box></geometry>
        <drake:proximity_properties>
          <drake:compliant_hydroelastic/>
          <drake:hydroelastic_modulus>5.0e7</drake:hydroelastic_modulus>
          <drake:mu_static>1.50</drake:mu_static>
          <drake:mu_dynamic>1.20</drake:mu_dynamic>
        </drake:proximity_properties>
      </collision>
      <visual name="visual">
        <geometry><box><size>5 5 0.1</size></box></geometry>
        <material><diffuse>0.9 0.9 0.9 1.0</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
"""
    )


def write_brick_sdf(brick_path: Path, size_xyz: list[float]) -> None:
    sx, sy, sz = size_xyz
    brick_path.write_text(
        f"""<?xml version="1.0"?>
<sdf xmlns:drake="drake.mit.edu" version="1.7">
  <model name="brick">
    <link name="brick_link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.001</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>0.001</iyy><iyz>0</iyz>
          <izz>0.001</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <drake:proximity_properties>
          <drake:rigid_hydroelastic/>
          <drake:mu_static>2.00</drake:mu_static>
          <drake:mu_dynamic>1.60</drake:mu_dynamic>
        </drake:proximity_properties>
      </collision>
      <visual name="visual">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
      </visual>
    </link>
  </model>
</sdf>
"""
    )


class PseudoInverseController(LeafSystem):
    def __init__(self, plant: MultibodyPlant, iiwa_model_name: str, wsg_model_name: str):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName(iiwa_model_name)
        self._G = plant.GetBodyByName("body", plant.GetModelInstanceByName(wsg_model_name)).body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort(f"{iiwa_model_name}.position", 7)
        self.DeclareVectorOutputPort(f"{iiwa_model_name}.velocity", 7, self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1", self._iiwa).velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7", self._iiwa).velocity_start()

    def CalcOutput(self, context: Context, output: BasicVector):
        V_G = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kV,
            self._G,
            np.zeros((3, 1)),
            self._W,
            self._W,
        )
        J_G = J_G[:, self.iiwa_start : self.iiwa_end + 1]
        # Damped least squares for robustness
        JJt = J_G @ J_G.T
        lam = 0.05
        v = J_G.T @ np.linalg.solve(JJt + (lam**2) * np.eye(6), V_G)
        v = np.clip(v, -1.5, 1.5)
        output.SetFromVector(v)


def sample_box_surface(size_xyz, num_samples=2500) -> PointCloud:
    sx, sy, sz = size_xyz
    areas = np.array([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy], dtype=float)
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


def design_grasp_for_brick(X_WO: RigidTransform, size_xyz, clearance_m=0.10):
    """Design grasp pose for brick similar to notebook's letter grasp."""
    sx, sy, sz = size_xyz
    # Grasp orientation: approach from above, pinch along shorter dimension
    # Similar to notebook: RollPitchYaw(0, 0, pi) @ RollPitchYaw(-pi/2, 0, 0)
    R_OG = (
        RollPitchYaw(0, 0, np.pi).ToRotationMatrix()
        @ RollPitchYaw(-np.pi / 2, 0, 0).ToRotationMatrix()
    )
    # Offset above brick center (similar to notebook's [0.07, 0.06, 0.12])
    p_OG = [0.0, 0.0, sz/2.0 + clearance_m]
    X_OG = RigidTransform(R_OG, p_OG)
    X_WG = X_WO @ X_OG
    return X_OG, X_WG


def design_pregrasp_pose(X_WG: RigidTransform) -> RigidTransform:
    """Pre-grasp: approach from behind (in -Y direction of gripper frame)."""
    X_GGApproach = RigidTransform([0.0, -0.15, 0.0])  # 15cm back
    return X_WG @ X_GGApproach


def design_pregoal_pose(X_WG: RigidTransform) -> RigidTransform:
    """Pre-goal: hover above goal position."""
    X_GGApproach = RigidTransform([0.0, 0.0, -0.15])  # 15cm above
    return X_WG @ X_GGApproach


def design_postgoal_pose(X_WG: RigidTransform) -> RigidTransform:
    """Post-goal: lift after placing."""
    X_GGApproach = RigidTransform([0.0, 0.0, -0.15])  # 15cm above
    return X_WG @ X_GGApproach


def make_trajectories(X_Gs: list[RigidTransform], fingers: np.ndarray, ts: list[float]):
    traj_pos = PiecewisePose.MakeLinear(ts, X_Gs)
    traj_V_G = traj_pos.MakeDerivative()
    traj_wsg = PiecewisePolynomial.FirstOrderHold(ts, fingers)
    return traj_V_G, traj_wsg


if __name__ == "__main__":
    # Randomize per run (brick placement, ICP init, etc.)
    try:
        seed_val = (int(time.time_ns()) ^ int.from_bytes(os.urandom(8), "little")) & 0xFFFFFFFF
        np.random.seed(seed_val)
    except Exception:
        np.random.seed(None)

    assets_dir = Path("assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    table_sdf_path = assets_dir / "table.sdf"
    write_table_sdf(table_sdf_path)
    brick_size = [0.16, 0.08, 0.04]
    bricks_dir = assets_dir / "bricks_geom"
    bricks_dir.mkdir(parents=True, exist_ok=True)
    brick_sdf_path = bricks_dir / "brick.sdf"
    write_brick_sdf(brick_sdf_path, brick_size)

    scenario_yaml = f"""
directives:
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [ 1.6]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
    X_PC:
        translation: [0, -0.5, 0]
        rotation: !Rpy {{ deg: [0, 0, 180]}}

- add_model:
    name: wsg
    file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
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

- add_model:
    name: brick
    file: file://{brick_sdf_path.resolve()}
    default_free_body_pose:
        brick_link:
            translation: [-0.3, 0.0, {brick_size[2]/2.0 + 0.01}]
            rotation: !Rpy {{ deg: [0.0, 0.0, {np.degrees(np.random.uniform(0.0, np.pi)):.1f}]}}

- add_frame:
    name: camera0_origin
    X_PF:
        base_frame: world
        rotation: !Rpy {{ deg: [-120.0, 0.0, 180.0]}}
        translation: [0, 2.0, 1.0]
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
        rotation: !Rpy {{ deg: [-120.0, 0.0, 90.0]}}
        translation: [2.0, 0.0, 1.0]
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
        translation: [-2.0, 0.0, 1.0]
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
        translation: [0, -2.0, 1.0]
- add_model:
    name: camera3
    file: package://manipulation/camera_box.sdf
- add_weld:
    parent: camera3_origin
    child: camera3::base

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

model_drivers:
  iiwa: !IiwaDriver
    control_mode: position_only
    hand_model_name: wsg
  wsg: !SchunkWsgDriver {{}}
"""

    print("Starting MeshCat… (check the VS Code Ports panel)")
    meshcat = StartMeshcat()

    scenario = LoadScenario(data=scenario_yaml)
    # Ensure DISPLAY for headless environments
    if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
        os.environ["DISPLAY"] = ":0"
    station = MakeHardwareStation(scenario, meshcat)

    # Build diagram with cameras (single diagram used for perception/ICP)
    builder = DiagramBuilder()
    builder.AddSystem(station)
    try:
        to_point_cloud = AddPointClouds(scenario=scenario, station=station, builder=builder, meshcat=meshcat)
        builder.ExportOutput(to_point_cloud["camera0"].get_output_port(), "camera_point_cloud0")
        builder.ExportOutput(to_point_cloud["camera1"].get_output_port(), "camera_point_cloud1")
        builder.ExportOutput(to_point_cloud["camera2"].get_output_port(), "camera_point_cloud2")
        builder.ExportOutput(to_point_cloud["camera3"].get_output_port(), "camera_point_cloud3")
    except Exception as e:
        print(f"Warning: AddPointClouds failed: {e}")

    # Hold posture and set gripper defaults for perception phase
    builder.Connect(station.GetOutputPort("iiwa.position_measured"), station.GetInputPort("iiwa.position"))
    builder.Connect(builder.AddSystem(ConstantVectorSource(np.array([0.107]))).get_output_port(),
                    station.GetInputPort("wsg.position"))
    try:
        builder.Connect(builder.AddSystem(ConstantVectorSource(np.array([200.0]))).get_output_port(),
                        station.GetInputPort("wsg.force_limit"))
    except Exception:
        pass

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(10)
    simulator.Initialize()

    # Perception: merge clouds, crop height, downsample, ICP brick
    print("\n=== PERCEPTION PHASE ===")
    context = simulator.get_context()
    pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
    pc1 = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
    pc2 = diagram.GetOutputPort("camera_point_cloud2").Eval(context)
    pc3 = diagram.GetOutputPort("camera_point_cloud3").Eval(context)
    print(f"Camera point cloud sizes: cam0={pc0.size()}, cam1={pc1.size()}, cam2={pc2.size()}, cam3={pc3.size()}")

    merged = Concatenate([pc0, pc1, pc2, pc3])
    print(f"Merged point cloud size: {merged.size()}")

    # Filter by height (remove table) and by region (focus on brick area)
    xyz = merged.xyzs()
    z_keep = (xyz[2, :] >= 0.025) & (xyz[2, :] <= 0.15)  # Brick height range only

    # Also filter by approximate brick location (brick should be around [-0.3, 0, 0])
    # Create a loose bounding box
    x_keep = (xyz[0, :] >= -0.7) & (xyz[0, :] <= -0.1)  # X range around brick
    y_keep = (xyz[1, :] >= -0.4) & (xyz[1, :] <= 0.4)   # Y range around brick

    keep = z_keep & x_keep & y_keep
    cropped = PointCloud(int(np.sum(keep)))
    cropped.mutable_xyzs()[:] = merged.xyzs()[:, keep]
    print(f"After spatial crop: {cropped.size()} points")

    scene_cloud = cropped.VoxelizedDownSample(0.005)  # Smaller voxels for better accuracy
    print(f"After downsampling (voxel_size=0.005): {scene_cloud.size()} points")

    # Visualize scene point cloud (red)
    meshcat.SetObject("debug/scene_cloud", scene_cloud, point_size=0.01, rgba=Rgba(1, 0, 0))

    model_cloud = sample_box_surface(brick_size, num_samples=2500)
    print(f"Model point cloud (brick surface): {model_cloud.size()} points")

    # Initial guess at centroid - use median to be robust to outliers
    xyz_points = scene_cloud.xyzs()
    centroid = np.median(xyz_points, axis=1)

    # Better initial guess: we know brick should be flat on table (no rotation)
    # Start with identity rotation
    X_init = RigidTransform(RotationMatrix.Identity(), centroid)
    print(f"ICP initial guess (median): xyz={centroid}")
    print(f"Scene cloud bounds: x=[{xyz_points[0,:].min():.3f}, {xyz_points[0,:].max():.3f}], "
          f"y=[{xyz_points[1,:].min():.3f}, {xyz_points[1,:].max():.3f}], "
          f"z=[{xyz_points[2,:].min():.3f}, {xyz_points[2,:].max():.3f}]")

    # Visualize model cloud at initial position (green)
    model_at_init = PointCloud(model_cloud.size())
    model_at_init.mutable_xyzs()[:] = X_init.rotation() @ model_cloud.xyzs() + X_init.translation().reshape(3, 1)
    meshcat.SetObject("debug/model_cloud_init", model_at_init, point_size=0.01, rgba=Rgba(0, 1, 0))

    print("\nRunning ICP (max 60 iterations)...")
    X_brick_est, icp_error = IterativeClosestPoint(
        p_Om=model_cloud.xyzs(), p_Ws=scene_cloud.xyzs(), X_Ohat=X_init, meshcat=meshcat, max_iterations=60
    )

    # Visualize final aligned model cloud (blue)
    model_aligned = PointCloud(model_cloud.size())
    model_aligned.mutable_xyzs()[:] = X_brick_est.rotation() @ model_cloud.xyzs() + X_brick_est.translation().reshape(3, 1)
    meshcat.SetObject("debug/model_cloud_aligned", model_aligned, point_size=0.01, rgba=Rgba(0, 0, 1))

    AddMeshcatTriad(meshcat, "debug/icp_brick_pose", X_PT=X_brick_est, length=0.12, radius=0.003)

    # Check ICP convergence
    print(f"\n=== ICP RESULTS ===")
    # Handle both scalar and array error returns
    error_val = float(icp_error) if np.isscalar(icp_error) else float(np.mean(icp_error))
    print(f"Final ICP error: {error_val:.6f}")
    if error_val > 0.01:
        print("⚠️  WARNING: ICP error is high - registration may not have converged!")
        print("   Consider: (1) adjusting initial guess, (2) increasing max_iterations,")
        print("             (3) improving point cloud quality (more cameras, better crop)")
    else:
        print("✓ ICP converged successfully")
    print(f"Estimated brick pose: xyz={X_brick_est.translation()}")
    print(f"                      rpy={RollPitchYaw(X_brick_est.rotation()).vector()}")

    # CONTROL: rebuild a final diagram including station + DIK controller + sources (matching the notebook).
    print("\n=== CONTROL PHASE ===")
    builder2 = DiagramBuilder()
    station2 = MakeHardwareStation(scenario, meshcat)
    builder2.AddSystem(station2)
    plant = station2.plant()

    # Waypoints and trajectories (matching notebook pattern)
    # TUNABLE PARAMETERS:
    grasp_clearance = 0.08  # Distance from brick surface to gripper base (tune for better grasp)

    plant_ctx_tmp = plant.GetMyContextFromRoot(station2.CreateDefaultContext())
    X_WGinitial = plant.EvalBodyPoseInWorld(
        plant_ctx_tmp, plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg"))
    )
    print(f"Initial gripper pose: xyz={X_WGinitial.translation()}")

    # Design grasp pose (returns both X_OG and X_WGpick)
    X_OG, X_WGpick = design_grasp_for_brick(X_brick_est, brick_size, clearance_m=grasp_clearance)
    print(f"Grasp offset from brick center: {X_OG.translation()}")

    # Pre-grasp: approach from behind
    X_WGprepick = design_pregrasp_pose(X_WGpick)

    # Goal: move brick to a different location (away from robot base!)
    offsets_xy = [
        np.array([-0.25, 0.00, 0.0]),  # Forward (away from robot)
        np.array([-0.15, 0.25, 0.0]),  # Forward-right
        np.array([-0.15, -0.25, 0.0]), # Forward-left
        np.array([-0.35, 0.00, 0.0]),  # Further forward
    ]
    goal_offset = offsets_xy[np.random.randint(len(offsets_xy))]
    X_WOgoal = RigidTransform(X_brick_est.rotation(), X_brick_est.translation() + goal_offset)
    X_WGgoal = X_WOgoal @ X_OG
    print(f"Goal brick position: {X_WOgoal.translation()}")
    print(f"Goal offset from current: {goal_offset[:2]}")

    # Pre-goal and post-goal: hover positions
    X_WGpregoal = design_pregoal_pose(X_WGgoal)
    X_WGpostgoal = design_postgoal_pose(X_WGgoal)

    # Build waypoint list matching notebook pattern:
    # initial -> prepick -> pick -> (close) -> pregoal -> goal -> (open) -> postgoal -> initial
    X_Gs = [
        X_WGinitial,    # 0: Start position
        X_WGprepick,    # 1: Approach brick
        X_WGpick,       # 2: At brick (gripper open)
        X_WGpick,       # 3: At brick (close gripper)
        X_WGpregoal,    # 4: Lift up with brick
        X_WGgoal,       # 5: At goal position
        X_WGgoal,       # 6: At goal (open gripper)
        X_WGpostgoal,   # 7: Lift after releasing
        X_WGinitial     # 8: Return to start
    ]
    ts = [0, 2.0, 3.5, 4.5, 6.5, 9.0, 10.0, 11.0, 13.0]
    opened = 0.107
    closed = 0.0
    fingers = np.asarray([opened, opened, opened, closed, closed, closed, opened, opened, opened]).reshape(1, -1)
    traj_V_G, traj_wsg = make_trajectories(X_Gs, fingers, ts)
    # Wire controller and sources
    V_src = builder2.AddSystem(TrajectorySource(traj_V_G))
    dik = builder2.AddSystem(PseudoInverseController(plant=plant, iiwa_model_name="iiwa", wsg_model_name="wsg"))
    integ = builder2.AddSystem(Integrator(7))
    wsg_src = builder2.AddSystem(TrajectorySource(traj_wsg))
    builder2.Connect(V_src.get_output_port(), dik.get_input_port(0))
    builder2.Connect(station2.GetOutputPort("iiwa.position_measured"), dik.get_input_port(1))
    builder2.Connect(dik.get_output_port(), integ.get_input_port())
    builder2.Connect(integ.get_output_port(), station2.GetInputPort("iiwa.position"))
    builder2.Connect(wsg_src.get_output_port(), station2.GetInputPort("wsg.position"))
    try:
        builder2.Connect(builder2.AddSystem(ConstantVectorSource(np.array([200.0]))).get_output_port(),
                         station2.GetInputPort("wsg.force_limit"))
    except Exception:
        pass
    diagram2 = builder2.Build()
    sim2 = Simulator(diagram2)
    ctx2 = sim2.get_mutable_context()
    
    q0 = plant.GetPositions(plant.GetMyContextFromRoot(ctx2), plant.GetModelInstanceByName("iiwa"))
    integ.GetMyContextFromRoot(ctx2).get_mutable_continuous_state_vector().SetFromVector(q0)

    # Visualize trajectory waypoints in Meshcat
    for i, X_G in enumerate(X_Gs):
        AddMeshcatTriad(meshcat, f"trajectory/waypoint_{i}", X_PT=X_G, length=0.08, radius=0.002, opacity=0.6)

    print(f"\nStarting pick-and-place simulation (duration: {ts[-1] + 0.5:.1f}s)...")
    print(f"Trajectory: {len(X_Gs)} waypoints over {len(ts)} time steps")

    sim2.set_target_realtime_rate(1.0)
    sim2.Initialize()
    sim2.AdvanceTo(ts[-1] + 0.5)

    print("\n=== SIMULATION COMPLETE ===")
    print("✓ Check MeshCat visualization to see the result")
    print("  - Red points: scene point cloud from cameras")
    print("  - Green points: model cloud at initial ICP guess")
    print("  - Blue points: model cloud after ICP alignment")
    print("  - Trajectory waypoints: small coordinate frames along path")