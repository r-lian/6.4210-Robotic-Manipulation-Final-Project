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


def antipodal_grasp_for_box(size_xyz, clearance_m=0.10) -> RigidTransform:
    sx, sy, _ = float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])
    # Pinch along the smaller of x/y; approach from +z_O
    use_y = sy < sx
    xg_o = np.array([0.0, 1.0, 0.0]) if use_y else np.array([1.0, 0.0, 0.0])
    yg_o = np.array([0.0, 0.0, -1.0])
    zg_o = np.cross(xg_o, yg_o)
    zg_o /= np.linalg.norm(zg_o)
    yg_o = np.cross(zg_o, xg_o)
    R_OG = RotationMatrix(np.column_stack((xg_o, yg_o, zg_o)))
    # Offset upward along -y_G (i.e., +z_O) by clearance
    return RigidTransform(R_OG, [0.0, -abs(clearance_m), 0.0])


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
        translation: [0.6, 0.0, 0.0]

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
            translation: [0.0, 0.0, {brick_size[2]/2.0 + 0.01}]
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
    context = simulator.get_context()
    pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
    pc1 = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
    pc2 = diagram.GetOutputPort("camera_point_cloud2").Eval(context)
    pc3 = diagram.GetOutputPort("camera_point_cloud3").Eval(context)
    merged = Concatenate([pc0, pc1, pc2, pc3])
    z = merged.xyzs()[2, :]
    keep = (z >= 0.02) & (z <= 0.25)
    cropped = PointCloud(int(np.sum(keep)))
    cropped.mutable_xyzs()[:] = merged.xyzs()[:, keep]
    scene_cloud = cropped.VoxelizedDownSample(0.01)

    model_cloud = sample_box_surface(brick_size, num_samples=2500)
    # Initial guess at centroid
    centroid = np.mean(scene_cloud.xyzs(), axis=1)
    X_init = RigidTransform(RotationMatrix.Identity(), centroid)
    X_brick_est, _ = IterativeClosestPoint(
        p_Om=model_cloud.xyzs(), p_Ws=scene_cloud.xyzs(), X_Ohat=X_init, meshcat=meshcat, max_iterations=60
    )
    AddMeshcatTriad(meshcat, "debug/icp_brick_pose", X_PT=X_brick_est, length=0.12, radius=0.003)

    # CONTROL: rebuild a final diagram including station + DIK controller + sources (matching the notebook).
    builder2 = DiagramBuilder()
    station2 = MakeHardwareStation(scenario, meshcat)
    builder2.AddSystem(station2)
    plant = station2.plant()
    # Waypoints and trajectories
    hover_h = 0.18
    plant_ctx_tmp = plant.GetMyContextFromRoot(station2.CreateDefaultContext())
    X_WGinitial = plant.EvalBodyPoseInWorld(
        plant_ctx_tmp, plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg"))
    )
    X_OG = antipodal_grasp_for_box(brick_size, clearance_m=0.10)
    X_grasp = X_brick_est @ X_OG
    X_hover = RigidTransform(X_grasp.rotation(), X_grasp.translation() + np.array([0.0, 0.0, hover_h]))
    offsets_xy = [
        np.array([0.20, 0.00, 0.0]),
        np.array([0.00, 0.20, 0.0]),
        np.array([-0.20, 0.00, 0.0]),
        np.array([0.00, -0.20, 0.0]),
    ]
    goal_offset = offsets_xy[np.random.randint(len(offsets_xy))]
    X_goal_obj = RigidTransform(X_brick_est.rotation(), X_brick_est.translation() + goal_offset)
    X_WGgoal = X_goal_obj @ X_OG
    X_WGgoal_hover = RigidTransform(X_WGgoal.rotation(), X_WGgoal.translation() + np.array([0.0, 0.0, hover_h]))
    X_Gs = [X_WGinitial, X_hover, X_grasp, X_grasp, X_hover, X_WGgoal_hover, X_WGgoal, X_WGgoal_hover]
    ts = [0, 1.5, 3.0, 4.0, 5.5, 8.0, 9.5, 11.0]
    opened = 0.107
    closed = 0.0
    fingers = np.asarray([opened, opened, closed, closed, closed, closed, opened, opened]).reshape(1, -1)
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
    sim2.set_target_realtime_rate(1.0)
    sim2.Initialize()
    sim2.AdvanceTo(ts[-1] + 0.5)