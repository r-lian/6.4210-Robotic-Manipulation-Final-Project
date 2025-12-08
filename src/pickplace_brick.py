"""
Pick and Place with Brick - Following notebook pattern
Adapted from 11_pickplace_initials_with_geometry notebook
"""

import os
from pathlib import Path

import numpy as np
from pydrake.all import (
    AddFrameTriadIllustration,
    BasicVector,
    Box,
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
    Trajectory,
    TrajectorySource,
)

from manipulation.icp import IterativeClosestPoint
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.station import (
    AddPointClouds,
    LoadScenario,
    MakeHardwareStation,
)

# Start meshcat
print("Starting MeshCat… (check the VS Code Ports panel)")
meshcat = StartMeshcat()
print("Click the link above to open Meshcat in your browser!")

# ============================================================================
# Part 1: Setting Up the Scenario
# ============================================================================

# Prepare assets directory
assets_dir = Path("assets")
bricks_dir = assets_dir / "bricks"
bricks_dir.mkdir(parents=True, exist_ok=True)

# Write table SDF
table_sdf_path = assets_dir / "table.sdf"
table_sdf_path.write_text(
    """<?xml version="1.0"?>
<sdf version="1.7">
    <model name="table">
        <pose>0 0 0 0 0 0</pose>
        <link name="table_link">
            <inertial>
                <mass>1.0</mass>
                <inertia>
                    <ixx>0.005833</ixx>
                    <ixy>0.0</ixy>
                    <ixz>0.0</ixz>
                    <iyy>0.005833</iyy>
                    <iyz>0.0</iyz>
                    <izz>0.005</izz>
                </inertia>
            </inertial>
            <collision name="collision">
                <geometry>
                    <box>
                        <size>2 2 0.1</size>
                    </box>
                </geometry>
            </collision>
            <visual name="visual">
                <geometry>
                    <box>
                        <size>2 2 0.1</size>
                    </box>
                </geometry>
                <material>
                    <diffuse>1.0 1.0 1.0 1.0</diffuse>
                </material>
            </visual>
        </link>
    </model>
</sdf>
"""
)

# Brick parameters - make it smaller than gripper opening (0.107m)
# Gripper opens to 0.107m, so brick should be smaller
# Size: length x width x height (grasp along width, so width < 0.107m)
brick_size = [0.08, 0.05, 0.03]  # x, y, z in meters (8cm x 5cm x 3cm)

# Write brick SDF
def write_brick_sdf(path: Path, size_xyz):
    sx, sy, sz = size_xyz
    path.write_text(
        f"""<?xml version="1.0"?>
<sdf xmlns:drake="drake.mit.edu" version="1.7">
  <model name="brick_model">
    <link name="brick_link">
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.0005</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>0.0005</iyy><iyz>0</iyz>
          <izz>0.0005</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <drake:proximity_properties>
          <drake:rigid_hydroelastic/>
          <drake:mu_static>0.90</drake:mu_static>
          <drake:mu_dynamic>0.80</drake:mu_dynamic>
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

brick_sdf_path = bricks_dir / "brick_0.sdf"
write_brick_sdf(brick_sdf_path, brick_size)

# Create scenario YAML (following notebook pattern)
table_sdf = f"{Path.cwd()}/assets/table.sdf"
brick_sdf = f"{Path.cwd()}/assets/bricks/brick_0.sdf"

scenario_yaml = f"""directives:
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
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 90]}}
- add_model:
    name: table
    file: file://{table_sdf}
- add_weld:
    parent: world
    child: table::table_link
    X_PC:
        translation: [0.0, 0.0, -0.05]
        rotation: !Rpy {{ deg: [0, 0, -90] }}
- add_model:
    name: brick
    file: file://{brick_sdf}
    default_free_body_pose:
        brick_link:
            translation: [-0.35, 0, {brick_size[2]/2:.4f}]
            rotation: !Rpy {{ deg: [0, 0, 0] }}
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

# Load scenario and create station (following notebook pattern)
scenario = LoadScenario(data=scenario_yaml)

# Ensure DISPLAY is set
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    os.environ["DISPLAY"] = ":0"

station = MakeHardwareStation(scenario, meshcat)

# Create builder and add station
builder = DiagramBuilder()
builder.AddSystem(station)

# Add point clouds
to_point_cloud = AddPointClouds(
    scenario=scenario, station=station, builder=builder, meshcat=meshcat
)

# Export point cloud outputs
builder.ExportOutput(to_point_cloud["camera0"].get_output_port(), "camera_point_cloud0")
builder.ExportOutput(to_point_cloud["camera1"].get_output_port(), "camera_point_cloud1")
builder.ExportOutput(to_point_cloud["camera2"].get_output_port(), "camera_point_cloud2")

# Build diagram
diagram = builder.Build()

# Publish initial state
context = diagram.CreateDefaultContext()
diagram.ForcedPublish(context)

# ============================================================================
# Part 2: Registering Point Clouds With ICP
# ============================================================================

print("\n=== Part 2: Getting Point Clouds and Running ICP ===\n")

# Get point clouds from cameras
camera0_pc = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
camera1_pc = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
camera2_pc = diagram.GetOutputPort("camera_point_cloud2").Eval(context)

# Merge point clouds
merged = Concatenate([camera0_pc, camera1_pc, camera2_pc])

# Crop by height to remove table
z = merged.xyzs()[2, :]
keep = (z >= 0.01) & (z <= 0.25)
cropped = PointCloud(int(np.sum(keep)))
cropped.mutable_xyzs()[:] = merged.xyzs()[:, keep]

# Downsample
downsampled = cropped.VoxelizedDownSample(0.005)

# Create model point cloud for brick (box surface sampling)
def sample_box_surface(size_xyz, num_samples=2000) -> PointCloud:
    """Sample points from box surface, following notebook pattern."""
    sx, sy, sz = size_xyz
    areas = np.array([sy*sz, sy*sz, sx*sz, sx*sz, sx*sy, sx*sy], dtype=float)
    probs = areas / np.sum(areas)
    face_ids = np.random.choice(6, size=num_samples, p=probs)
    u = np.random.uniform(-0.5, 0.5, size=num_samples)
    v = np.random.uniform(-0.5, 0.5, size=num_samples)
    pts = np.zeros((3, num_samples))
    mask = face_ids == 0; pts[:, mask] = np.vstack([np.full(np.sum(mask), +sx/2), sy*u[mask], sz*v[mask]])
    mask = face_ids == 1; pts[:, mask] = np.vstack([np.full(np.sum(mask), -sx/2), sy*u[mask], sz*v[mask]])
    mask = face_ids == 2; pts[:, mask] = np.vstack([sx*u[mask], np.full(np.sum(mask), +sy/2), sz*v[mask]])
    mask = face_ids == 3; pts[:, mask] = np.vstack([sx*u[mask], np.full(np.sum(mask), -sy/2), sz*v[mask]])
    mask = face_ids == 4; pts[:, mask] = np.vstack([sx*u[mask], sy*v[mask], np.full(np.sum(mask), +sz/2)])
    mask = face_ids == 5; pts[:, mask] = np.vstack([sx*u[mask], sy*v[mask], np.full(np.sum(mask), -sz/2)])
    model = PointCloud(num_samples)
    model.mutable_xyzs()[:] = pts
    return model

N_SAMPLE_POINTS = 1500
model_cloud = sample_box_surface(brick_size, num_samples=N_SAMPLE_POINTS)

# Visualize
meshcat.SetObject("brick_point_cloud", downsampled, point_size=0.05, rgba=Rgba(1, 0, 0))
meshcat.SetObject("brick_model_cloud", model_cloud, point_size=0.05, rgba=Rgba(0, 1, 0))

# Get actual brick pose from plant for initial guess
plant = station.plant()
plant_context = diagram.GetSubsystemContext(plant, context)
world_frame = plant.world_frame()
model_brick = plant.GetModelInstanceByName("brick")
frame_brick = plant.GetFrameByName("brick_link", model_instance=model_brick)
X_PC_brick = plant.CalcRelativeTransform(plant_context, world_frame, frame_brick)

print(f"Brick initial pose (from plant): {X_PC_brick.translation()}")

# Run ICP
MAX_ITERATIONS = 70
X_brick_initial = X_PC_brick
model_points = model_cloud.xyzs()
scene_points = downsampled.xyzs()

print("Running ICP...")
brick_X_Ohat, brick_error = IterativeClosestPoint(
    p_Om=model_points,
    p_Ws=scene_points,
    X_Ohat=X_brick_initial,
    meshcat=meshcat,
    max_iterations=MAX_ITERATIONS
)

# Check error
error_brick = brick_X_Ohat.inverse().multiply(X_PC_brick)
rpy = RollPitchYaw(error_brick.rotation()).vector()
xyz = error_brick.translation()
print(f"Brick ICP error: rpy: {rpy}, xyz: {xyz}")

# Visualize detected brick
brick_box = Box(brick_size[0], brick_size[1], brick_size[2])
meshcat.SetObject("brick_detected", brick_box, Rgba(1, 0, 0, 0.4))
meshcat.SetTransform("brick_detected", brick_X_Ohat)
AddMeshcatTriad(meshcat, "brick_triad", X_PT=brick_X_Ohat, length=0.1, radius=0.003)

# ============================================================================
# Part 3: Pick and Place with Registered Geometries
# ============================================================================

print("\n=== Part 3: Pick and Place ===\n")

# Trajectory design functions (following notebook pattern)
def design_grasp_pose(X_WO: RigidTransform) -> tuple[RigidTransform, RigidTransform]:
    """Design grasp pose relative to brick."""
    R_OG = (
        RollPitchYaw(0, 0, np.pi).ToRotationMatrix()
        @ RollPitchYaw(-np.pi / 2, 0, 0).ToRotationMatrix()
    )
    # Grasp from top, offset by half brick height + small clearance
    p_OG = np.array([0.0, 0.0, brick_size[2] / 2.0 + 0.01])
    X_OG = RigidTransform(R_OG, p_OG)
    X_WG = X_WO.multiply(X_OG)
    return X_OG, X_WG

def design_pregrasp_pose(X_WG: RigidTransform) -> RigidTransform:
    """Design approach pose before grasping."""
    X_GGApproach = RigidTransform([0.0, -0.2, 0.0])
    X_WGApproach = X_WG.multiply(X_GGApproach)
    return X_WGApproach

def design_pregoal_pose(X_WG: RigidTransform) -> RigidTransform:
    """Design approach pose before placing."""
    X_GGApproach = RigidTransform([0.0, 0.0, -0.2])
    X_WGApproach = X_WG.multiply(X_GGApproach)
    return X_WGApproach

def design_goal_pose(X_WO: RigidTransform, X_OG: RigidTransform) -> RigidTransform:
    """Design goal placement pose - place brick 0.3m to the right."""
    X_WOgoal = RigidTransform(
        X_WO.rotation(),  # Keep same orientation
        X_WO.translation() + np.array([0.3, 0.0, 0.0])  # Move 0.3m to the right
    )
    X_WGgoal = X_WOgoal.multiply(X_OG)
    return X_WGgoal

def design_postgoal_pose(X_WG: RigidTransform) -> RigidTransform:
    """Design retract pose after placing."""
    X_GGApproach = RigidTransform([0.0, 0.0, -0.2])
    X_WGApproach = X_WG.multiply(X_GGApproach)
    return X_WGApproach

def make_trajectory(
    X_Gs: list[RigidTransform], finger_values: np.ndarray, sample_times: list[float]
) -> tuple[Trajectory, PiecewisePolynomial]:
    """Create trajectory from keyframes."""
    robot_position_trajectory = PiecewisePose.MakeLinear(sample_times, X_Gs)
    robot_velocity_trajectory = robot_position_trajectory.MakeDerivative()
    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(sample_times, finger_values)
    return robot_velocity_trajectory, traj_wsg_command

# PseudoInverseController (following notebook pattern)
class PseudoInverseController(LeafSystem):
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

# Rebuild diagram with controller (following notebook pattern)
print("Rebuilding diagram with pick-and-place controller...")

# Get initial poses from the existing diagram before rebuilding
# Use the plant we already have from earlier
plant_context = diagram.GetSubsystemContext(plant, context)
X_WGinitial = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("body"))
X_WOinitial = brick_X_Ohat  # Use ICP result

# Now recreate station and builder (following notebook - they call the function again)
# We need a fresh station because the old one is already in a diagram
station = MakeHardwareStation(scenario, meshcat)
builder = DiagramBuilder()
builder.AddSystem(station)

# Add point clouds
to_point_cloud = AddPointClouds(
    scenario=scenario, station=station, builder=builder, meshcat=meshcat
)

# Get plant from the new station
plant_pnp = station.GetSubsystemByName("plant")

# Build trajectory keyframes
X_OG, X_WGpick = design_grasp_pose(X_WOinitial)
X_WGprepick = design_pregrasp_pose(X_WGpick)
X_WGgoal = design_goal_pose(X_WOinitial, X_OG)
X_WGpregoal = design_pregoal_pose(X_WGgoal)
X_WGpostgoal = design_postgoal_pose(X_WGgoal)

# Constants for finger distances
opened = 0.107
closed = 0.0

# Keyframes: initial → prepick → pick → pregoal → goal → postgoal → initial
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

gripper_poses = [keyframe[1] for keyframe in keyframes]
finger_states = np.asarray([keyframe[2] for keyframe in keyframes]).reshape(1, -1)
sample_times = [3 * i for i in range(len(gripper_poses))]

# Create trajectories
traj_V_G, traj_wsg_command = make_trajectory(gripper_poses, finger_states, sample_times)

# Add systems
V_G_source = builder.AddSystem(TrajectorySource(traj_V_G))
controller = builder.AddSystem(PseudoInverseController(plant_pnp))
integrator = builder.AddSystem(Integrator(7))
wsg_source = builder.AddSystem(TrajectorySource(traj_wsg_command))

# Wire connections
builder.Connect(V_G_source.get_output_port(), controller.get_input_port(0))
builder.Connect(controller.get_output_port(), integrator.get_input_port())
builder.Connect(integrator.get_output_port(), station.GetInputPort("iiwa.position"))
builder.Connect(station.GetOutputPort("iiwa.position_measured"), controller.get_input_port(1))
builder.Connect(wsg_source.get_output_port(), station.GetInputPort("wsg.position"))

# Add frame triads for visualization
try:
    scenegraph = station.GetSubsystemByName("scene_graph")
    AddFrameTriadIllustration(
        scene_graph=scenegraph,
        body=plant_pnp.GetBodyByName("brick_link"),
        length=0.1,
    )
    AddFrameTriadIllustration(
        scene_graph=scenegraph,
        body=plant_pnp.GetBodyByName("body"),
        length=0.1
    )
except:
    pass  # Skip visualization if not available

# Build diagram
diagram = builder.Build()

# Create simulator
simulator = Simulator(diagram)
context = simulator.get_mutable_context()
station_context = station.GetMyContextFromRoot(context)

# Initialize integrator with current joint positions
integrator.set_integral_value(
    integrator.GetMyContextFromRoot(context),
    plant_pnp.GetPositions(
        plant_pnp.GetMyContextFromRoot(context),
        plant_pnp.GetModelInstanceByName("iiwa"),
    ),
)

diagram.ForcedPublish(context)
print(f"Simulation will run for {traj_V_G.end_time()} seconds")

# Run simulation
print("Running pick-and-place simulation...")
simulator.set_target_realtime_rate(5.0)  # Run at 5x speed for faster simulation
simulator.AdvanceTo(traj_V_G.end_time())
print("Pick-and-place complete!")

print("\nSimulation complete. Keeping visualization open...")
import time
time.sleep(30)

