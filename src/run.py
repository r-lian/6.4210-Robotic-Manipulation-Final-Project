from pydrake.all import StartMeshcat, DiagramBuilder, Simulator, RigidTransform, RotationMatrix, PointCloud, Concatenate, LeafSystem, MultibodyPlant, BasicVector, Context, JacobianWrtVariable, PiecewisePose, PiecewisePolynomial, Trajectory, TrajectorySource, Integrator
from manipulation.letter_generation import create_sdf_asset_from_letter
from manipulation.station import LoadScenario, MakeHardwareStation, AddPointClouds
from manipulation.icp import IterativeClosestPoint
from manipulation.meshcat_utils import AddMeshcatTriad
import time
import numpy as np
import os
from pathlib import Path


if __name__ == "__main__":

    # ---------- Minimal Diff-IK utilities (not wired by default) ----------
    class PseudoInverseController(LeafSystem):
        """
        Differential IK: maps desired spatial velocity of the gripper V_WG (6) to
        joint velocities v (7 for iiwa14) via damped least-squares Jacobian inverse.
        Model and gripper body names are parameters to keep usage flexible.
        """
        def __init__(self, plant: MultibodyPlant, iiwa_model_name: str, gripper_body_name: str = "body", damping: float = 1e-3, v_limit: float = 1.5):
            LeafSystem.__init__(self)
            self._plant = plant
            self._plant_context = plant.CreateDefaultContext()
            self._iiwa = plant.GetModelInstanceByName(iiwa_model_name)
            self._G = plant.GetBodyByName(gripper_body_name).body_frame()
            self._W = plant.world_frame()
            self._damping = damping
            self._v_limit = v_limit  # rad/s cap per joint

            # Assume iiwa14 joint names exist; use them to get a contiguous velocity block.
            self._vel_start = plant.GetJointByName("iiwa_joint_1", self._iiwa).velocity_start()
            self._vel_end = plant.GetJointByName("iiwa_joint_7", self._iiwa).velocity_start()
            self._n_vel = self._vel_end - self._vel_start + 1

            self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
            self.q_port = self.DeclareVectorInputPort("iiwa.position", 7)
            self.DeclareVectorOutputPort("iiwa.velocity", self._n_vel, self.CalcOutput)

        def CalcOutput(self, context: Context, output: BasicVector):
            V_WG = self.V_G_port.Eval(context)
            q = self.q_port.Eval(context)
            self._plant.SetPositions(self._plant_context, self._iiwa, q)

            J = self._plant.CalcJacobianSpatialVelocity(
                self._plant_context,
                JacobianWrtVariable.kV,
                self._G,
                [0.0, 0.0, 0.0],
                self._W,
                self._W,
            )
            J = J[:, self._vel_start:self._vel_end + 1]

            # Damped least squares: v = Jᵀ(JJᵀ + λ²I)⁻¹ V
            JJt = J @ J.T
            lam2I = (self._damping ** 2) * np.eye(JJt.shape[0])
            v = J.T @ np.linalg.solve(JJt + lam2I, V_WG)

            # Velocity limiting (uniform scale if any component exceeds cap)
            vmax = np.max(np.abs(v)) if v.size else 0.0
            if vmax > self._v_limit:
                v = (self._v_limit / vmax) * v
            output.SetFromVector(v)

    def make_trajectory(X_Gs: list[RigidTransform], finger_values: np.ndarray, sample_times: list[float]) -> tuple[Trajectory, PiecewisePolynomial]:
        robot_position_trajectory = PiecewisePose.MakeLinear(sample_times, X_Gs)
        robot_velocity_trajectory = robot_position_trajectory.MakeDerivative()
        traj_wsg_command = PiecewisePolynomial.FirstOrderHold(sample_times, finger_values)
        return robot_velocity_trajectory, traj_wsg_command

    def setup_diff_ik_demo(builder: DiagramBuilder, station, iiwa_model_name: str, wsg_model_name: str, gripper_body_name: str, X_Gs: list[RigidTransform], finger_values: np.ndarray, sample_times: list[float]):
        """
        Optional helper to wire Diff-IK into the existing station.
        Not called by default; call manually to enable.
        """
        traj_V_G, traj_wsg = make_trajectory(X_Gs, finger_values, sample_times)
        V_G_source = builder.AddSystem(TrajectorySource(traj_V_G))
        controller = builder.AddSystem(PseudoInverseController(plant=station.plant(), iiwa_model_name=iiwa_model_name, gripper_body_name=gripper_body_name))
        integrator = builder.AddSystem(Integrator(7))
        wsg_source = builder.AddSystem(TrajectorySource(traj_wsg))

        # Wire systems
        builder.Connect(V_G_source.get_output_port(), controller.get_input_port(0))
        builder.Connect(controller.get_output_port(), integrator.get_input_port())
        builder.Connect(integrator.get_output_port(), station.GetInputPort(f"{iiwa_model_name}.position"))
        builder.Connect(station.GetOutputPort(f"{iiwa_model_name}.position_measured"), controller.get_input_port(1))
        builder.Connect(wsg_source.get_output_port(), station.GetInputPort(f"{wsg_model_name}.position"))
        return V_G_source, controller, integrator, wsg_source

    def place(brick_name: str, X_WGoal: RigidTransform, sim_duration: float = 12.0):
        """
        Builds a minimal pick-and-place run for a single brick:
        - Chooses the nearer arm (left/right) by XY distance to base
        - Plans a simple vertical approach / retreat
        - Wires differential IK and runs a short sim
        Noninvasive: constructs its own builder/diagram; does not modify the main one.
        """
        # Recreate station for this short run (idempotent with same scenario).
        local_builder = DiagramBuilder()
        local_station = MakeHardwareStation(LoadScenario(data=scenario_yaml), meshcat)
        local_builder.AddSystem(local_station)

        # Query brick world pose from the local station context (default state).
        plant = local_station.plant()
        ctx = local_station.CreateDefaultContext()
        try:
            model = plant.GetModelInstanceByName(brick_name)
        except Exception:
            print(f"place(): could not find model '{brick_name}'")
            return
        brick_frame = plant.GetFrameByName("brick_link", model)
        X_WB = plant.CalcRelativeTransform(ctx, plant.world_frame(), brick_frame)

        p = X_WB.translation()
        # Pick nearer arm by distance to base welds at x=±0.5, y=0.
        left_base = np.array([0.5, 0.0])
        right_base = np.array([-0.5, 0.0])
        d_left = np.linalg.norm(p[:2] - left_base)
        d_right = np.linalg.norm(p[:2] - right_base)
        use_left = d_left <= d_right
        iiwa_name = "iiwa_left" if use_left else "iiwa_right"
        wsg_name = "wsg_left" if use_left else "wsg_right"

        # Simple grasp frames: approach from above, align gripper Z with world -Z.
        hover_offset = 0.20
        grasp_offset = 0.02  # small penetration offset above top face
        # Yaw aligned to brick yaw in XY plane (approx from current X_WB rotation).
        yaw = np.arctan2(X_WB.rotation().matrix()[1, 0], X_WB.rotation().matrix()[0, 0])
        R_WG_down = RotationMatrix.MakeZRotation(yaw) @ RotationMatrix.MakeXRotation(np.pi)
        p_hover = p.copy(); p_hover[2] += hover_offset
        p_grasp = p.copy(); p_grasp[2] += grasp_offset

        # Goal hover/placement above target pose
        p_goal = X_WGoal.translation()
        R_goal = X_WGoal.rotation()
        p_goal_hover = p_goal.copy(); p_goal_hover[2] += hover_offset

        # Build keyframes for the gripper pose.
        X_Gs = [
            RigidTransform(R_WG_down, p_hover),
            RigidTransform(R_WG_down, p_grasp),
            RigidTransform(R_WG_down, p_hover),
            RigidTransform(R_goal, p_goal_hover),
            RigidTransform(R_goal, p_goal),
            RigidTransform(R_goal, p_goal_hover),
        ]
        opened = 0.107
        closed = 0.0
        finger_values = np.asarray([opened, closed, closed, closed, opened, opened]).reshape(1, -1)
        sample_times = [0, 2, 4, 7, 9, 11]

        # Wire diff-ik
        V_G_source, controller, integrator, wsg_source = setup_diff_ik_demo(
            local_builder, local_station, iiwa_name, wsg_name, "body", X_Gs, finger_values, sample_times
        )

        local_diagram = local_builder.Build()
        # Set integrator initial q to measured q
        sim = Simulator(local_diagram)
        sim_ctx = sim.get_mutable_context()
        station_ctx = local_station.GetMyContextFromRoot(sim_ctx)
        q0 = plant.GetPositions(plant.GetMyContextFromRoot(sim_ctx), plant.GetModelInstanceByName(iiwa_name))
        integrator.GetMyContextFromRoot(sim_ctx).get_mutable_continuous_state_vector().SetFromVector(q0)
        sim.set_target_realtime_rate(1.0)
        sim.Initialize()
        sim.AdvanceTo(sim_duration)

    # Prepare assets directory and write a table and brick SDFs so they can be part of the station plant.
    assets_dir = Path("assets")
    bricks_dir = assets_dir / "bricks"
    bricks_dir.mkdir(parents=True, exist_ok=True)

    # Write a table SDF (2 x 2 x 0.1) as in the reference notebook; top at z=0 when welded at z=-0.05.
    table_sdf_path = assets_dir / "table.sdf"
    table_sdf_path.write_text(
        """<?xml version="1.0"?>
<sdf xmlns:drake="drake.mit.edu" version="1.7">
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
                <drake:proximity_properties>
                    <drake:compliant_hydroelastic/>
                    <drake:hydroelastic_modulus>5.0e7</drake:hydroelastic_modulus>
                    <drake:mu_static>0.90</drake:mu_static>
                    <drake:mu_dynamic>0.80</drake:mu_dynamic>
                </drake:proximity_properties>
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

    # Build base of scenario (robots, grippers, cameras frames/boxes); bricks/table appended below.
    scenario_yaml_base = f"""directives:
    - add_model:
        name: iiwa_left
        file: package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf
        default_joint_positions:
            iiwa_joint_1: [0]
            iiwa_joint_2: [0]
            iiwa_joint_3: [0]
            iiwa_joint_4: [0]
            iiwa_joint_5: [0]
            iiwa_joint_6: [0]
            iiwa_joint_7: [0]
    - add_weld:
        parent: world
        child: iiwa_left::iiwa_link_0
        X_PC:
            translation: [0.5, 0, 0]

    - add_model:
        name: wsg_left
        file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf
    - add_weld:
        parent: iiwa_left::iiwa_link_7
        child: wsg_left::body
        X_PC:
            translation: [0, 0, 0.09]
            rotation: !Rpy {{ deg: [90, 0, 90]}}

    - add_model:
        name: iiwa_right
        file: package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf
        default_joint_positions:
            iiwa_joint_1: [0]
            iiwa_joint_2: [0]
            iiwa_joint_3: [0]
            iiwa_joint_4: [0]
            iiwa_joint_5: [0]
            iiwa_joint_6: [0]
            iiwa_joint_7: [0]
    - add_weld:
        parent: world
        child: iiwa_right::iiwa_link_0
        X_PC:
            translation: [-0.5, 0, 0]

    - add_model:
        name: wsg_right
        file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf
    - add_weld:
        parent: iiwa_right::iiwa_link_7
        child: wsg_right::body
        X_PC:
            translation: [0, 0, 0.09]
            rotation: !Rpy {{ deg: [90, 0, 90]}}

    # Camera frames and simple visuals (used for registering sensors via scenario cameras)
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
            rotation: !Rpy {{ deg: [-125.0, 0.0, 90.0]}}
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
"""

    # Append table model (welded to world so top is at z=0).
    table_directives = f"""
    - add_model:
        name: table
        file: file://{table_sdf_path.resolve()}
    - add_weld:
        parent: world
        child: table::table_link
        X_PC:
            translation: [0.0, 0.0, -0.05]
"""

    # Define brick geometry and write SDFs (single-link box). Then add models with randomized poses.
    brick_size = [0.2, 0.1, 0.06]   # x, y, z in meters
    num_bricks = 10
    x_min, x_max = -0.9, 0.9
    y_min, y_max = -0.35, 0.35
    min_arm_clearance = 0.4
    min_brick_spacing = 0.25
    arm_bases = np.array([[0.5, 0.0], [-0.5, 0.0]])
    placed_xy = []

    def write_brick_sdf(path: Path, size_xyz):
        sx, sy, sz = size_xyz
        path.write_text(
            f"""<?xml version="1.0"?>
<sdf xmlns:drake="drake.mit.edu" version="1.7">
  <model name="brick_model">
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

    bricks_directives_lines = []
    for i in range(num_bricks):
        # rejection sampling for valid placement with clearances
        found = False
        for _ in range(200):
            x = float(np.random.uniform(x_min, x_max))
            y = float(np.random.uniform(y_min, y_max))
            if np.min(np.hypot(x - arm_bases[:, 0], y - arm_bases[:, 1])) < min_arm_clearance:
                continue
            if placed_xy:
                prev = np.array(placed_xy)
                if np.min(np.hypot(x - prev[:, 0], y - prev[:, 1])) < min_brick_spacing:
                    continue
            found = True
            break
        if not found:
            continue
        placed_xy.append((x, y))
        # Place on table top (z=0), with a small offset to avoid initial penetration
        z = 3.5
        yaw_deg = float(np.degrees(np.random.uniform(0.0, np.pi)))
        brick_sdf_path = bricks_dir / f"brick_{i}.sdf"
        # Always (re)write to ensure correct contact properties.
        write_brick_sdf(brick_sdf_path, brick_size)
        bricks_directives_lines.append(
            f"""
    - add_model:
        name: brick{i}
        file: file://{brick_sdf_path.resolve()}
        default_free_body_pose:
            brick_link:
                translation: [{x:.3f}, {y:.3f}, {z:.3f}]
                rotation: !Rpy {{ deg: [0.0, 0.0, {yaw_deg:.1f}]}}
"""
        )

    bricks_directives = "".join(bricks_directives_lines)

    cameras_block = """

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

    scenario_yaml = scenario_yaml_base + table_directives + bricks_directives + cameras_block

    print("Starting MeshCat… (check the VS Code Ports panel)")
    meshcat = StartMeshcat()

    scenario = LoadScenario(data=scenario_yaml)

    # TODO: Create HardwareStation with the scenario and meshcat
    station = MakeHardwareStation(scenario, meshcat)

    builder = DiagramBuilder()
    builder.AddSystem(station)
    # Add point cloud outputs for the three cameras (for ICP)
    try:
        to_point_cloud = AddPointClouds(
            scenario=scenario, station=station, builder=builder, meshcat=meshcat
        )
        builder.ExportOutput(to_point_cloud["camera0"].get_output_port(), "camera_point_cloud0")
        builder.ExportOutput(to_point_cloud["camera1"].get_output_port(), "camera_point_cloud1")
        builder.ExportOutput(to_point_cloud["camera2"].get_output_port(), "camera_point_cloud2")
    except Exception as e:
        print(f"Warning: AddPointClouds failed; cameras may be missing from scenario: {e}")
    diagram = builder.Build()

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(10)
    simulator.Initialize()
    simulator.AdvanceTo(30)

    # ---------------- ICP demo: merge camera clouds -> crop -> downsample -> register a brick ----------------
    try:
        context = simulator.get_context()
        # Grab point clouds exposed on the diagram
        pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
        pc1 = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
        pc2 = diagram.GetOutputPort("camera_point_cloud2").Eval(context)

        # Merge
        merged = Concatenate([pc0, pc1, pc2])
        # Crop by height to remove most ground and robot; keep expected brick band
        z = merged.xyzs()[2, :]
        keep = (z >= 0.02) & (z <= 0.25)
        cropped = PointCloud(int(np.sum(keep)))
        cropped.mutable_xyzs()[:] = merged.xyzs()[:, keep]
        # Optional: visualize merged/cropped scene cloud
        meshcat.SetObject("scene_cloud_cropped", cropped, point_size=0.02)

        # Downsample for faster ICP
        scene_cloud = cropped.VoxelizedDownSample(0.01)

        # Build a simple model point cloud for our brick (box), centered at origin
        def sample_box_surface(size_xyz, num_samples=2000) -> PointCloud:
            sx, sy, sz = size_xyz
            # Uniformly sample faces by area
            areas = np.array([sy*sz, sy*sz, sx*sz, sx*sz, sx*sy, sx*sy], dtype=float)
            probs = areas / np.sum(areas)
            face_ids = np.random.choice(6, size=num_samples, p=probs)
            u = np.random.uniform(-0.5, 0.5, size=num_samples)
            v = np.random.uniform(-0.5, 0.5, size=num_samples)
            pts = np.zeros((3, num_samples))
            # +/- X faces
            mask = face_ids == 0
            pts[:, mask] = np.vstack([np.full(np.sum(mask), +sx/2), sy*u[mask], sz*v[mask]])
            mask = face_ids == 1
            pts[:, mask] = np.vstack([np.full(np.sum(mask), -sx/2), sy*u[mask], sz*v[mask]])
            # +/- Y faces
            mask = face_ids == 2
            pts[:, mask] = np.vstack([sx*u[mask], np.full(np.sum(mask), +sy/2), sz*v[mask]])
            mask = face_ids == 3
            pts[:, mask] = np.vstack([sx*u[mask], np.full(np.sum(mask), -sy/2), sz*v[mask]])
            # +/- Z faces
            mask = face_ids == 4
            pts[:, mask] = np.vstack([sx*u[mask], sy*v[mask], np.full(np.sum(mask), +sz/2)])
            mask = face_ids == 5
            pts[:, mask] = np.vstack([sx*u[mask], sy*v[mask], np.full(np.sum(mask), -sz/2)])
            model = PointCloud(num_samples)
            model.mutable_xyzs()[:] = pts
            return model

        model_cloud = sample_box_surface(brick_size, num_samples=2500)
        meshcat.SetObject("brick_model_cloud", model_cloud, point_size=0.02)

        # Initial guess: put model at the scene centroid, identity rotation
        centroid = np.mean(scene_cloud.xyzs(), axis=1)
        X_init = RigidTransform(RotationMatrix.Identity(), centroid)

        # Run ICP
        model_pts = model_cloud.xyzs()
        scene_pts = scene_cloud.xyzs()
        X_brick_est, err = IterativeClosestPoint(
            p_Om=model_pts,
            p_Ws=scene_pts,
            X_Ohat=X_init,
            meshcat=meshcat,
            max_iterations=50,
        )
        print("ICP estimated brick pose (X_WO):", X_brick_est)
        print("ICP residual error:", err)
        # Show the aligned model
        meshcat.SetTransform("brick_model_aligned", X_brick_est)
        meshcat.SetObject("brick_model_aligned/pc", model_cloud, point_size=0.03)
        # Add a triad at the estimated pose for quick visual performance check.
        AddMeshcatTriad(meshcat, "icp_estimate_frame", X_brick_est, length=0.1, radius=0.003)

        place(brick_name="brick_0", X_WGoal=RigidTransform(RotationMatrix.Identity(), np.array([0.0, 0.0, 0.0])), sim_duration=12.0)
        place(brick_name="brick_1", X_WGoal=RigidTransform(RotationMatrix.Identity(), np.array([0.0, 0.0, 1.0])), sim_duration=12.0)
    except Exception as e:
        print(f"ICP pipeline warning: {e}")

    time.sleep(30)