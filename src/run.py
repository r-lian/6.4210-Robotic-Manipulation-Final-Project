from pydrake.all import StartMeshcat, DiagramBuilder, Simulator, RigidTransform, RotationMatrix, PointCloud, Concatenate, LeafSystem, MultibodyPlant, BasicVector, Context, JacobianWrtVariable, PiecewisePose, PiecewisePolynomial, Trajectory, TrajectorySource, Integrator, ConstantVectorSource, Rgba, Box
from manipulation.letter_generation import create_sdf_asset_from_letter
from manipulation.station import LoadScenario, MakeHardwareStation, AddPointClouds
from manipulation.icp import IterativeClosestPoint
from manipulation.meshcat_utils import AddMeshcatTriad
import time
import numpy as np
import os
from pathlib import Path
import traceback


if __name__ == "__main__":

    # ---------- Minimal Diff-IK utilities (not wired by default) ----------
    
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
                        <size>5 5 0.1</size>
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
                        <size>5 5 0.1</size>
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
            translation: [0.8, 0, 0]

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
            translation: [-0.8, 0, 0]

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
            translation: [0, 1.4, 0.5]
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
            translation: [1.4, 0.0, 0.5]
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
            translation: [-1.4, 0, 0.5]
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
            translation: [0, -1.4, 0.5]
    - add_model:  
        name: camera3
        file: package://manipulation/camera_box.sdf
    - add_weld:
        parent: camera3_origin
        child: camera3::base

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
    arm_bases = np.array([[0.8, 0.0], [-0.8, 0.0]])
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

    class PseudoInverseController(LeafSystem):
        """
        Differential IK: maps desired spatial velocity of the gripper V_WG (6) to
        joint velocities v (7 for iiwa14) via damped least-squares Jacobian inverse.
        Model and gripper body names are parameters to keep usage flexible.
        """
        def __init__(self, plant: MultibodyPlant, iiwa_model_name: str, gripper_body_name: str, damping: float = 1e-3, v_limit: float = 1.5):
            LeafSystem.__init__(self)
            self._plant = plant
            self._plant_context = plant.CreateDefaultContext()
            self._iiwa = plant.GetModelInstanceByName(iiwa_model_name)
            self._G = plant.GetBodyByName("body", plant.GetModelInstanceByName(gripper_body_name)).body_frame()
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
                np.array([0.0, 0.0, 0.0]),
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

    def place_from_pose(X_WO: RigidTransform, X_WGoal: RigidTransform, sim_duration: float = 12.0):
        """
        DEBUG: Pick/place using an ICP-estimated brick pose (no name lookup).
        Implements a simple antipodal grasp on the box: approach from +Z of the object and close across Y.
        """
        local_builder = DiagramBuilder()
        local_station = MakeHardwareStation(LoadScenario(data=scenario_yaml), meshcat)
        local_builder.AddSystem(local_station)
        plant = local_station.plant()
        # Select arm by XY distance
        p = X_WO.translation()
        left_base = np.array([0.8, 0.0]); right_base = np.array([-0.8, 0.0])
        use_left = np.linalg.norm(p[:2] - left_base) <= np.linalg.norm(p[:2] - right_base)
        iiwa_name = "iiwa_left" if use_left else "iiwa_right"
        wsg_name = "wsg_left" if use_left else "wsg_right"
        # Antipodal grasp w.r.t. object frame: gripper Z along -Z_object, fingers across Y_object
        R_WO = X_WO.rotation()
        R_OG = RotationMatrix.MakeZRotation(np.pi) @ RotationMatrix.MakeXRotation(-np.pi/2)  # align fingers with +Y_O
        X_WG_grasp = X_WO.multiply(RigidTransform(R_OG, [0.0, 0.0, brick_size[2] / 2.0 + 0.01]))
        hover_offset = 0.18
        X_WG_hover = RigidTransform(X_WG_grasp.rotation(), X_WG_grasp.translation() + np.array([0.0, 0.0, hover_offset]))
        # Goal poses
        p_goal_hover = X_WGoal.translation() + np.array([0.0, 0.0, hover_offset])
        X_WG_goal_hover = RigidTransform(X_WGoal.rotation(), p_goal_hover)
        X_WG_goal = X_WGoal
        # Keyframes
        X_Gs = [X_WG_hover, X_WG_grasp, X_WG_hover, X_WG_goal_hover, X_WG_goal, X_WG_goal_hover]
        opened = 0.107; closed = 0.0
        finger_values = np.asarray([opened, closed, closed, closed, opened, opened]).reshape(1, -1)
        sample_times = [0, 2, 4, 7, 9, 11]
        # Draw keyframes and path (debug)
        for i, X in enumerate(X_Gs):
            AddMeshcatTriad(meshcat, f"debug/icp_place/key_{i}", X_PT=X, length=0.12, radius=0.003)
        traj_pos = PiecewisePose.MakeLinear(sample_times, X_Gs)
        for s in range(len(sample_times) - 1):
            t0, t1 = sample_times[s], sample_times[s + 1]
            for j in range(1, 5):
                t = t0 + j * (t1 - t0) / 5.0
                pc_dot = PointCloud(1); pc_dot.mutable_xyzs()[:] = traj_pos.GetPose(t).translation().reshape(3, 1)
                meshcat.SetObject(f"debug/icp_place/path_{s}_{j}", pc_dot, point_size=0.05, rgba=Rgba(1, 0, 0, 1))
        # Wire DIK
        V_G_source = local_builder.AddSystem(TrajectorySource(traj_pos.MakeDerivative()))
        controller = local_builder.AddSystem(PseudoInverseController(plant=local_station.plant(), iiwa_model_name=iiwa_name, gripper_body_name=wsg_name))
        integrator = local_builder.AddSystem(Integrator(7))
        wsg_source = local_builder.AddSystem(TrajectorySource(PiecewisePolynomial.FirstOrderHold(sample_times, finger_values)))
        local_builder.Connect(V_G_source.get_output_port(), controller.get_input_port(0))
        local_builder.Connect(controller.get_output_port(), integrator.get_input_port())
        local_builder.Connect(integrator.get_output_port(), local_station.GetInputPort(f"{iiwa_name}.position"))
        local_builder.Connect(local_station.GetOutputPort(f"{iiwa_name}.position_measured"), controller.get_input_port(1))
        local_builder.Connect(wsg_source.get_output_port(), local_station.GetInputPort(f"{wsg_name}.position"))
        # Hold other arm
        other_iiwa = "iiwa_right" if iiwa_name == "iiwa_left" else "iiwa_left"
        other_wsg = "wsg_right" if wsg_name == "wsg_left" else "wsg_left"
        local_builder.Connect(local_station.GetOutputPort(f"{other_iiwa}.position_measured"), local_station.GetInputPort(f"{other_iiwa}.position"))
        local_builder.Connect(local_builder.AddSystem(ConstantVectorSource(np.array([0.107]))).get_output_port(), local_station.GetInputPort(f"{other_wsg}.position"))
        # Sim
        local_diagram = local_builder.Build()
        sim = Simulator(local_diagram); sim.set_target_realtime_rate(1.0)
        sim.Initialize(); sim.AdvanceTo(sim_duration)

    def estimate_brick_poses_from_scene(
        scene_cloud: PointCloud,
        model_cloud: PointCloud,
        meshcat=None,
        label_prefix: str = "debug/icp_multi",
        voxel_xy: float = 0.05,
        min_points: int = 80,
        expected_count: int = 10,
        base_clearance_m: float = 0.25,
        max_icp_err: float = 0.02,  # kept for API compatibility; not used in simplified filter
    ) -> list[RigidTransform]:
        """
        Segment the scene cloud into brick-sized clusters in XY using a coarse voxel grid,
        then run ICP per cluster to estimate X_WO for each brick.
        Returns a list of RigidTransform.
        """
        xyz = scene_cloud.xyzs()  # 3xN
        if xyz.shape[1] == 0:
            return []
        # RANSAC plane removal (remove table plane) to reduce outliers before clustering.
        P = xyz.T  # Nx3
        N = P.shape[0]
        if N >= 50:
            best_inliers = np.zeros(N, dtype=bool)
            best_count = 0
            iters = 200
            dist_thresh = 0.005  # 5 mm to plane (tighter to avoid eating bricks)
            for _ in range(iters):
                idx = np.random.choice(N, 3, replace=False)
                p1, p2, p3 = P[idx]
                v1 = p2 - p1
                v2 = p3 - p1
                n = np.cross(v1, v2)
                norm = np.linalg.norm(n)
                if norm < 1e-6:
                    continue
                n = n / norm
                # Prefer near-horizontal planes (table): normal close to +Z or -Z
                if abs(n[2]) < 0.95:
                    continue
                d = np.abs((P - p1) @ n)
                inliers = d < dist_thresh
                count = int(np.sum(inliers))
                if count > best_count:
                    best_count = count
                    best_inliers = inliers
            # Remove the table plane inliers if it is reasonably large
            if best_count > 0.05 * N:
                P = P[~best_inliers]
        # Simple density filtering via voxel counts to drop sparse outliers
        if P.shape[0] == 0:
            return []
        keys_all = np.floor(P[:, :2] / voxel_xy).astype(np.int64)
        from collections import defaultdict, deque
        cell_counts = defaultdict(int)
        for k in map(tuple, keys_all):
            cell_counts[k] += 1
        keep_mask = np.array([cell_counts[tuple(k)] >= 3 for k in keys_all], dtype=bool)
        P = P[keep_mask]
        if P.shape[0] == 0:
            return []
        xyz_f = P.T  # 3xM filtered
        # 2D voxel clustering on filtered XY
        xy = xyz_f[:2, :].T  # Mx2
        keys = np.floor(xy / voxel_xy).astype(np.int64)
        # Map cell -> indices
        cell_to_indices = defaultdict(list)
        for idx, k in enumerate(map(tuple, keys)):
            cell_to_indices[k].append(idx)
        cells = set(cell_to_indices.keys())
        visited = set()
        clusters = []
        # 8-neighbor connectivity
        neighbors = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
        for c in cells:
            if c in visited:
                continue
            q = deque([c]); visited.add(c)
            comp_cells = [c]
            while q:
                cx = q.popleft()
                for dx, dy in neighbors:
                    nb = (cx[0] + dx, cx[1] + dy)
                    if nb in cells and nb not in visited:
                        visited.add(nb)
                        q.append(nb)
                        comp_cells.append(nb)
            # Collect indices for this component
            comp_idx = np.concatenate([np.asarray(cell_to_indices[cell], dtype=int) for cell in comp_cells])
            if comp_idx.size >= min_points:
                clusters.append(comp_idx)
        # Known robot base locations to exclude spurious clusters near the pedestals.
        left_base_xy = np.array([0.8, 0.0])
        right_base_xy = np.array([-0.8, 0.0])
        results: list[tuple[RigidTransform, float, int]] = []  # (pose, icp_err, num_points)
        # Prepare model points
        model_pts = model_cloud.xyzs()
        # Estimate brick XY footprint radius from model cloud (half diagonal in XY) with a small margin
        x_span = float(np.max(model_pts[0, :]) - np.min(model_pts[0, :])) if model_pts.shape[1] else 0.2
        y_span = float(np.max(model_pts[1, :]) - np.min(model_pts[1, :])) if model_pts.shape[1] else 0.1
        r_brick = 0.5 * float(np.hypot(x_span, y_span)) + 0.01  # meters
        r_base = base_clearance_m  # interpret as base radius for overlap test
        for i, comp_idx in enumerate(clusters):
            pts = xyz_f[:, comp_idx]
            # Initial guess: centroid + yaw from PCA on XY
            centroid = np.mean(pts, axis=1)
            pts_xy = pts[:2, :].T - centroid[:2]
            # PCA via SVD
            try:
                u, s, vh = np.linalg.svd(pts_xy, full_matrices=False)
                axis = vh[0]  # principal direction in XY
                yaw = float(np.arctan2(axis[1], axis[0]))
            except Exception:
                yaw = 0.0
            X_init = RigidTransform(RotationMatrix.MakeZRotation(yaw), centroid)
            # ICP per cluster
            try:
                X_est, err = IterativeClosestPoint(
                    p_Om=model_pts,
                    p_Ws=pts,
                    X_Ohat=X_init,
                    meshcat=meshcat,
                    max_iterations=50,
                )
            except Exception:
                X_est, err = X_init, float("inf")
            # Overlap rejection: if estimated brick footprint overlaps either base footprint, skip
            pos_xy = X_est.translation()[:2]
            d_left = float(np.linalg.norm(pos_xy - left_base_xy))
            d_right = float(np.linalg.norm(pos_xy - right_base_xy))
            if min(d_left, d_right) <= (r_brick + r_base):
                continue
            # Keep for later ranking/filtering
            # err can be an array; reduce to a scalar (mean) for ranking/filtering
            try:
                err_scalar = float(np.mean(np.asarray(err).ravel()))
            except Exception:
                err_scalar = float(err) if np.isscalar(err) else float("inf")
            results.append((X_est, err_scalar, int(comp_idx.size)))
            # Debug viz
            if meshcat is not None:
                AddMeshcatTriad(meshcat, f"{label_prefix}/pose_{i}", X_PT=X_est, length=0.12, radius=0.003)
        # Simplified filtering: ONLY base clearance + min_points. Keep up to expected_count by lowest ICP error.
        filtered = sorted(results, key=lambda t: t[1])[: min(expected_count, len(results))]
        if meshcat is not None:
            # Print quick diagnostics to console
            print(f"[ICP] clusters={len(clusters)} kept={len(filtered)} (base_clearance {base_clearance_m}m)")
        return [pose for (pose, _, _) in filtered]


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
        z = brick_size[2] / 2.0 + 0.01
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

    camera3:
        name: camera3
        depth: True
        X_PB:
            base_frame: camera3::base

model_drivers:
    iiwa_left: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg_left
    wsg_left: !SchunkWsgDriver {}
    iiwa_right: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg_right
    wsg_right: !SchunkWsgDriver {}
    """

    scenario_yaml = scenario_yaml_base + table_directives + bricks_directives + cameras_block

    print("Starting MeshCat… (check the VS Code Ports panel)")
    meshcat = StartMeshcat()

    scenario = LoadScenario(data=scenario_yaml)

    # TODO: Create HardwareStation with the scenario and meshcat
    # Ensure a DISPLAY is set to avoid pyvirtualdisplay/Xvfb startup in headless CI.
    if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
        os.environ["DISPLAY"] = ":0"
    station = MakeHardwareStation(scenario, meshcat)

    builder = DiagramBuilder()
    builder.AddSystem(station)
    # Add point cloud outputs for the cameras (for ICP)
    try:
        to_point_cloud = AddPointClouds(
            scenario=scenario, station=station, builder=builder, meshcat=meshcat
        )
        builder.ExportOutput(to_point_cloud["camera0"].get_output_port(), "camera_point_cloud0")
        builder.ExportOutput(to_point_cloud["camera1"].get_output_port(), "camera_point_cloud1")
        builder.ExportOutput(to_point_cloud["camera2"].get_output_port(), "camera_point_cloud2")
        builder.ExportOutput(to_point_cloud["camera3"].get_output_port(), "camera_point_cloud3")
    except Exception as e:
        print(f"Warning: AddPointClouds failed; cameras may be missing from scenario: {e}")
    
    # Ensure required driver inputs are connected to hold posture (explicit connections).
    # Position hold: measured -> commanded
    builder.Connect(
        station.GetOutputPort("iiwa_left.position_measured"),
        station.GetInputPort("iiwa_left.position"),
    )
    builder.Connect(
        station.GetOutputPort("iiwa_right.position_measured"),
        station.GetInputPort("iiwa_right.position"),
    )

    # Open grippers by default
    builder.Connect(
        builder.AddSystem(ConstantVectorSource(np.array([0.107]))).get_output_port(),
        station.GetInputPort("wsg_left.position"),
    )
    builder.Connect(
        builder.AddSystem(ConstantVectorSource(np.array([0.107]))).get_output_port(),
        station.GetInputPort("wsg_right.position"),
    )

    diagram = builder.Build()

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(10)
    simulator.Initialize()
    # simulator.AdvanceTo(30)

    # ---------------- ICP -> antipodal grasp -> IK place ----------------
    try:
        context = simulator.get_context()
        # Grab point clouds exposed on the diagram
        pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
        pc1 = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
        pc2 = diagram.GetOutputPort("camera_point_cloud2").Eval(context)
        pc3 = diagram.GetOutputPort("camera_point_cloud3").Eval(context)

        # Merge
        merged = Concatenate([pc0, pc1, pc2, pc3])
        # Crop by height to remove most ground and robot; keep expected brick band
        z = merged.xyzs()[2, :]
        keep = (z >= 0.02) & (z <= 0.25)
        cropped = PointCloud(int(np.sum(keep)))
        cropped.mutable_xyzs()[:] = merged.xyzs()[:, keep]
        # Optional: visualize merged/cropped scene cloud
        meshcat.SetObject("debug/icp/scene_cropped", cropped, point_size=0.02)

        # Downsample for faster ICP
        scene_cloud = cropped.VoxelizedDownSample(0.01)

        # Build a simple model point cloud for our brick (box), centered at origin
        def sample_box_surface(size_xyz, num_samples=2000) -> PointCloud:
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

        model_cloud = sample_box_surface(brick_size, num_samples=2500)
        meshcat.SetObject("debug/icp/model_cloud", model_cloud, point_size=0.02)

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
        
        # Show the aligned model
        meshcat.SetTransform("debug/icp/aligned_pose", X_brick_est)
        meshcat.SetObject("debug/icp/aligned_pc", model_cloud, point_size=0.03)
        AddMeshcatTriad(meshcat, "debug/icp/pose", X_PT=X_brick_est, length=0.1, radius=0.003)

        # Multi-brick estimation via clustering + ICP, then run place on the first target
        print("Estimating brick poses...")
        try:
            poses = estimate_brick_poses_from_scene(
                scene_cloud=scene_cloud,
                model_cloud=model_cloud,
                meshcat=meshcat,
                label_prefix="debug/icp_multi",
            )
        except Exception as e:
            print(f"[ICP] estimate_brick_poses_from_scene failed: {e}")
            traceback.print_exc()
            poses = []
        print("poses:", len(poses))
        if len(poses) > 0:
            X_first = poses[0]
            print(len(poses), "poses")
            X_goal = RigidTransform(X_first.rotation(), X_first.translation() + np.array([0.20, 0.0, 0.0]))
            print("X_first:", X_first)
            print("X_goal:", X_goal)
            # DEBUG: overlay ICP poses as translucent red rectangular prisms (brick-sized)
            brick_box = Box(brick_size[0], brick_size[1], brick_size[2])
            meshcat.SetObject("debug/icp_multi/X_first_box", brick_box, Rgba(1, 0, 0, 0.4))  # red
            meshcat.SetTransform("debug/icp_multi/X_first_box", X_first)
            meshcat.SetObject("debug/icp_multi/X_goal_box", brick_box, Rgba(0, 0, 1, 0.4))  # blue
            meshcat.SetTransform("debug/icp_multi/X_goal_box", X_goal)
            AddMeshcatTriad(meshcat, "debug/icp_multi/X_first_tri", X_PT=X_first, length=0.12, radius=0.003)
            AddMeshcatTriad(meshcat, "debug/icp_multi/X_goal_tri", X_PT=X_goal, length=0.12, radius=0.003)
            place_from_pose(X_first, X_goal, sim_duration=12.0)
        else:
            print("ICP clustering found no bricks.")
    except Exception as e:
        print(f"ICP pipeline warning: {e}")

    

    time.sleep(30)