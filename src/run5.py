from manipulation.utils import RenderDiagram
import pydot
from pydrake.all import InverseKinematics, Solve, StartMeshcat, DiagramBuilder, Simulator, RigidTransform, RotationMatrix, PointCloud, Concatenate, LeafSystem, MultibodyPlant, BasicVector, Context, JacobianWrtVariable, PiecewisePose, PiecewisePolynomial, Trajectory, TrajectorySource, Integrator, ConstantVectorSource, Rgba, Box, RollPitchYaw, Adder
from manipulation.letter_generation import create_sdf_asset_from_letter
from manipulation.station import LoadScenario, MakeHardwareStation, AddPointClouds
from manipulation.icp import IterativeClosestPoint
from manipulation.meshcat_utils import AddMeshcatTriad
import time
import numpy as np
import os
from pathlib import Path
import traceback
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from enum import Enum


if __name__ == "__main__":

    # Ensure randomized brick placement each run
    try:
        seed_val = (int(time.time_ns()) ^ int.from_bytes(os.urandom(8), "little")) & 0xFFFFFFFF
        np.random.seed(seed_val)
        print(f"Random seed: {seed_val}")
    except Exception:
        np.random.seed(None)
        print("Random seed: None (using system time)")


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
                    <drake:mu_static>1.50</drake:mu_static>
                    <drake:mu_dynamic>1.20</drake:mu_dynamic>
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

    # ---------- Minimal Diff-IK controller and place() (basic version) ----------
    class PseudoInverseController(LeafSystem):
        def __init__(self, plant: MultibodyPlant, iiwa_model_name: str, wsg_model_name: str):
            LeafSystem.__init__(self)
            self._plant = plant
            self._plant_context = plant.CreateDefaultContext()
            self._iiwa = plant.GetModelInstanceByName(iiwa_model_name)
            self._G = plant.GetBodyByName("body", plant.GetModelInstanceByName(wsg_model_name)).body_frame()
            self._W = plant.world_frame()

            self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
            self.q_port = self.DeclareVectorInputPort("iiwa.position", 7)
            self.DeclareVectorOutputPort("iiwa.velocity", 7, self.CalcOutput)
            self.iiwa_start = plant.GetJointByName("iiwa_joint_1", self._iiwa).velocity_start()
            self.iiwa_end = plant.GetJointByName("iiwa_joint_7", self._iiwa).velocity_start()

        def CalcOutput(self, context: Context, output: BasicVector):
            """
            fill in our code below.
            """

            # evaluate the V_G_port and q_port on the current context to get those values.
            V_G = self.V_G_port.Eval(context)
            q = self.q_port.Eval(context)

            # update the positions of the internal _plant_context according to `q`.
            # HINT: you can write to a plant context by calling `self._plant.SetPositions`
            self._plant.SetPositions(self._plant_context, self._iiwa, q)

            # Compute the gripper jacobian
            # HINT: the jacobian is 6 x N, with N being the number of DOFs.
            # We only want the 6 x 7 submatrix corresponding to the IIWA
            J_G = self._plant.CalcJacobianSpatialVelocity(
                self._plant_context,
                JacobianWrtVariable.kV,
                self._G,
                [0,0,0],
                self._W,
                self._W
            )
            # compute `v` by mapping the gripper velocity (from the V_G_port) to the joint space
            J_G = J_G[:, self.iiwa_start:self.iiwa_end + 1]
            v = np.linalg.pinv(J_G) @ V_G
            # print(v)
            output.SetFromVector(v)

    # for normal IK with global optimization
    def solve_ik(
        X_WG: RigidTransform,
        max_tries: int = 10,
        base_pose: np.ndarray | None = None,
        iiwa_name: str = "iiwa_left",
        wsg_name: str = "wsg_left",
    ) -> np.ndarray | None:
        plant = station.plant()
        wsg_model = plant.GetModelInstanceByName(wsg_name)
        G_body = plant.GetBodyByName("body", wsg_model)
        
        # Note: passing in a plant_context is necessary for collision-free constraints!
        station_ctx = station.CreateDefaultContext()
        # Get the plant's subsystem Context from the station's Diagram Context
        try:
            plant_ctx = station.GetSubsystemContext(plant, station_ctx)
        except Exception:
            plant_ctx = plant.GetMyContextFromRoot(station_ctx)

        ik = InverseKinematics(plant, plant_ctx)
        # Active arm 7-DoF contiguous block (avoid fancy indexing on ik.q())
        iiwa_model = plant.GetModelInstanceByName(iiwa_name)
        start = plant.GetJointByName("iiwa_joint_1", iiwa_model).position_start()
        q_variables = ik.q()[start:start + 7]
        prog = ik.prog()
        prog.AddQuadraticErrorCost(np.eye(len(q_variables)), np.zeros(len(q_variables)), q_variables)

        ik.AddPositionConstraint(
            frameB=G_body.body_frame(),
            p_BQ=np.zeros(3),
            frameA=plant.world_frame(),
            p_AQ_lower=X_WG.translation() - np.array([0.001, 0.001, 0.001]),
            p_AQ_upper=X_WG.translation() + np.array([0.001, 0.001, 0.001]),
        )
        ik.AddOrientationConstraint(
            frameAbar=plant.world_frame(),
            R_AbarA=X_WG.rotation(),
            frameBbar=G_body.body_frame(),
            R_BbarB=RotationMatrix(),
            theta_bound=1 * np.pi / 180,
        )

        # Use smaller clearance for stacking (bricks need to be close)
        # 3mm clearance allows tight stacking while preventing penetration
        ik.AddMinimumDistanceLowerBoundConstraint(0.003, 1e-6)  # 3mm clearance

        for count in range(max_tries):
            # TODO: Compute a random initial guess here, within the joint limits of the robot
            q_guess = plant.GetPositions(plant_ctx, plant.GetModelInstanceByName(iiwa_name))
            lower_jl, upper_jl = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
            for i in range(len(q_variables)):
                if q_guess[i] > lower_jl[i] and q_guess[i] < upper_jl[i]:
                    continue
                if np.isinf(lower_jl[i]):
                    lower_jl[i] = -np.pi
                if np.isinf(upper_jl[i]):
                    upper_jl[i] = np.pi
                q_guess[i] = np.random.uniform(lower_jl[i], upper_jl[i])
            
            prog.SetInitialGuess(q_variables, q_guess)

            result = Solve(prog)

            if result.is_success():
                print("Succeeded in %d tries!" % (count + 1))
                q_full = result.GetSolution(q_variables)
                return q_full

        print("Failed!")
        return None

    def diffik(
        poses: list[RigidTransform],
        times: list[float],
        iiwa_name: str,
        wsg_name: str,
        finger_values: np.ndarray | None = None,
        q0: np.ndarray | None = None,
    ) -> None:
        traj_pos = PiecewisePose.MakeLinear(times, poses)
        traj_VG = traj_pos.MakeDerivative()

        # ---- DEBUG triads (unchanged) ----
        seg_id = int(time.time() * 1000)
        t0, tf = float(times[0]), float(times[-1])
        n_samples = max(6, min(30, 3 * len(times)))
        for k, tk in enumerate(np.linspace(t0, tf, n_samples)):
            AddMeshcatTriad(
                meshcat,
                f"debug/diffik/seg_{seg_id}_{k}",
                X_PT=traj_pos.GetPose(tk),
                length=0.05,
                radius=0.002,
            )

        # ---- USE EXISTING SIMULATOR + DIAGRAM ----
        ctx = simulator.get_mutable_context()
        plant = station.plant()
        plant_ctx = plant.GetMyContextFromRoot(ctx)
        iiwa_model = plant.GetModelInstanceByName(iiwa_name)

        # Seed q0 from current sim state if not provided
        if q0 is None:
            q0 = plant.GetPositions(plant_ctx, iiwa_model)
        else:
            q0 = np.asarray(q0, dtype=float)

        # Keep plant pose consistent before starting the trajectory
        plant.SetPositions(plant_ctx, iiwa_model, q0)

        # ---- Get controller + gripper command ports that were exported at build time ----
        if "left" in iiwa_name:
            V_G_port = diagram.GetInputPort("V_G_left_cmd")
            wsg_port_cmd = diagram.GetInputPort("wsg_left.position")
            diagram.GetInputPort("iiwa_left.source_select").FixValue(ctx, [1])
        else:
            V_G_port = diagram.GetInputPort("V_G_right_cmd")
            wsg_port_cmd = diagram.GetInputPort("wsg_right.position")
            diagram.GetInputPort("iiwa_right.source_select").FixValue(ctx, [1])

        # CRITICAL: Initialize integrator state to current position before starting diffik
        integrator_sys = integrator_left if "left" in iiwa_name else integrator_right
        integrator_ctx = diagram.GetMutableSubsystemContext(integrator_sys, ctx)
        integrator_sys.set_integral_value(integrator_ctx, q0)

        # Optional finger trajectory: interpolate scalar values over the same times
        finger_values_arr = None
        if finger_values is not None:
            finger_values_arr = np.asarray(finger_values, dtype=float).flatten()
            assert len(finger_values_arr) == len(times), \
                "finger_values must have same length as times"

        # Publish once so visuals don't snap when we start moving
        diagram.ForcedPublish(ctx)

        # ---- Run diff IK loop: send V_WG(t), step the sim ----
        base_time = ctx.get_time()

        # Use smaller timesteps for smoother velocity commands
        dt = 0.05  # 50ms timesteps for smoother tracking
        num_steps = int((tf - t0) / dt) + 1
        time_samples = np.linspace(t0, tf, num_steps)

        # Temporarily reduce publish rate for speed (publish every N steps)
        publish_interval = 5  # Only visualize every 5th step for speed

        for i, tk in enumerate(time_samples):
            tk = float(tk)
            target_time = base_time + (tk - t0)

            # Spatial velocity command from trajectory
            V_WG = traj_VG.value(tk).ravel()   # shape (6,)
            V_G_port.FixValue(ctx, V_WG)

            # Optional gripper command
            if finger_values_arr is not None:
                finger_cmd = float(np.interp(tk, times, finger_values_arr))
                wsg_port_cmd.FixValue(ctx, [finger_cmd])

            simulator.AdvanceTo(target_time)

            # Publish visualization only periodically for speed
            if i % publish_interval == 0 or i == len(time_samples) - 1:
                diagram.ForcedPublish(ctx)

    # RRT for collision-free joint-space path planning with time-based obstacle avoidance
    def rrt_plan(
        q_start: np.ndarray,
        q_goal: np.ndarray,
        iiwa_name: str,
        max_iterations: int = 1000,
        step_size: float = 0.2,
        goal_bias: float = 0.15,
        other_arm_trajectory: list[np.ndarray] | None = None,
    ) -> list[np.ndarray] | None:
        """
        Simple RRT planner in joint space with time-based collision checking.
        Checks collision against other arm's reserved positions at matching simulation times.
        Returns path from q_start to q_goal, or None if no path found.
        """
        plant = station.plant()
        iiwa_model = plant.GetModelInstanceByName(iiwa_name)

        # Get other arm info and current planning start time
        other_iiwa_name = "iiwa_right" if iiwa_name == "iiwa_left" else "iiwa_left"
        other_iiwa_model = plant.GetModelInstanceByName(other_iiwa_name)
        other_arm_name = "right" if iiwa_name == "iiwa_left" else "left"

        # Get current simulation time as planning start time
        ctx = simulator.get_mutable_context()
        t_planning_start = ctx.get_time()

        # Create a separate context for collision checking
        station_ctx = station.CreateDefaultContext()
        try:
            plant_ctx = station.GetSubsystemContext(plant, station_ctx)
        except Exception:
            plant_ctx = plant.GetMyContextFromRoot(station_ctx)

        # Get current position of other arm for fallback
        sim_plant_ctx = plant.GetMyContextFromRoot(ctx)
        q_other_current = plant.GetPositions(sim_plant_ctx, other_iiwa_model)

        # Collision checker using time-based obstacle reservations
        def is_collision_free(q, estimated_time_from_start):
            """
            Check collision for configuration q against other arm's reserved position.
            estimated_time_from_start: float in seconds, how long into the future this config will be reached
            """
            plant.SetPositions(plant_ctx, iiwa_model, q)

            # Look up other arm's position at this future time using obstacle reservations
            t_future = t_planning_start + estimated_time_from_start
            t_bucket = int(t_future * 10)  # 100ms resolution

            if t_bucket in obstacle_reservations and other_arm_name in obstacle_reservations[t_bucket]:
                q_other = obstacle_reservations[t_bucket][other_arm_name]
            else:
                # No reservation at this time - use current position as fallback
                q_other = q_other_current

            plant.SetPositions(plant_ctx, other_iiwa_model, q_other)

            query_object = plant.get_geometry_query_input_port().Eval(plant_ctx)
            distances = query_object.ComputeSignedDistancePairwiseClosestPoints()
            for pair in distances:
                if pair.distance < 0.01:  # 1cm clearance
                    return False
            return True

        # Check if start is collision-free
        if not is_collision_free(q_start, 0.0):
            print(f"RRT: Start config in collision for {iiwa_name}")
            return None

        # Joint limits
        lower_limits = plant.GetPositionLowerLimits()[plant.GetJointByName("iiwa_joint_1", iiwa_model).position_start():][:7]
        upper_limits = plant.GetPositionUpperLimits()[plant.GetJointByName("iiwa_joint_1", iiwa_model).position_start():][:7]

        # RRT tree: list of (config, parent_index, estimated_time_from_start)
        tree = [(q_start.copy(), -1, 0.0)]

        # Estimate execution time per radian of joint motion (rough heuristic)
        time_per_radian = 0.08  # ~0.08s per radian of joint motion

        for iteration in range(max_iterations):
            # Sample random configuration (with goal bias)
            if np.random.rand() < goal_bias:
                q_rand = q_goal.copy()
            else:
                q_rand = np.random.uniform(lower_limits, upper_limits)

            # Find nearest node in tree
            nearest_idx = min(range(len(tree)), key=lambda i: np.linalg.norm(tree[i][0] - q_rand))
            q_nearest, _, time_nearest = tree[nearest_idx]

            # Steer toward q_rand
            direction = q_rand - q_nearest
            dist = np.linalg.norm(direction)
            if dist > step_size:
                q_new = q_nearest + (direction / dist) * step_size
                step_taken = step_size
            else:
                q_new = q_rand
                step_taken = dist

            # Estimate time to reach q_new from q_nearest
            time_for_step = step_taken * time_per_radian
            time_new = time_nearest + time_for_step

            # Collision check using time-based obstacle lookup
            if not is_collision_free(q_new, time_new):
                continue

            # Add to tree with cumulative estimated time
            tree.append((q_new.copy(), nearest_idx, time_new))

            # Check if we reached goal
            if np.linalg.norm(q_new - q_goal) < 0.15:  # 0.15 rad tolerance
                # Extract path by backtracking
                path = []
                idx = len(tree) - 1
                while idx != -1:
                    path.append(tree[idx][0])
                    idx = tree[idx][1]
                path.reverse()
                print(f"RRT found path with {len(path)} waypoints in {iteration + 1} iterations for {iiwa_name}")

                # Save the complete RRT tree for visualization
                import pickle
                tree_data = {
                    'tree': tree,  # list of (q, parent_idx, time)
                    'path': path,
                    'iiwa_name': iiwa_name,
                    'q_start': q_start,
                    'q_goal': q_goal,
                }
                tree_file = f"rrt_tree_{iiwa_name}.pkl"
                with open(tree_file, 'wb') as f:
                    pickle.dump(tree_data, f)
                print(f"Saved RRT tree with {len(tree)} nodes to {tree_file}")

                return path

        print(f"RRT failed for {iiwa_name} after {max_iterations} iterations")
        return None

    # Minimal joint-space polynomial interpolation to a target q
    def joint_poly_interp_to_q(
        q_target: np.ndarray | None,
        iiwa_name: str,
        wsg_name: str,
        duration_s: float = 0.8,  # Reduced from 1.5s for speed
        opened: float = 0.107,
    ) -> None:
        if q_target is None or q_target.shape[0] != 7:
            return
        # Use the single global station/simulator; compute absolute-time cubic interpolation and step the sim.
        ctx = simulator.get_mutable_context()
        plant = station.plant()
        plant_ctx = plant.GetMyContextFromRoot(ctx)
        iiwa_model = plant.GetModelInstanceByName(iiwa_name)
        q_now = plant.GetPositions(plant_ctx, iiwa_model)
        t0 = ctx.get_time()
        tf = t0 + float(duration_s)

        # Use CubicWithContinuousAcceleration for smoother motion with constrained acceleration
        # Set zero velocity at start and end to minimize acceleration spikes
        q_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            [t0, tf],
            np.column_stack([q_now, q_target]),
            np.zeros(7),  # zero velocity at start
            np.zeros(7)   # zero velocity at end
        )
        source = iiwa_left_trajectory_source if iiwa_name == "iiwa_left" else iiwa_right_trajectory_source
        source.UpdateTrajectory(q_traj)
        diagram.GetInputPort(f"{iiwa_name}.source_select").FixValue(ctx, [0])

        # Set gripper position
        wsg_port = diagram.GetInputPort(f"{wsg_name}.position")
        wsg_port.FixValue(ctx, [opened])

        # Step through with periodic publishing for speed
        num_steps = 10  # Fixed number of intermediate steps
        dt_step = (tf - t0) / num_steps
        for i in range(num_steps + 1):
            t_next = t0 + i * dt_step
            simulator.AdvanceTo(t_next)
            if i % 3 == 0:  # Publish every 3rd step
                diagram.ForcedPublish(ctx)


    # --------- Analytic antipodal grasp for a box in its own frame (O) ----------
    def compute_antipodal_grasp_box_O(
        size_oxo: list[float] | np.ndarray,
        prefer_axis: str | None = None,
        finger_clearance_m: float = 0.10,
    ) -> RigidTransform:
        """
        Construct an analytic antipodal top-down pinch grasp for an axis-aligned box in object frame O.
        Gripper G axes:
          - x_G: closing direction, aligned with +e_x (pinch on ±X faces) or +e_y (pinch on ±Y faces).
          - y_G: -e_z (approach downwards in world after composing with X_WO).
          - z_G: x_G × y_G (right-hand).
        G origin at box center, then offset by -finger_clearance along G.y (i.e., +z_O) to be above the part.
        """
        sx, sy, sz = float(size_oxo[0]), float(size_oxo[1]), float(size_oxo[2])
        # choose pinch axis (default to smaller horizontal extent; tie -> x)
        axis = prefer_axis if prefer_axis in ("x", "y") else ("x" if sx <= sy else "y")
        if axis == "y":
            xg_o = np.array([0.0, 1.0, 0.0])
        else:
            xg_o = np.array([1.0, 0.0, 0.0])
        yg_o = np.array([0.0, 0.0, -1.0])
        zg_o = np.cross(xg_o, yg_o); zg_o /= np.linalg.norm(zg_o)
        yg_o = np.cross(zg_o, xg_o)  # re-orthogonalize
        R_OG = RotationMatrix(np.column_stack((xg_o, yg_o, zg_o)))
        p_OG_o = np.zeros(3)
        # offset along -y_G (i.e., +z_O) to create a hover/clearance
        return RigidTransform(R_OG, p_OG_o) @ RigidTransform([0.0, -abs(finger_clearance_m), 0.0])
    def place(X_WO: RigidTransform, X_WGoal: RigidTransform, sim_duration: float = 8.0):
        """
        Basic pick/place using a minimal DIK controller.
        - Chooses the nearer arm by XY distance to the brick.
        - Builds a simple hover -> grasp -> hover -> goal-hover -> goal -> goal-hover waypoint trajectory.
        - Wires DIK, runs a short sim.
        """
        # Use global station/simulator (no local builders)
        plant = station.plant()
        # Choose arm by proximity
        p = np.array(X_WO.translation())
        left_base = np.array([0.65, 0.0])
        right_base = np.array([-0.65, 0.0])
        use_left = np.linalg.norm(p[:2] - left_base) <= np.linalg.norm(p[:2] - right_base)
        iiwa_name = "iiwa_left" if use_left else "iiwa_right"
        wsg_name = "wsg_left" if use_left else "wsg_right"

        # Waypoints (use analytic antipodal grasp for the box)
        hover = 0.18
        ctx = simulator.get_mutable_context()
        plant_ctx = plant.GetMyContextFromRoot(ctx)
        X_WGinitial = plant.EvalBodyPoseInWorld(plant_ctx, plant.GetBodyByName("body", plant.GetModelInstanceByName(wsg_name)))
        X_OG = compute_antipodal_grasp_box_O(brick_size)
        X_grasp = X_WO @ X_OG
        X_hover = RigidTransform(X_grasp.rotation(), X_grasp.translation() + np.array([0.0, 0.0, hover]))
        X_WGgoal = X_WGoal @ X_OG 
        X_WGgoal_hover = RigidTransform(X_WGgoal.rotation(), X_WGgoal.translation() + np.array([0.0, 0.0, hover]))

        # Gripper (open/close) profile
        opened = 0.107
        closed = 0.0

        # 1) IK: initial -> prepick hover
        q_pre_pick = solve_ik(X_hover, iiwa_name=iiwa_name, wsg_name=wsg_name)
        joint_poly_interp_to_q(q_pre_pick, iiwa_name, wsg_name, duration_s=0.8, opened=opened)

        # 2) DIK: prepick -> grasp -> postpick (faster timing)
        poses_pick = [X_hover, X_grasp, X_grasp, X_hover]
        times_pick = [0.0, 0.8, 1.2, 2.0]  # Reduced from [0.0, 1.5, 3.0, 4.0]
        fingers_pick = np.asarray([opened, opened, closed, closed]).reshape(1, -1)
        diffik(poses_pick, times_pick, iiwa_name, wsg_name, fingers_pick, q0=q_pre_pick if q_pre_pick is not None else None)
        
        # 3) IK: postpick -> preplace hover
        q_pre_place = solve_ik(X_WGgoal_hover, iiwa_name=iiwa_name, wsg_name=wsg_name)
        # Polynomial interpolate joints to the preplace hover (keep gripper closed!)
        # Slower duration for smoother transport with held brick
        joint_poly_interp_to_q(q_pre_place, iiwa_name, wsg_name, duration_s=1.5, opened=closed)
        # 4) DIK: preplace -> place -> postplace
        poses_place = [X_WGgoal_hover, X_WGgoal, X_WGgoal, X_WGgoal_hover]
        times_place = [0.0, 0.8, 1.2, 2.0]  # Reduced from [0.0, 1.5, 3.0]
        fingers_place = np.asarray([closed, closed, opened, opened]).reshape(1, -1)
        diffik(poses_place, times_place, iiwa_name, wsg_name, fingers_place, q0=q_pre_place if q_pre_place is not None else None)

        # Reset inputs and trajectories to clean state after place operation
        ctx = simulator.get_mutable_context()
        plant = station.plant()
        plant_ctx = plant.GetMyContextFromRoot(ctx)

        # Get current joint positions for both arms
        iiwa_left_model = plant.GetModelInstanceByName("iiwa_left")
        iiwa_right_model = plant.GetModelInstanceByName("iiwa_right")
        q_left_current = plant.GetPositions(plant_ctx, iiwa_left_model)
        q_right_current = plant.GetPositions(plant_ctx, iiwa_right_model)

        # Return to initial positions (defined in scenario YAML) instead of holding at current
        t_now = ctx.get_time()
        t_reset = t_now + 2.0  # 2 seconds to return to initial position

        # Use smooth cubic trajectory back to initial positions with zero velocity at endpoints
        q_left_return = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            [t_now, t_reset],
            np.column_stack([q_left_current, qL0]),
            np.zeros(7),  # zero velocity at start
            np.zeros(7)   # zero velocity at end
        )
        q_right_return = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            [t_now, t_reset],
            np.column_stack([q_right_current, qR0]),
            np.zeros(7),  # zero velocity at start
            np.zeros(7)   # zero velocity at end
        )

        # Update trajectory sources to return to initial positions
        iiwa_left_trajectory_source.UpdateTrajectory(q_left_return)
        iiwa_right_trajectory_source.UpdateTrajectory(q_right_return)

        # Reset to trajectory source mode (not diffik mode)
        diagram.GetInputPort("iiwa_left.source_select").FixValue(ctx, [0])
        diagram.GetInputPort("iiwa_right.source_select").FixValue(ctx, [0])

        # Reset velocity commands to zero
        diagram.GetInputPort("V_G_left_cmd").FixValue(ctx, np.zeros(6))
        diagram.GetInputPort("V_G_right_cmd").FixValue(ctx, np.zeros(6))

        # Reset grippers to open position
        diagram.GetInputPort("wsg_left.position").FixValue(ctx, [opened])
        diagram.GetInputPort("wsg_right.position").FixValue(ctx, [opened])

    # ========== EVENT LOOP ARCHITECTURE WITH SEMAPHORE CONTROL ==========

    # Arm state for tracking current execution phase
    class ArmState(Enum):
        IDLE = "idle"
        PLANNING = "planning"
        EXECUTING = "executing"

    # Task for an arm
    @dataclass
    class ArmTask:
        X_source: RigidTransform
        X_goal: RigidTransform
        brick_idx: int

    # Global state
    left_task_queue = Queue()
    right_task_queue = Queue()
    obstacle_reservations = {}  # time_step -> {"left": q, "right": q}

    # Arm states
    arm_states = {"left": ArmState.IDLE, "right": ArmState.IDLE}
    arm_plans = {"left": None, "right": None}

    # Background planning threads
    arm_planning_threads = {"left": None, "right": None}
    arm_planning_locks = {"left": threading.Lock(), "right": threading.Lock()}

    def plan_arm_task(arm_name: str, task: ArmTask):
        """
        PLAN phase: Compute IK, run RRT with obstacle avoidance, reserve obstacle space.
        This does NOT advance the simulator - it only computes the plan.
        Returns: dict with trajectory waypoints, or None if planning failed
        """
        iiwa_name = f"iiwa_{arm_name}"
        wsg_name = f"wsg_{arm_name}"

        plant = station.plant()
        ctx = simulator.get_mutable_context()
        plant_ctx = plant.GetMyContextFromRoot(ctx)
        iiwa_model = plant.GetModelInstanceByName(iiwa_name)

        # Compute grasp poses
        X_OG = compute_antipodal_grasp_box_O(brick_size)
        X_grasp = task.X_source @ X_OG
        X_prepick = RigidTransform(X_grasp.rotation(), X_grasp.translation() + np.array([0, 0, 0.18]))
        X_goal_grasp = task.X_goal @ X_OG
        X_preplace = RigidTransform(X_goal_grasp.rotation(), X_goal_grasp.translation() + np.array([0, 0, 0.18]))

        # IK for key waypoints
        q_current = plant.GetPositions(plant_ctx, iiwa_model)
        q_prepick = solve_ik(X_prepick, iiwa_name=iiwa_name, wsg_name=wsg_name)
        q_preplace = solve_ik(X_preplace, iiwa_name=iiwa_name, wsg_name=wsg_name)

        if q_prepick is None or q_preplace is None:
            print(f"[{arm_name}] IK failed for brick {task.brick_idx}")
            return None

        # Build obstacle trajectory from OTHER arm's reservations
        other_name = "right" if arm_name == "left" else "left"
        other_trajectory = []
        for t_step in sorted(obstacle_reservations.keys()):
            if other_name in obstacle_reservations[t_step]:
                other_trajectory.append(obstacle_reservations[t_step][other_name])

        # RRT to prepick with obstacle avoidance
        rrt_path = rrt_plan(q_current, q_prepick, iiwa_name, max_iterations=500,
                           other_arm_trajectory=other_trajectory if other_trajectory else None)

        if not rrt_path or len(rrt_path) < 2:
            print(f"[{arm_name}] RRT failed, using direct path")
            rrt_path = [q_current, q_prepick]

        # Reserve obstacle space (use actual simulation time, not indices)
        # CRITICAL: Reserve DENSELY (every 100ms) to cover interpolated motion between waypoints
        ctx = simulator.get_mutable_context()
        t_start = ctx.get_time()
        path_duration = len(rrt_path) * 0.6

        # Create dense interpolation of the path: sample every 100ms
        reservation_dt = 0.1  # Reserve every 100ms
        num_samples = int(path_duration / reservation_dt) + 1

        for sample_idx in range(num_samples):
            t_sample = t_start + sample_idx * reservation_dt
            t_bucket = int(t_sample * 10)  # 100ms resolution

            # Interpolate configuration at this time
            # Find which waypoint segment we're in
            t_rel = sample_idx * reservation_dt  # Time from start of path
            waypoint_idx_float = t_rel / path_duration * (len(rrt_path) - 1)
            waypoint_idx = int(waypoint_idx_float)
            waypoint_idx = min(waypoint_idx, len(rrt_path) - 2)  # Don't exceed bounds

            # Linear interpolation between waypoints
            alpha = waypoint_idx_float - waypoint_idx
            q_interp = (1 - alpha) * rrt_path[waypoint_idx] + alpha * rrt_path[waypoint_idx + 1]

            if t_bucket not in obstacle_reservations:
                obstacle_reservations[t_bucket] = {}
            obstacle_reservations[t_bucket][arm_name] = q_interp

        print(f"[{arm_name}] Planned brick {task.brick_idx}: RRT {len(rrt_path)} waypoints, reserved {num_samples} time buckets (dense sampling)")

        return {
            "rrt_path": rrt_path,
            "q_prepick": q_prepick,
            "q_preplace": q_preplace,
            "X_prepick": X_prepick,
            "X_grasp": X_grasp,
            "X_preplace": X_preplace,
            "X_goal_grasp": X_goal_grasp,
            "task": task,
        }

    def load_arm_inputs_nonblocking(arm_name: str, plan: dict):
        """
        Load inputs (trajectory sources, gripper commands) for an arm WITHOUT advancing simulator.
        This sets up what the arm should do, but the main loop will execute it.
        """
        iiwa_name = f"iiwa_{arm_name}"
        wsg_name = f"wsg_{arm_name}"
        source = iiwa_left_trajectory_source if arm_name == "left" else iiwa_right_trajectory_source
        opened, closed = 0.107, 0.0

        ctx = simulator.get_mutable_context()
        t_now = ctx.get_time()

        # Load RRT path trajectory
        rrt_path = plan["rrt_path"]
        duration = len(rrt_path) * 0.6
        times = np.linspace(t_now, t_now + duration, len(rrt_path))
        q_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            times, np.column_stack(rrt_path), np.zeros(7), np.zeros(7))

        source.UpdateTrajectory(q_traj)
        diagram.GetInputPort(f"{iiwa_name}.source_select").FixValue(ctx, [0])
        diagram.GetInputPort(f"{wsg_name}.position").FixValue(ctx, [opened])

        return t_now + duration  # Return when this motion will complete

    # Execution phases for state machine
    class ExecutionPhase(Enum):
        IDLE = "idle"
        MOVING_TO_PREPICK = "moving_to_prepick"
        PICKING = "picking"
        TRANSPORTING = "transporting"
        PLACING = "placing"
        RETURNING_HOME = "returning_home"

    # Track execution state per arm
    arm_exec_phases = {"left": ExecutionPhase.IDLE, "right": ExecutionPhase.IDLE}
    arm_phase_end_times = {"left": 0.0, "right": 0.0}

    # ========== INPUT-ONLY MOTION PRIMITIVES (NO SIMULATOR ADVANCE) ==========

    def set_joint_trajectory_inputs(
        arm_name: str,
        q_target: np.ndarray,
        duration_s: float,
        gripper_pos: float = 0.107,
    ) -> None:
        """
        Set up joint trajectory inputs for an arm WITHOUT advancing simulator.
        """
        iiwa_name = f"iiwa_{arm_name}"
        wsg_name = f"wsg_{arm_name}"

        if q_target is None or q_target.shape[0] != 7:
            return

        ctx = simulator.get_mutable_context()
        plant = station.plant()
        plant_ctx = plant.GetMyContextFromRoot(ctx)
        iiwa_model = plant.GetModelInstanceByName(iiwa_name)
        q_now = plant.GetPositions(plant_ctx, iiwa_model)
        t0 = ctx.get_time()
        tf = t0 + float(duration_s)

        # Use CubicWithContinuousSecondDerivatives for smoother motion
        q_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            [t0, tf],
            np.column_stack([q_now, q_target]),
            np.zeros(7),
            np.zeros(7)
        )

        source = iiwa_left_trajectory_source if arm_name == "left" else iiwa_right_trajectory_source
        source.UpdateTrajectory(q_traj)
        diagram.GetInputPort(f"{iiwa_name}.source_select").FixValue(ctx, [0])
        diagram.GetInputPort(f"{wsg_name}.position").FixValue(ctx, [gripper_pos])

    def set_diffik_velocity_inputs(
        arm_name: str,
        traj_VG: Trajectory,
        t_start: float,
        t_end: float,
        finger_values: list[float],
        finger_times: list[float],
        q0: np.ndarray | None = None,
    ) -> None:
        """
        Set up differential IK velocity inputs for an arm WITHOUT advancing simulator.
        The centralized loop will read traj_VG.value(t) and set V_G_port at each timestep.
        """
        iiwa_name = f"iiwa_{arm_name}"
        wsg_name = f"wsg_{arm_name}"

        ctx = simulator.get_mutable_context()
        plant = station.plant()
        plant_ctx = plant.GetMyContextFromRoot(ctx)
        iiwa_model = plant.GetModelInstanceByName(iiwa_name)

        # Seed q0 from current sim state if not provided
        if q0 is None:
            q0 = plant.GetPositions(plant_ctx, iiwa_model)
        else:
            q0 = np.asarray(q0, dtype=float)

        # Switch to diffik mode
        if arm_name == "left":
            diagram.GetInputPort("iiwa_left.source_select").FixValue(ctx, [1])
            integrator_sys = integrator_left
        else:
            diagram.GetInputPort("iiwa_right.source_select").FixValue(ctx, [1])
            integrator_sys = integrator_right

        # CRITICAL: Initialize integrator state to current position before starting diffik
        integrator_ctx = diagram.GetMutableSubsystemContext(integrator_sys, ctx)
        integrator_sys.set_integral_value(integrator_ctx, q0)

    def setup_arm_motion(arm_name: str, plan: dict):
        """
        Set up the complete pick-place motion for one arm.
        This ONLY sets inputs for the FIRST phase - does NOT advance simulator.
        Returns phase schedule: list of (phase, duration, data) tuples.
        The centralized loop will execute these phases sequentially.
        """
        iiwa_name = f"iiwa_{arm_name}"
        wsg_name = f"wsg_{arm_name}"
        opened, closed = 0.107, 0.0

        # Extract plan data
        rrt_path = plan["rrt_path"]
        q_prepick = plan["q_prepick"]
        q_preplace = plan["q_preplace"]
        X_prepick = plan["X_prepick"]
        X_grasp = plan["X_grasp"]
        X_preplace = plan["X_preplace"]
        X_goal_grasp = plan["X_goal_grasp"]

        # Build complete phase schedule (all phases with durations and data)
        phase_schedule = []

        # Phase 1: Move to prepick via RRT path
        phase_schedule.append((
            ExecutionPhase.MOVING_TO_PREPICK,
            len(rrt_path) * 0.6,  # duration
            {"q_path": rrt_path, "gripper": opened}
        ))

        # Phase 2: Pick (diffik: prepick -> grasp -> prepick)
        phase_schedule.append((
            ExecutionPhase.PICKING,
            2.0,  # duration
            {
                "poses": [X_prepick, X_grasp, X_grasp, X_prepick],
                "times": [0.0, 0.8, 1.2, 2.0],
                "fingers": [opened, opened, closed, closed],
                "q0": q_prepick
            }
        ))

        # Phase 3: Transport to preplace
        phase_schedule.append((
            ExecutionPhase.TRANSPORTING,
            1.5,  # duration
            {"q_target": q_preplace, "gripper": closed}
        ))

        # Phase 4: Place (diffik: preplace -> goal -> preplace)
        phase_schedule.append((
            ExecutionPhase.PLACING,
            2.0,  # duration
            {
                "poses": [X_preplace, X_goal_grasp, X_goal_grasp, X_preplace],
                "times": [0.0, 0.8, 1.2, 2.0],
                "fingers": [closed, closed, opened, opened],
                "q0": q_preplace
            }
        ))

        # Phase 5: Return home
        q_home = qL0 if arm_name == "left" else qR0
        phase_schedule.append((
            ExecutionPhase.RETURNING_HOME,
            2.0,  # duration
            {"q_target": q_home, "gripper": opened}
        ))

        # Set up inputs for first phase ONLY (subsequent phases will be set up as they start)
        first_phase, first_duration, first_data = phase_schedule[0]
        set_joint_trajectory_inputs(arm_name, np.array(first_data["q_path"][-1]), first_duration, first_data["gripper"])

        print(f"[{arm_name}] Set up phase schedule with {len(phase_schedule)} phases")
        return phase_schedule

    def background_planning_worker(arm_name: str, task: ArmTask):
        """
        Background thread worker function for planning.
        Runs RRT and IK planning WITHOUT blocking the simulator.
        """
        global arm_states, arm_plans, obstacle_reservations, arm_planning_locks

        import time as time_module
        queue = left_task_queue if arm_name == "left" else right_task_queue

        try:
            t_start_wall = time_module.time()
            print(f"[{arm_name}] Background planning started for brick {task.brick_idx}")

            # PLANNING phase: compute plan (no simulator advance)
            plan = plan_arm_task(arm_name, task)

            t_end_wall = time_module.time()
            planning_duration_wall = t_end_wall - t_start_wall
            print(f"[{arm_name}] Background planning took {planning_duration_wall:.3f}s (wall time)")

            # Atomically update state with lock
            with arm_planning_locks[arm_name]:
                if plan is None:
                    print(f"[{arm_name}] Planning failed, re-queuing task brick {task.brick_idx}")
                    queue.put(task)
                    arm_states[arm_name] = ArmState.IDLE
                else:
                    arm_plans[arm_name] = plan
                    arm_states[arm_name] = ArmState.EXECUTING
                    print(f"[{arm_name}] Background planning complete, ready to execute")

        except Exception as e:
            print(f"[{arm_name}] Background planning exception: {e}, re-queuing task brick {task.brick_idx}")
            import traceback
            traceback.print_exc()
            with arm_planning_locks[arm_name]:
                queue.put(task)
                arm_states[arm_name] = ArmState.IDLE

    def process_arm_planning(arm_name: str):
        """
        Process planning phase for an arm by launching background thread.
        Returns True if planning work was started, False if idle or already executing
        """
        global arm_states, arm_plans, obstacle_reservations, arm_planning_threads, arm_planning_locks

        queue = left_task_queue if arm_name == "left" else right_task_queue

        # Only plan if IDLE
        with arm_planning_locks[arm_name]:
            if arm_states[arm_name] == ArmState.IDLE:
                try:
                    task = queue.get(block=False)
                    ctx = simulator.get_mutable_context()
                    t_start_planning = ctx.get_time()
                    print(f"[{arm_name}] t={t_start_planning:.2f}s Got task: brick {task.brick_idx}, queue_remaining={queue.qsize()}")
                    arm_states[arm_name] = ArmState.PLANNING
                    print(f"[{arm_name}] Launching background planning thread for brick {task.brick_idx}")

                    # Launch background planning thread
                    planning_thread = threading.Thread(
                        target=background_planning_worker,
                        args=(arm_name, task),
                        daemon=True
                    )
                    planning_thread.start()
                    arm_planning_threads[arm_name] = planning_thread
                    return True

                except Empty:
                    # No tasks available
                    return False

        return False

    # Track per-arm phase schedules and current phase
    arm_phase_schedules = {"left": None, "right": None}
    arm_current_phase_idx = {"left": 0, "right": 0}
    arm_diffik_trajectories = {"left": None, "right": None}  # Store diffik trajectories for continuous update

    def start_arm_execution(arm_name: str):
        """
        Initiate execution for an arm by setting up its phase schedule.
        This ONLY sets inputs for the first phase - does NOT advance simulator.
        """
        global arm_states, arm_plans, arm_phase_schedules, arm_current_phase_idx, arm_phase_end_times

        if arm_states[arm_name] != ArmState.EXECUTING:
            return False

        plan = arm_plans[arm_name]
        print(f"[{arm_name}] Starting execution of brick {plan['task'].brick_idx}")

        # Get phase schedule from setup_arm_motion
        phase_schedule = setup_arm_motion(arm_name, plan)
        arm_phase_schedules[arm_name] = phase_schedule
        arm_current_phase_idx[arm_name] = 0

        # Set end time for first phase
        ctx = simulator.get_mutable_context()
        t_now = ctx.get_time()
        first_phase, first_duration, first_data = phase_schedule[0]
        arm_phase_end_times[arm_name] = t_now + first_duration
        arm_exec_phases[arm_name] = first_phase

        print(f"[{arm_name}] Started phase {first_phase.value}, ends at t={arm_phase_end_times[arm_name]:.2f}s")
        return True

    def update_arm_phase(arm_name: str, current_time: float):
        """
        Check if current phase is complete and transition to next phase if needed.
        This ONLY updates inputs - does NOT advance simulator.
        Returns True if phase transitioned, False otherwise.
        """
        global arm_exec_phases, arm_phase_end_times, arm_current_phase_idx, arm_phase_schedules
        global arm_states, arm_plans, obstacle_reservations, arm_diffik_trajectories

        # Check if arm is executing
        if arm_states[arm_name] != ArmState.EXECUTING:
            return False

        # Check if current phase is complete
        if current_time < arm_phase_end_times[arm_name]:
            # Still executing current phase - update diffik if needed
            if arm_exec_phases[arm_name] in [ExecutionPhase.PICKING, ExecutionPhase.PLACING]:
                # Update diffik velocity command
                traj_data = arm_diffik_trajectories[arm_name]
                if traj_data is not None:
                    traj_VG, finger_values, finger_times, phase_start_time = traj_data
                    t_rel = current_time - phase_start_time

                    # Get velocity command from trajectory
                    V_WG = traj_VG.value(t_rel).ravel()

                    # Set velocity command
                    if arm_name == "left":
                        V_G_port = diagram.GetInputPort("V_G_left_cmd")
                        wsg_port = diagram.GetInputPort("wsg_left.position")
                    else:
                        V_G_port = diagram.GetInputPort("V_G_right_cmd")
                        wsg_port = diagram.GetInputPort("wsg_right.position")

                    ctx = simulator.get_mutable_context()
                    V_G_port.FixValue(ctx, V_WG)

                    # Interpolate gripper command
                    finger_cmd = float(np.interp(t_rel, finger_times, finger_values))
                    wsg_port.FixValue(ctx, [finger_cmd])

            return False

        # Phase complete - transition to next phase
        phase_schedule = arm_phase_schedules[arm_name]
        current_idx = arm_current_phase_idx[arm_name]

        if current_idx + 1 >= len(phase_schedule):
            # All phases complete
            print(f"[{arm_name}] All phases complete, returning to IDLE")

            # Clear reservations for this arm
            for t_step in list(obstacle_reservations.keys()):
                if arm_name in obstacle_reservations[t_step]:
                    del obstacle_reservations[t_step][arm_name]
                    if not obstacle_reservations[t_step]:
                        del obstacle_reservations[t_step]

            arm_plans[arm_name] = None
            arm_states[arm_name] = ArmState.IDLE
            arm_exec_phases[arm_name] = ExecutionPhase.IDLE
            arm_phase_schedules[arm_name] = None
            arm_diffik_trajectories[arm_name] = None
            return True

        # Move to next phase
        current_idx += 1
        arm_current_phase_idx[arm_name] = current_idx
        next_phase, next_duration, next_data = phase_schedule[current_idx]

        print(f"[{arm_name}] Transitioning to phase {next_phase.value} (duration={next_duration:.2f}s)")

        # Set up inputs for next phase
        iiwa_name = f"iiwa_{arm_name}"
        wsg_name = f"wsg_{arm_name}"
        opened, closed = 0.107, 0.0

        if next_phase == ExecutionPhase.MOVING_TO_PREPICK:
            # Set joint trajectory
            q_path = next_data["q_path"]
            duration = len(q_path) * 0.6
            ctx = simulator.get_mutable_context()
            t_now = ctx.get_time()
            times = np.linspace(t_now, t_now + duration, len(q_path))
            q_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                times, np.column_stack(q_path), np.zeros(7), np.zeros(7))

            source = iiwa_left_trajectory_source if arm_name == "left" else iiwa_right_trajectory_source
            source.UpdateTrajectory(q_traj)
            diagram.GetInputPort(f"{iiwa_name}.source_select").FixValue(ctx, [0])
            diagram.GetInputPort(f"{wsg_name}.position").FixValue(ctx, [next_data["gripper"]])

        elif next_phase in [ExecutionPhase.PICKING, ExecutionPhase.PLACING]:
            # Set up diffik velocity trajectory
            poses = next_data["poses"]
            times_rel = next_data["times"]
            fingers = next_data["fingers"]
            q0 = next_data["q0"]

            # Create pose trajectory
            traj_pos = PiecewisePose.MakeLinear(times_rel, poses)
            traj_VG = traj_pos.MakeDerivative()

            # Initialize diffik integrator
            set_diffik_velocity_inputs(arm_name, traj_VG, times_rel[0], times_rel[-1], fingers, times_rel, q0)

            # Store trajectory data for continuous updates
            ctx = simulator.get_mutable_context()
            phase_start_time = ctx.get_time()
            arm_diffik_trajectories[arm_name] = (traj_VG, fingers, times_rel, phase_start_time)

        elif next_phase in [ExecutionPhase.TRANSPORTING, ExecutionPhase.RETURNING_HOME]:
            # Set joint trajectory
            set_joint_trajectory_inputs(arm_name, next_data["q_target"], next_duration, next_data["gripper"])
            arm_diffik_trajectories[arm_name] = None  # Clear diffik data

        # Update phase tracking
        arm_exec_phases[arm_name] = next_phase
        ctx = simulator.get_mutable_context()
        arm_phase_end_times[arm_name] = ctx.get_time() + next_duration

        print(f"[{arm_name}] Phase {next_phase.value} started, ends at t={arm_phase_end_times[arm_name]:.2f}s")
        return True

    def run_dual_arm_system(tasks: list[tuple]):
        """
        ========== CENTRALIZED SIMULATION LOOP FOR TRULY CONCURRENT DUAL-ARM EXECUTION ==========

        Main coordinator: assigns tasks to queues and runs arms with truly simultaneous execution.

        Architecture:
        - Single centralized loop continuously advances the simulator
        - Both arms set inputs asynchronously (trajectories, gripper commands)
        - No locks needed - only one thread advances the simulator
        - Arms can execute their motions truly in parallel

        tasks: list of (brick_idx, X_source, X_goal, arm_name)
        """
        global obstacle_reservations
        obstacle_reservations.clear()

        # Populate task queues
        for brick_idx, X_source, X_goal, arm_name in tasks:
            task = ArmTask(X_source=X_source, X_goal=X_goal, brick_idx=brick_idx)
            if arm_name == "left":
                left_task_queue.put(task)
            else:
                right_task_queue.put(task)

        print(f"Task queues: left={left_task_queue.qsize()}, right={right_task_queue.qsize()}")
        print(f"Starting centralized simulation loop with interleaved execution...")

        # Interleaving: delay right arm by half a task duration to stagger execution
        # This prevents both arms from working in the center at the same time
        right_arm_delay = 4.0  # seconds (roughly half of pick-place cycle)
        right_arm_can_start_time = right_arm_delay

        # Centralized simulation loop parameters
        dt = 0.01  # 10ms timesteps (same as integrator max step size)
        publish_interval = 5  # Publish visualization every N steps

        step_count = 0

        # Main centralized loop: continuously advance simulator while updating arm states
        while not left_task_queue.empty() or not right_task_queue.empty() or \
              arm_states["left"] != ArmState.IDLE or arm_states["right"] != ArmState.IDLE:

            # Get current simulation time
            ctx = simulator.get_mutable_context()
            current_time = ctx.get_time()

            # PHASE 1: Process planning for both arms (if idle and have tasks)
            # Interleaving: left arm can always plan, right arm only after delay
            process_arm_planning("left")
            if current_time >= right_arm_can_start_time:
                process_arm_planning("right")

            # PHASE 2: Start execution for arms that just finished planning
            # Both arms can start execution simultaneously!
            for arm_name in ["left", "right"]:
                if arm_states[arm_name] == ArmState.EXECUTING and arm_phase_schedules[arm_name] is None:
                    # Arm just transitioned to EXECUTING, initialize its phase schedule
                    start_arm_execution(arm_name)

                    # ============ VISUALIZATION: RRT path triads ============
                    # Visualize the RRT path by sampling it and showing gripper poses
                    plan = arm_plans[arm_name]
                    rrt_path = plan["rrt_path"]
                    brick_idx = plan['task'].brick_idx

                    # Use a separate plant context for visualization (don't touch simulator context!)
                    plant = station.plant()
                    viz_ctx = plant.CreateDefaultContext()
                    iiwa_name = f"iiwa_{arm_name}"
                    wsg_name = f"wsg_{arm_name}"
                    iiwa_model = plant.GetModelInstanceByName(iiwa_name)
                    wsg_model = plant.GetModelInstanceByName(wsg_name)
                    G_body = plant.GetBodyByName("body", wsg_model)

                    # Sample every 2nd waypoint to avoid clutter
                    for i in range(0, len(rrt_path), 2):
                        q = rrt_path[i]
                        plant.SetPositions(viz_ctx, iiwa_model, q)
                        X_WG = plant.EvalBodyPoseInWorld(viz_ctx, G_body)

                        AddMeshcatTriad(
                            meshcat,
                            f"debug/rrt/{arm_name}/brick{brick_idx}/waypoint_{i}",
                            X_PT=X_WG,
                            length=0.06,
                            radius=0.002,
                            opacity=0.6
                        )

                    print(f"[{arm_name}] Drew {len(rrt_path)//2} RRT triads for brick {brick_idx}")
                    # ============ END VISUALIZATION ============

            # PHASE 3: Update phase states for both arms
            # Both arms update their inputs independently based on current time
            update_arm_phase("left", current_time)
            update_arm_phase("right", current_time)

            # PHASE 4: Advance simulator by one timestep
            # This is the ONLY place where simulator.AdvanceTo() is called!
            # Both arms' inputs have been set above, so they execute simultaneously
            next_time = current_time + dt
            simulator.AdvanceTo(next_time)

            # Periodic visualization publishing for speed
            step_count += 1
            if step_count % publish_interval == 0:
                diagram.ForcedPublish(ctx)

            # Check if we're done (all queues empty and arms idle)
            if left_task_queue.empty() and right_task_queue.empty() and \
               arm_states["left"] == ArmState.IDLE and arm_states["right"] == ArmState.IDLE:
                print(f"All tasks completed at t={current_time:.2f}s!")
                break

        print(f"Centralized simulation loop complete. Total steps: {step_count}")

    # Build base of scenario (robots, grippers, cameras frames/boxes); bricks/table appended below.
    scenario_yaml_base = f"""directives:
    - add_model:
        name: iiwa_left
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
        child: iiwa_left::iiwa_link_0
        X_PC:
            translation: [0.65, 0, 0]

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
            iiwa_joint_1: [-1.57]
            iiwa_joint_2: [0.1]
            iiwa_joint_3: [0]
            iiwa_joint_4: [-1.2]
            iiwa_joint_5: [0]
            iiwa_joint_6: [ 1.6]
            iiwa_joint_7: [0]
    - add_weld:
        parent: world
        child: iiwa_right::iiwa_link_0
        X_PC:
            translation: [-0.65, 0, 0]

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
            rotation: !Rpy {{ deg: [-130.0, 0.0, 180.0]}}
            translation: [0, 2.5, 0.8]
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
            rotation: !Rpy {{ deg: [-130.0, 0.0, 90.0]}}
            translation: [2.5, 0.0, 0.8]
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
            rotation: !Rpy {{ deg: [-130.0, 0.0, -90.0]}}
            translation: [-2.5, 0, 0.8]
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
            rotation: !Rpy {{ deg: [-130.0, 0.0, 0.0]}}
            translation: [0, -2.5, 0.8]
    - add_model:
        name: camera3
        file: package://manipulation/camera_box.sdf
    - add_weld:
        parent: camera3_origin
        child: camera3::base

    # Stack monitoring cameras - positioned at sides of brick stack at [0,0]
    - add_frame:
        name: stack_camera_left_origin
        X_PF:
            base_frame: world
            rotation: !Rpy {{ deg: [0.0, -30.0, 90.0]}}
            translation: [0.0, 0.25, 0.25]
    - add_model:
        name: stack_camera_left
        file: package://manipulation/camera_box.sdf
    - add_weld:
        parent: stack_camera_left_origin
        child: stack_camera_left::base

    - add_frame:
        name: stack_camera_right_origin
        X_PF:
            base_frame: world
            rotation: !Rpy {{ deg: [0.0, -30.0, -90.0]}}
            translation: [0.0, -0.25, 0.25]
    - add_model:
        name: stack_camera_right
        file: package://manipulation/camera_box.sdf
    - add_weld:
        parent: stack_camera_right_origin
        child: stack_camera_right::base

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
    brick_size = [0.16, 0.08, 0.04]   # x, y, z in meters
    num_bricks = 8
    # Spawn bricks farther from the bases but not too far (within table bounds)
    x_band_inner, x_band_outer = 0.6, 1.2
    y_min, y_max = -0.6, 0.6
    min_arm_clearance = 0.5
    min_brick_spacing = 0.25
    arm_bases = np.array([[0.65, 0.0], [-0.65, 0.0]])
    placed_xy = []

    def write_brick_sdf(path: Path, size_xyz):
        sx, sy, sz = size_xyz
        # Compute inertia based on geometry: I = (1/12) * m * (h^2 + d^2) for each axis
        mass = 0.1  # 100 grams (increased from 30g for more stability)
        ixx = (1/12) * mass * (sy**2 + sz**2)
        iyy = (1/12) * mass * (sx**2 + sz**2)
        izz = (1/12) * mass * (sx**2 + sy**2)
        path.write_text(
            f"""<?xml version="1.0"?>
<sdf xmlns:drake="drake.mit.edu" version="1.7">
  <model name="brick_model">
    <link name="brick_link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx:.6f}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{iyy:.6f}</iyy><iyz>0</iyz>
          <izz>{izz:.6f}</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <drake:proximity_properties>
          <drake:compliant_hydroelastic/>
          <drake:hydroelastic_modulus>1.0e8</drake:hydroelastic_modulus>
          <drake:mu_static>1.0</drake:mu_static>
          <drake:mu_dynamic>1.0</drake:mu_dynamic>
        </drake:proximity_properties>
      </collision>
      <visual name="visual">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <material>
          <diffuse>0.5 0.0 0.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
        )


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
        left_base_xy = np.array([0.65, 0.0])
        right_base_xy = np.array([-0.65, 0.0])
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
            # sample x in an outward band, symmetrically about the origin
            sign = 1.0 if np.random.rand() > 0.5 else -1.0
            x = float(sign * np.random.uniform(x_band_inner, x_band_outer))
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
    # IMPORTANT: Pass meshcat=None to disable continuous point cloud visualization (huge speedup!)
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
    
    # Seed iiwa position inputs to current measured posture (simplest "hold" until we command)
    plant = station.plant()

    # Use a default context to read the nominal starting posture
    plant_ctx0 = plant.CreateDefaultContext()
    iiwaL = plant.GetModelInstanceByName("iiwa_left")
    iiwaR = plant.GetModelInstanceByName("iiwa_right")

    qL0 = plant.GetPositions(plant_ctx0, iiwaL)
    qR0 = plant.GetPositions(plant_ctx0, iiwaR)

    # Make constant trajectories that just hold that posture
    # (zero-order hold from t=0 to t=0 with identical endpoints)
    qL_traj0 = PiecewisePolynomial.ZeroOrderHold(
        [0.0, 1.0], np.column_stack([qL0, qL0])
    )
    qR_traj0 = PiecewisePolynomial.ZeroOrderHold(
        [0.0, 1.0], np.column_stack([qR0, qR0])
    )

    iiwa_left_trajectory_source = builder.AddSystem(
        TrajectorySource(qL_traj0)
    )
    iiwa_right_trajectory_source = builder.AddSystem(
        TrajectorySource(qR_traj0)
    )

    # ----- VectorSwitch to choose between TrajectorySource (u0) and DIK (u1) -----
    from pydrake.systems.framework import LeafSystem, BasicVector
    class VectorSwitch(LeafSystem):
        """
        Outputs one of two vector inputs based on an integer mode.
        mode = 0 → output u0
        mode = 1 → output u1
        """
        def __init__(self, size: int):
            super().__init__()
            self.DeclareVectorInputPort("u0", BasicVector(size))
            self.DeclareVectorInputPort("u1", BasicVector(size))
            self._mode = self.DeclareVectorInputPort("mode", BasicVector(1))
            self.DeclareVectorOutputPort("y", BasicVector(size), self.CalcOutput)
        def CalcOutput(self, context, output):
            mode_val = int(round(self._mode.Eval(context)[0]))  # 0 or 1
            if mode_val == 0:
                output.SetFromVector(self.get_input_port(0).Eval(context))
            else:
                # print("using DIK HAHAHAHAHA")
                output.SetFromVector(self.get_input_port(1).Eval(context))

    # Left switch: u0 = traj source, u1 = DIK adder

    controller_left = builder.AddSystem(
        PseudoInverseController(
            plant=plant,
            iiwa_model_name="iiwa_left",
            wsg_model_name="wsg_left",
        )
    )

    builder.ExportInput(
        controller_left.get_input_port(0),   # "V_WG" in controller
        "V_G_left_cmd",
    )

    builder.Connect(station.GetOutputPort("iiwa_left.position_measured"),
                    controller_left.get_input_port(1))
    integrator_left = builder.AddSystem(Integrator(7))

    iiwa_left_switch = builder.AddSystem(VectorSwitch(7))
    builder.Connect(iiwa_left_trajectory_source.get_output_port(),
                    iiwa_left_switch.get_input_port(0))

    builder.Connect(controller_left.get_output_port(),
                    integrator_left.get_input_port())
    builder.Connect(integrator_left.get_output_port(),
                    iiwa_left_switch.get_input_port(1))
    builder.Connect(iiwa_left_switch.get_output_port(),
                    station.GetInputPort("iiwa_left.position"))

    builder.ExportInput(iiwa_left_switch.get_input_port(2), "iiwa_left.source_select")


    # Right switch: u0 = traj source, u1 = DIK adder
    
    controller_right = builder.AddSystem(
        PseudoInverseController(
            plant=plant,
            iiwa_model_name="iiwa_right",
            wsg_model_name="wsg_right",
        )
    )

    builder.ExportInput(
        controller_right.get_input_port(0),   # "V_WG" in controller
        "V_G_right_cmd",
    )

    builder.Connect(station.GetOutputPort("iiwa_right.position_measured"),
                    controller_right.get_input_port(1))
    integrator_right = builder.AddSystem(Integrator(7))

    iiwa_right_switch = builder.AddSystem(VectorSwitch(7))
    builder.Connect(iiwa_right_trajectory_source.get_output_port(),
                    iiwa_right_switch.get_input_port(0))

    builder.Connect(controller_right.get_output_port(),
                    integrator_right.get_input_port())
    builder.Connect(integrator_right.get_output_port(),
                    iiwa_right_switch.get_input_port(1))
    builder.Connect(iiwa_right_switch.get_output_port(),
                    station.GetInputPort("iiwa_right.position"))

    builder.ExportInput(iiwa_right_switch.get_input_port(2), "iiwa_right.source_select")


    # WSG gripper control - export input ports for control
    builder.ExportInput(station.GetInputPort("wsg_left.position"), "wsg_left.position")
    builder.ExportInput(station.GetInputPort("wsg_right.position"), "wsg_right.position")

    diagram = builder.Build()

    dot = diagram.GetGraphvizString(max_depth=2)
    graphs = pydot.graph_from_dot_data(dot)
    graphs[0].write_png("diagram.png")

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(0.0)  # 0 = as fast as possible!

    # Speed up simulation by using larger max timestep
    integrator = simulator.get_mutable_integrator()
    integrator.set_maximum_step_size(0.01)  # 10ms max timestep (default is often 1ms)
    integrator.set_target_accuracy(1e-2)  # Relax accuracy for speed (default is 1e-3)

    # Seed switch modes to use trajectory sources (0) by default.
    ctx0 = simulator.get_mutable_context()
    diagram.GetInputPort("iiwa_left.source_select").FixValue(ctx0, 0)
    diagram.GetInputPort("iiwa_right.source_select").FixValue(ctx0, 0)
    diagram.GetInputPort("V_G_left_cmd").FixValue(ctx0, np.zeros(6))
    diagram.GetInputPort("V_G_right_cmd").FixValue(ctx0, np.zeros(6))
    # Initialize grippers to open position
    diagram.GetInputPort("wsg_left.position").FixValue(ctx0, [0.107])
    diagram.GetInputPort("wsg_right.position").FixValue(ctx0, [0.107])
    simulator.Initialize()

    # ---------------- ICP -> antipodal grasp -> IK place ----------------
    context = simulator.get_mutable_context()

    print("Capturing point clouds for ICP (one-time only)...")
    # Grab point clouds exposed on the diagram (only once, not continuously!)
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

    # Visualize and save the point cloud
    meshcat.SetObject("point_cloud", cropped, point_size=0.003, rgba=Rgba(0.8, 0.8, 0.8, 0.5))

    # Save point cloud to file for later visualization
    np.save("point_cloud.npy", cropped.xyzs().T)
    print(f"Saved point cloud to point_cloud.npy")

    # Downsample for faster ICP
    scene_cloud = cropped.VoxelizedDownSample(0.01)
    print(f"Point cloud captured: {scene_cloud.size()} points")

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
    # meshcat.SetObject("debug/icp/model_cloud", model_cloud, point_size=0.02)

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
    # meshcat.SetTransform("debug/icp/aligned_pose", X_brick_est)
    # meshcat.SetObject("debug/icp/aligned_pc", model_cloud, point_size=0.03)
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
    print(f"ICP found {len(poses)} poses")

    # ICP hack: use ground truth brick poses from plant
    # Get all ground truth brick poses
    plant_ctx = plant.GetMyContextFromRoot(context)
    ground_truth_poses = []
    for i in range(num_bricks):
        try:
            brick_body = plant.GetBodyByName("brick_link", plant.GetModelInstanceByName(f"brick{i}"))
            X_WB = plant.EvalBodyPoseInWorld(plant_ctx, brick_body)
            trans = X_WB.translation()
            # Only bricks on table (z < 0.1m, around brick_z/2 height)
            if np.all(np.isfinite(trans)) and 0.0 < trans[2] < 0.1:
                ground_truth_poses.append(X_WB)
        except Exception:
            continue

    print(f"Ground truth: {len(ground_truth_poses)} bricks on table")

    # Filter out ICP poses that don't match any ground truth brick (bad poses)
    filtered_poses = []
    for icp_pose in poses:
        icp_pos = icp_pose.translation()
        # Check if this ICP pose is close to any ground truth brick
        min_dist = float('inf')
        for gt_pose in ground_truth_poses:
            gt_pos = gt_pose.translation()
            dist = np.linalg.norm(icp_pos - gt_pos)
            min_dist = min(min_dist, dist)

        # Keep ICP pose if it's within 5cm of a ground truth brick
        if min_dist < 0.05:
            filtered_poses.append(icp_pose)

    print(f"After filtering: {len(filtered_poses)} valid ICP poses")

    # If we don't have enough poses, use ground truth for missing bricks
    if len(filtered_poses) < num_bricks:
        print(f"Using ground truth to fill missing poses ({num_bricks - len(filtered_poses)} bricks)")
        # Find which ground truth bricks are not already matched by ICP
        for gt_pose in ground_truth_poses:
            if len(filtered_poses) >= num_bricks:
                break
            gt_pos = gt_pose.translation()
            # Check if this ground truth brick is already covered by an ICP pose
            already_matched = False
            for icp_pose in filtered_poses:
                icp_pos = icp_pose.translation()
                if np.linalg.norm(icp_pos - gt_pos) < 0.05:
                    already_matched = True
                    break

            if not already_matched:
                filtered_poses.append(gt_pose)
                # Visualize ground truth poses we're using
                AddMeshcatTriad(meshcat, f"debug/ground_truth_used/brick_{len(filtered_poses)}",
                               X_PT=gt_pose, length=0.12, radius=0.003)

    poses = filtered_poses
    print(f"Final: {len(poses)} poses for pyramid building")

    # Shift ICP poses down for better gripping (ICP tends to estimate center too high)
    # Apply a downward offset of 1.5cm to all poses for more secure grip
    z_shift_m = -0.03  # 3cm downward (increased from 1cm)
    for i, pose in enumerate(poses):
        trans = pose.translation()
        trans_shifted = trans + np.array([0.0, 0.0, z_shift_m])
        poses[i] = RigidTransform(pose.rotation(), trans_shifted)
    print(f"Applied {-z_shift_m*1000:.0f}mm downward shift to ICP poses for better gripping")

    # Simple 1-stack scenario: stack all bricks vertically at (0, 0)
    brick_x, brick_y, brick_z = brick_size
    table_height = 0.0  # Table surface at z=0

    X_goals = []
    z_offset = table_height + brick_z / 2  # First brick: half-height above table

    # Stack all 8 bricks vertically with gap to prevent sinking
    vertical_gap = 0.005  # 5mm gap between bricks to prevent penetration (increased from 2mm)
    for i in range(8):
        X_goals.append(RigidTransform(RotationMatrix.Identity(), np.array([0.0, 0.0, z_offset])))
        z_offset += brick_z + vertical_gap  # Next brick on top with gap

    print(f"Generated {len(X_goals)} goal positions (vertical stack with {vertical_gap*1000:.0f}mm gaps)")

    # # PYRAMID VERSION (COMMENTED OUT):
    # # Define pyramid goal positions: 4-3-2-1 layers centered at (0, 0)
    # # Brick dimensions from brick_size
    # brick_x, brick_y, brick_z = brick_size
    # table_height = 0.0  # Table surface at z=0
    #
    # # Small clearance to prevent collision accidents (2mm horizontal gap between bricks)
    # horizontal_gap = 0.002  # 2mm gap
    # brick_x_with_gap = brick_x + horizontal_gap
    #
    # X_goals = []
    # z_offset = table_height + brick_z / 2  # First layer: half-height above table
    #
    # # Layer 1 (bottom): 4 bricks in a row along x-axis
    # for i in range(4):
    #     x = (i - 1.5) * brick_x_with_gap  # Centers at -1.5, -0.5, 0.5, 1.5 brick widths (with gap)
    #     y = 0.0
    #     z = z_offset
    #     X_goals.append(RigidTransform(RotationMatrix.Identity(), np.array([x, y, z])))
    #
    # # Layer 2: 3 bricks, offset by half brick width
    # z_offset += brick_z
    # for i in range(3):
    #     x = (i - 1.0) * brick_x_with_gap  # Centers at -1.0, 0.0, 1.0 brick widths (with gap)
    #     y = 0.0
    #     z = z_offset
    #     X_goals.append(RigidTransform(RotationMatrix.Identity(), np.array([x, y, z])))
    #
    # # Layer 3: 2 bricks
    # z_offset += brick_z
    # for i in range(2):
    #     x = (i - 0.5) * brick_x_with_gap  # Centers at -0.5, 0.5 brick widths (with gap)
    #     y = 0.0
    #     z = z_offset
    #     X_goals.append(RigidTransform(RotationMatrix.Identity(), np.array([x, y, z])))
    #
    # # Layer 4 (top): 1 brick
    # z_offset += brick_z
    # X_goals.append(RigidTransform(RotationMatrix.Identity(), np.array([0.0, 0.0, z_offset])))
    #
    # print(f"Generated {len(X_goals)} pyramid goal positions")

    # ============ VISUALIZATION: Brick goal overlays (comment out to disable) ============
    # Show transparent red/blue boxes at goal positions colored by which arm will place them
    brick_box = Box(brick_size[0], brick_size[1], brick_size[2])
    left_base = np.array([0.65, 0.0])
    right_base = np.array([-0.65, 0.0])

    for idx, (X_source, X_goal) in enumerate(zip(poses[:len(X_goals)], X_goals)):
        # Determine which arm will place this brick
        p = X_source.translation()[:2]
        use_left = np.linalg.norm(p - left_base) <= np.linalg.norm(p - right_base)

        # Color code: red for left arm, blue for right arm
        color = Rgba(1.0, 0.0, 0.0, 0.3) if use_left else Rgba(0.0, 0.0, 1.0, 0.3)

        meshcat.SetObject(f"debug/goals/brick_{idx}", brick_box, color)
        meshcat.SetTransform(f"debug/goals/brick_{idx}", X_goal)
    # ============ END VISUALIZATION ============

    # Synchronous dual-arm pick and place
    if len(poses) >= len(X_goals):
        print(f"Placing {len(X_goals)} bricks into pyramid (synchronous dual-arm)...")

        # Group bricks into pairs based on which arm is closer
        left_base = np.array([0.65, 0.0])
        right_base = np.array([-0.65, 0.0])

        tasks = []  # List of (idx, X_source, X_goal, arm_name)
        for idx, (X_source, X_goal) in enumerate(zip(poses[:len(X_goals)], X_goals)):
            p = X_source.translation()[:2]
            use_left = np.linalg.norm(p - left_base) <= np.linalg.norm(p - right_base)
            arm_name = "left" if use_left else "right"
            tasks.append((idx, X_source, X_goal, arm_name))

        print(f"\nTask assignment:")
        for task in tasks:
            print(f"  Brick {task[0]+1}: {task[3]} arm")

        # Use new event loop architecture for concurrent execution
        print(f"\n{'='*60}")
        print(f"CONCURRENT DUAL-ARM EXECUTION")
        print(f"{'='*60}")
        run_dual_arm_system(tasks)
    else:
        print(f"Error: Found {len(poses)} bricks but need {len(X_goals)} for pyramid")

    time.sleep(30)