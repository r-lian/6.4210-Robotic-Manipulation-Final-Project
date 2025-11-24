"""Rollout of a mocap trajectory in simulation of a controlled humanoid robot
lifting a non-actuated mannequin from a chair.
The humanoid and mannequin are welded to the floor at one of their feet."""
import argparse
from collections import OrderedDict
from os import path

import numpy as np

from pydrake.geometry import (
    CollisionFilterDeclaration,
    DrakeVisualizer,
    GeometrySet,
    MeshcatVisualizer,
    StartMeshcat,
)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.meshcat import (
    ContactVisualizer,
    ContactVisualizerParams,
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    ConnectContactResultsToDrakeVisualizer,
    ContactModel,
    DiscreteContactApproximation,
    MultibodyPlant,
)
from pydrake.multibody.tree import RevoluteJoint
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import (
    InverseDynamicsController,
    PidController,
)
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource, MatrixGain

"""from anzu.common.runfiles import find_anzu_resource_or_throw
from anzu.tactile.mocap.helpers.common import (
    TrajectorySource,
    TrajectorySourceSystem,
    add_welded_robot,
    base_link_pose,
    get_frames_welded_robot,
)
from anzu.tactile.mocap.helpers.kinematics import (
    build_skel,
    forward_kinematics,
)
from anzu.tactile.mocap.helpers.mocap_player_helpers import load_bvh_file"""


def make_simulator(args, meshcat):
    """Makes a simulator."""
    time_step = 0.001

    bvh_file_path = args.mocap_trajectory_file
    skel = build_skel(bvh_file_path)
    # Loads the bvh file and skips the first 20 frames.
    frames = load_bvh_file(bvh_file_path, skip=20)
    # Calculates the locations and orientations of all joints frames.
    forward_kinematics(skel, frames)

    cr_bvh_file_path = path.join(
        # "anzu",
        "tactile",
        "mocap",
        "example_motions",
        "mannequin.bvh",
    )

    cr_skel = build_skel(cr_bvh_file_path)
    # Loads the bvh file and skips the first 20 frames.
    cr_frames = load_bvh_file(cr_bvh_file_path, skip=20)
    forward_kinematics(cr_skel, cr_frames)

    builder = DiagramBuilder()

    # Consturct a multibody plant that contains care giver, care receiver,
    # and a chair.
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=time_step
    )

    chair_sdf_file_path = path.join(
        "tactile",
        "mocap",
        "models",
        "chair.sdf",
    )

    floor_sdf_file_path = path.join(
        "tactile",
        "mocap",
        "models",
        "floor.sdf",
    )

    parser = Parser(plant)
    parser.AddModels(find_anzu_resource_or_throw(chair_sdf_file_path))
    parser.AddModels(find_anzu_resource_or_throw(floor_sdf_file_path))

    # Adds a care giver.
    cg_sdf_file_path = path.join(
        "tactile",
        "mocap",
        "models",
        "eric_welded.sdf",
    )

    _, cg_base_link_quat = base_link_pose(skel)
    model_idx = add_welded_robot(
        plant,
        cg_sdf_file_path,
        skel,
    )

    # Add a care receiver.
    cr_sdf_file_path = path.join(
        "tactile",
        "mocap",
        "models",
        "mannequin_welded.sdf",
    )

    _, cr_base_link_quat = base_link_pose(cr_skel)
    cr_model_idx = add_welded_robot(
        plant,
        cr_sdf_file_path,
        cr_skel,
    )

    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)

    if args.contact_model == "hydroelastic_with_fallback":
        plant.set_contact_model(ContactModel.kHydroelasticWithFallback)
    elif args.contact_model == "point":
        plant.set_contact_model(ContactModel.kPoint)
    else:
        raise ValueError("Not supported contact model.")

    # Filtering collisions between two consecutive links.
    filter_manager = scene_graph.collision_filter_manager()

    for m_skel, m_idx in zip([skel, cr_skel], [model_idx, cr_model_idx]):
        for joint in m_skel:
            if joint.name[-4:] == "_end" or joint.parent is None:
                continue

            parent = plant.GetBodyByName(f"{joint.parent.name}_link", m_idx)
            child = plant.GetBodyByName(f"{joint.name}_link", m_idx)

            geo_set = GeometrySet(
                plant.GetCollisionGeometriesForBody(parent)
                + plant.GetCollisionGeometriesForBody(child)
            )
            filter_manager.Apply(
                declaration=CollisionFilterDeclaration().ExcludeWithin(geo_set)
            )

    plant.Finalize()

    if meshcat:
        # Add a MeshcatVisualizer.
        # Might be useful to make a gif file.
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

        ContactVisualizer.AddToBuilder(
            builder,
            plant,
            meshcat,
            ContactVisualizerParams(
                radius=0.005,
                newtons_per_meter=5000.0,
            ),
        )
    else:
        # Add a DrakeVisualizer.
        DrakeVisualizer.AddToBuilder(builder, scene_graph)

        ConnectContactResultsToDrakeVisualizer(
            builder,
            plant,
            scene_graph,
        )

    # zero torque for the care receiver.
    num_actuators = plant.num_actuators(cr_model_idx)
    zero_torque = ConstantVectorSource(np.zeros(num_actuators))
    zero_torque = builder.AddSystem(zero_torque)
    builder.Connect(
        zero_torque.get_output_port(),
        plant.get_actuation_input_port(cr_model_idx),
    )

    # Construct a target source from the care giver bvh file.
    # Packs all joint angles of joints found in the multibody plant.
    trajectory_source = TrajectorySource(
        skel,
        plant,
        model_idx,
        frames,
        cg_base_link_quat,
    )
    trajectory_source = builder.AddSystem(
        TrajectorySourceSystem(trajectory_source)
    )

    if args.controller == "IDC":
        controller_plant = MultibodyPlant(time_step)

        add_welded_robot(
            controller_plant,
            cg_sdf_file_path,
            skel,
        )
        controller_plant.Finalize()

        num_actuators = controller_plant.num_actuators()
        kps = 100 * np.ones((num_actuators, 1))
        kis = np.zeros_like(kps)
        kds = 10 * np.ones_like(kps)

        controller = InverseDynamicsController(
            controller_plant, kps, kis, kds, False
        )
        controller = builder.AddSystem(controller)

        # gf_to_act converts a generalized force to an actuation.
        mat = np.zeros(
            (plant.num_actuators(model_idx), plant.num_velocities(model_idx))
        )

        for idx in plant.GetJointActuatorIndices(model_idx):
            actuator = plant.get_joint_actuator(idx)
            joint = actuator.joint()

            mat[actuator.input_start(), joint.velocity_start()] = 1.0

        gf_to_act = builder.AddSystem(MatrixGain(mat))

        builder.Connect(
            plant.get_state_output_port(model_idx),
            controller.get_input_port_estimated_state(),
        )
        builder.Connect(
            trajectory_source.get_output_port(),
            controller.get_input_port_desired_state(),
        )
        builder.Connect(
            controller.get_output_port(),
            gf_to_act.get_input_port(),
        )
        builder.Connect(
            gf_to_act.get_output_port(),
            plant.get_actuation_input_port(model_idx),
        )
    elif args.controller == "PID":
        # constructs a PidController.
        pd_coefficients = OrderedDict(
            Spine=(2000, 200),
            Spine1=(2000, 200),
            Neck=(100, 10),
            Head=(100, 10),
            LeftShoulder=(1000, 100),
            LeftArm=(1000, 100),
            LeftForeArm=(1000, 100),
            LeftHand=(200, 20),
            LeftUpLeg=(1000, 100),
            LeftLeg=(1000, 100),
            LeftFoot=(200, 20),
            LeftToeBase=(100, 10),
            RightShoulder=(1000, 100),
            RightArm=(1000, 100),
            RightForeArm=(1000, 100),
            RightHand=(200, 20),
            RightUpLeg=(2000, 200),
            RightLeg=(2000, 200),
            RightToeBase=(100, 10),
            base_link_RightLeg=(2000, 200),
            base_link_RightFoot=(200, 20),
        )

        # Mapping from plant indices to model indices.

        # Plant `starts` of model joints and actuators.
        pos_starts = []
        vel_starts = []
        input_starts = []

        for idx in plant.GetJointActuatorIndices(model_idx):
            actuator = plant.get_joint_actuator(idx)
            joint = actuator.joint()

            if not isinstance(joint, RevoluteJoint):
                continue

            pos_starts.append(joint.position_start())
            vel_starts.append(joint.velocity_start())
            input_starts.append(actuator.input_start())

        # Plant position indices to model position indices.
        pos_map = {}
        for m, p in enumerate(sorted(pos_starts)):
            pos_map[p] = m

        # Plant velocity indices to model velocity indices.
        vel_map = {}
        for m, p in enumerate(sorted(vel_starts)):
            vel_map[p] = m

        # Plant actuator indices to model actuator indices.
        act_map = {}
        for m, p in enumerate(sorted(input_starts)):
            act_map[p] = m

        # PidController assumes that num_positions() == num_velocities().
        kp, ki, kd = np.zeros((3, plant.num_positions(model_idx)))

        for name, (p, d) in pd_coefficients.items():
            for suffix in "rpy":
                joint = plant.GetJointByName(f"{name}_{suffix}", model_idx)
                pos = joint.position_start()
                kp[pos_map[pos]] = p
                kd[pos_map[pos]] = d

        # Projection to the actuation.
        output_projection = np.zeros(
            (
                plant.num_actuators(model_idx),
                plant.num_velocities(model_idx),
            )
        )

        for idx in plant.GetJointActuatorIndices(model_idx):
            actuator = plant.get_joint_actuator(idx)
            joint = actuator.joint()

            act_idx = act_map[actuator.input_start()]
            vel_idx = vel_map[joint.velocity_start()]
            output_projection[act_idx, vel_idx] = 1

        num_states = plant.num_multibody_states(model_idx)

        controller = builder.AddSystem(
            PidController(
                state_projection=np.eye(num_states),
                kp=kp,
                ki=ki,
                kd=kd,
                output_projection=output_projection,
            )
        )

        builder.Connect(
            trajectory_source.get_output_port(),
            controller.get_input_port_desired_state(),
        )
        builder.Connect(
            plant.get_state_output_port(model_idx),
            controller.get_input_port_estimated_state(),
        )
        builder.Connect(
            controller.get_output_port(),
            plant.get_actuation_input_port(model_idx),
        )

    # Constructs a simulator from a diagram.
    diagram = builder.Build()
    simulator = Simulator(diagram)

    simulator.set_target_realtime_rate(1)

    # Initialize the pose of the care giver.
    context = simulator.get_context()
    plant_context = plant.GetMyContextFromRoot(context)
    target_state_context = trajectory_source.GetMyContextFromRoot(context)
    state = trajectory_source.get_output_port().Eval(target_state_context)
    plant.SetPositionsAndVelocities(plant_context, model_idx, state)

    # Initialize the position of the care receiver.
    cr_frames = get_frames_welded_robot(cr_skel, cr_frames, cr_base_link_quat)

    for name, quat in cr_frames.items():
        rpy = -RollPitchYaw(quat[0].conjugate()).vector()

        for angle, suffix in zip(rpy, "rpy"):
            joint = plant.GetJointByName(f"{name}_{suffix}", cr_model_idx)
            joint.set_angle(plant_context, angle)

    # The mannequin does not have joints corresponding to the shoulder joints
    # of OptiTrack model. Thus these joints are locked.
    # Note that the OptiTrack shoulder joint refers to the joint that links
    # the Spine1 link and the Shoulder link.
    # Don't be confused with the joint that link the Shoulder link and the
    # ArmLink, which is the Arm joint.
    for joint_name in ["LeftShoulder", "RightShoulder"]:
        for axis in "rpy":
            joint = plant.GetJointByName(f"{joint_name}_{axis}", cr_model_idx)
            angle = joint.get_angle(plant_context)
            joint.set_position_limits([[angle]], [[angle]])

    # Limits the DoFs of these joints and makes them essentially
    # revolute joints.
    lock_axes = [
        ("Leg", "py"),
        ("ForeArm", "ry"),
        ("ToeBase", "py"),
    ]
    for joint_name, axes in lock_axes:
        for prefix in ["Left", "Right"]:
            for axis in axes:
                name = f"{prefix}{joint_name}_{axis}"
                joint = plant.GetJointByName(name, cr_model_idx)
                angle = joint.get_angle(plant_context)
                joint.set_position_limits([[angle]], [[angle]])

    simulator.Initialize()

    return simulator


def main():
    """Entry point"""
    default_mocap_trajectory_file_path = path.join(
        "anzu",
        "tactile",
        "mocap",
        "example_motions",
        "eric.bvh",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mocap-trajectory-file",
        type=str,
        default=default_mocap_trajectory_file_path,
        help="A bvh mocap trajectory file for the care giver.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="IDC",
        help="the controller type ['IDC', 'PID']",
    )
    parser.add_argument(
        "--contact-model",
        type=str,
        default="hydroelastic_with_fallback",
        help="MultibodyPlant contact model. It should be either "
        "hydroelastic_with_fallback or point.",
    )
    parser.add_argument(
        "--meshcat",
        action="store_true",
        help="If specified, MeshcatVisualizer is used.",
    )
    args = parser.parse_args()

    meshcat = None

    if args.meshcat:
        meshcat = StartMeshcat()

    simulator = make_simulator(args, meshcat)

    if meshcat:
        meshcat.StartRecording()
    simulator.AdvanceTo(8)
    if meshcat:
        meshcat.StopRecording()
        meshcat.PublishRecording()


if __name__ == "__main__":
    main()