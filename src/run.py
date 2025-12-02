from pydrake.all import StartMeshcat, DiagramBuilder, Simulator, AddMultibodyPlantSceneGraph, Box, SpatialInertia, UnitInertia, RigidTransform, CoulombFriction, RotationMatrix, MeshcatVisualizer
from manipulation.letter_generation import create_sdf_asset_from_letter
from manipulation.station import LoadScenario, MakeHardwareStation
import time
import numpy as np


if __name__ == "__main__":

    scenario_yaml = f"""directives:
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
    """

    print("Starting MeshCat… (check the VS Code Ports panel)")
    meshcat = StartMeshcat()

    scenario = LoadScenario(data=scenario_yaml)

    # TODO: Create HardwareStation with the scenario and meshcat
    station = MakeHardwareStation(scenario, meshcat)

    builder = DiagramBuilder()
    builder.AddSystem(station)
    # Add a separate plant/scene_graph for bricks (noninvasive to station)
    bricks_plant, bricks_scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    # Add a simple ground so bricks rest at z=0
    ground_thickness = 0.05
    ground_shape = Box(5.0, 5.0, ground_thickness)
    ground_X_WG = RigidTransform([0.0, 0.0, -ground_thickness / 2.0])  # top at z=0
    bricks_plant.RegisterCollisionGeometry(
        bricks_plant.world_body(), ground_X_WG, ground_shape,
        "ground_collision", CoulombFriction(1.0, 1.0)
    )
    bricks_plant.RegisterVisualGeometry(
        bricks_plant.world_body(), ground_X_WG, ground_shape,
        "ground_visual", np.array([0.25, 0.25, 0.28, 1.0])
    )

    # --- Add "brick" bodies (scattered randomly) ---
    brick_size = [0.2, 0.1, 0.06]   # x, y, z in meters
    brick_mass = 1.0                # kg

    brick_inertia = UnitInertia.SolidBox(*brick_size)
    brick_spatial_inertia = SpatialInertia(
        mass=brick_mass,
        p_PScm_E=[0.0, 0.0, 0.0],   # COM at body frame origin
        G_SP_E=brick_inertia
    )

    brick_shape = Box(*brick_size)

    # Create N randomly scattered bricks with proper z above ground and random yaw
    num_bricks = 10
    for i in range(num_bricks):
        name = f"brick{i}"
        body = bricks_plant.AddRigidBody(name, brick_spatial_inertia)
        bricks_plant.RegisterCollisionGeometry(
            body, RigidTransform(), brick_shape, name + "_collision",
            CoulombFriction(0.9, 0.8),
        )
        bricks_plant.RegisterVisualGeometry(
            body, RigidTransform(), brick_shape, name + "_visual",
            np.array([0.8, 0.3, 0.1, 1.0]),
        )
        # Bounds near the robots; tweak as needed
        x = float(np.random.uniform(0.3, 0.9))
        y = float(np.random.uniform(-0.35, 0.35))
        z = brick_size[2] / 2.0 + 0.01
        yaw = float(np.random.uniform(0.0, np.pi))
        X_WB_i = RigidTransform(RotationMatrix.MakeZRotation(yaw), [x, y, z])
        bricks_plant.SetDefaultFreeBodyPose(body, X_WB_i)

    # Finalize bricks plant and visualize via MeshCat
    bricks_plant.Finalize()
    MeshcatVisualizer.AddToBuilder(builder, bricks_scene_graph, meshcat)
    diagram = builder.Build()

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1)
    simulator.Initialize()
    simulator.AdvanceTo(1)

    time.sleep(30)