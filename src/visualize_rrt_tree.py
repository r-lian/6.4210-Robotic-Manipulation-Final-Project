#!/usr/bin/env python3
"""
Visualize saved RRT tree in 3D end-effector space.
Loads a saved RRT tree pickle file and draws the exploration tree in meshcat.
"""

import pickle
import numpy as np
from pydrake.all import (
    DiagramBuilder,
    Meshcat,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    RigidTransform,
    Rgba,
    Sphere,
    PointCloud,
)


def visualize_rrt_tree(tree_filename):
    """Load and visualize an RRT tree in 3D space."""

    # Load the saved tree data
    print(f"Loading tree from {tree_filename}...")
    with open(tree_filename, 'rb') as f:
        tree_data = pickle.load(f)

    tree = tree_data['tree']
    path = tree_data['path']
    iiwa_name = tree_data['iiwa_name']
    q_start = tree_data['q_start']
    q_goal = tree_data['q_goal']

    print(f"Loaded tree with {len(tree)} nodes for {iiwa_name}")
    print(f"Path has {len(path)} waypoints")

    # Set up Drake plant and meshcat
    builder = DiagramBuilder()
    plant, scene_graph = MultibodyPlant.AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    # Load the IIWA model
    parser = Parser(plant)
    iiwa_model = parser.AddModelsFromUrl("package://drake_models/iiwa_description/urdf/iiwa14_spheres_collision.urdf")[0]
    plant.Finalize()

    # Start meshcat
    meshcat = Meshcat()
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    # Get the end-effector body (link 7)
    ee_body = plant.GetBodyByName("iiwa_link_7", iiwa_model)

    # Compute end-effector poses for all nodes
    print("Computing end-effector poses for all nodes...")
    ee_poses = []
    for i, (q, parent_idx, time) in enumerate(tree):
        plant.SetPositions(plant_context, iiwa_model, q)
        ee_transform = plant.EvalBodyPoseInWorld(plant_context, ee_body)
        ee_poses.append(ee_transform.translation())

    # Draw all tree edges
    print("Drawing tree edges...")
    for i, (q, parent_idx, time) in enumerate(tree):
        if parent_idx == -1:
            # Root node - draw as a sphere
            meshcat.SetObject(f"tree/node_{i}", Sphere(0.01), Rgba(0, 1, 0, 1))
            meshcat.SetTransform(f"tree/node_{i}", RigidTransform(ee_poses[i]))
        else:
            # Draw edge from parent to this node
            parent_pos = ee_poses[parent_idx]
            current_pos = ee_poses[i]

            # Create a line by computing midpoint and length
            midpoint = (parent_pos + current_pos) / 2
            direction = current_pos - parent_pos
            length = np.linalg.norm(direction)

            if length > 0:
                # Draw as thin cylinder representing edge
                # Use gray color for tree edges
                edge_path = f"tree/edge_{i}"

                # Simple line visualization using meshcat line geometry
                meshcat.SetLine(edge_path,
                               np.column_stack([parent_pos, current_pos]),
                               line_width=1.0,
                               rgba=Rgba(0.5, 0.5, 0.5, 0.3))

    # Highlight the solution path in a different color
    print("Drawing solution path...")
    path_node_indices = []
    for path_q in path:
        # Find the node index for this configuration
        for i, (q, _, _) in enumerate(tree):
            if np.allclose(q, path_q, atol=1e-6):
                path_node_indices.append(i)
                break

    # Draw path edges in bright color
    for i in range(len(path_node_indices) - 1):
        idx1 = path_node_indices[i]
        idx2 = path_node_indices[i + 1]
        pos1 = ee_poses[idx1]
        pos2 = ee_poses[idx2]

        edge_path = f"path/edge_{i}"
        meshcat.SetLine(edge_path,
                       np.column_stack([pos1, pos2]),
                       line_width=3.0,
                       rgba=Rgba(1, 0, 0, 1))

    # Draw start and goal markers
    start_idx = 0
    goal_idx = path_node_indices[-1] if path_node_indices else -1

    meshcat.SetObject("markers/start", Sphere(0.02), Rgba(0, 1, 0, 1))
    meshcat.SetTransform("markers/start", RigidTransform(ee_poses[start_idx]))

    if goal_idx >= 0:
        meshcat.SetObject("markers/goal", Sphere(0.02), Rgba(1, 0, 0, 1))
        meshcat.SetTransform("markers/goal", RigidTransform(ee_poses[goal_idx]))

    print(f"\nVisualization ready!")
    print(f"- Gray edges: RRT exploration tree ({len(tree)} nodes)")
    print(f"- Red edges: Solution path ({len(path)} waypoints)")
    print(f"- Green sphere: Start configuration")
    print(f"- Red sphere: Goal configuration")
    print(f"\nView at: {meshcat.web_url()}")
    print("Press Ctrl+C to exit...")

    # Keep the visualization open
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nClosing visualization.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualize_rrt_tree.py <tree_filename.pkl>")
        print("\nExample:")
        print("  python visualize_rrt_tree.py rrt_tree_iiwa_left.pkl")
        sys.exit(1)

    tree_filename = sys.argv[1]
    visualize_rrt_tree(tree_filename)