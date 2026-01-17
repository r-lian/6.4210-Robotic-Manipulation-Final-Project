"""
Generate a multi-panel figure showing the RANSAC point cloud processing pipeline.
Uses plotly for 3D visualization to avoid matplotlib conflicts.

Panels:
(A) Raw RGB-D view from one camera
(B) Fused point cloud with table still present
(C) RANSAC plane segmentation (table vs outliers)
(D) Cropped brick region after plane removal and bounding box crop
"""

import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pydrake.all import (
    DiagramBuilder,
    MeshcatVisualizer,
    Simulator,
    StartMeshcat,
)

from manipulation.scenarios import AddRgbdSensors
from manipulation.station import LoadScenario, MakeHardwareStation


def write_brick_sdf(path: Path, size_xyz, brick_id: int):
    """Write a simple brick SDF file."""
    sx, sy, sz = size_xyz
    mass = 0.1  # 100 grams
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
        <geometry>
          <box><size>{sx} {sy} {sz}</size></box>
        </geometry>
        <drake:proximity_properties>
          <drake:compliant_hydroelastic/>
          <drake:hydroelastic_modulus>5.0e7</drake:hydroelastic_modulus>
          <drake:mu_dynamic>0.6</drake:mu_dynamic>
          <drake:mu_static>0.7</drake:mu_static>
        </drake:proximity_properties>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>{sx} {sy} {sz}</size></box>
        </geometry>
        <material>
          <diffuse>[0.8 0.2 0.1 1.0]</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
    )


def write_table_sdf(path: Path):
    """Write a simple table SDF file."""
    path.write_text(
        """<?xml version="1.0"?>
<sdf xmlns:drake="drake.mit.edu" version="1.7">
  <model name="table">
    <link name="table_link">
      <inertial>
        <mass>50.0</mass>
        <inertia>
          <ixx>10.0</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>10.0</iyy><iyz>0</iyz>
          <izz>10.0</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>4.0 4.0 0.1</size></box>
        </geometry>
        <drake:proximity_properties>
          <drake:rigid_hydroelastic/>
          <drake:mu_dynamic>0.5</drake:mu_dynamic>
          <drake:mu_static>0.6</drake:mu_static>
        </drake:proximity_properties>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>4.0 4.0 0.1</size></box>
        </geometry>
        <material>
          <diffuse>[0.9 0.9 0.8 1.0]</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
    )


def ransac_plane_fit(points, n_iterations=200, distance_threshold=0.005):
    """
    Fit a plane to points using RANSAC.

    Args:
        points: Nx3 array of points
        n_iterations: number of RANSAC iterations
        distance_threshold: max distance for a point to be considered an inlier

    Returns:
        inliers: boolean mask of inliers
        plane_params: (a, b, c, d) where ax + by + cz + d = 0
    """
    N = points.shape[0]
    if N < 3:
        return np.zeros(N, dtype=bool), None

    best_inliers = np.zeros(N, dtype=bool)
    best_count = 0
    best_plane = None

    for _ in range(n_iterations):
        # Sample 3 random points
        idx = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = points[idx]

        # Compute plane normal
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

        # Compute distance from all points to plane
        d = np.abs((points - p1) @ n)
        inliers = d < distance_threshold
        count = int(np.sum(inliers))

        if count > best_count:
            best_count = count
            best_inliers = inliers
            # Store plane as (a, b, c, d) where ax + by + cz + d = 0
            best_plane = (n[0], n[1], n[2], -n @ p1)

    return best_inliers, best_plane


def setup_scene():
    """Set up Drake scene with table, bricks, and cameras."""
    meshcat = StartMeshcat()
    builder = DiagramBuilder()

    # Create assets directory
    assets_dir = Path("/workspaces/bricklaying/assets")
    assets_dir.mkdir(exist_ok=True)
    bricks_dir = assets_dir / "bricks"
    bricks_dir.mkdir(exist_ok=True)

    # Write table SDF
    table_sdf_path = assets_dir / "table.sdf"
    write_table_sdf(table_sdf_path)

    # Build scenario YAML with cameras and table
    scenario_yaml_base = """
directives:
    # Camera frames and simple visuals
    - add_frame:
        name: camera0_origin
        X_PF:
            base_frame: world
            rotation: !Rpy { deg: [-130.0, 0.0, 180.0]}
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
            rotation: !Rpy { deg: [-130.0, 0.0, 90.0]}
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
            rotation: !Rpy { deg: [-130.0, 0.0, -90.0]}
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
            rotation: !Rpy { deg: [-130.0, 0.0, 0.0]}
            translation: [0, -2.5, 0.8]
    - add_model:
        name: camera3
        file: package://manipulation/camera_box.sdf
    - add_weld:
        parent: camera3_origin
        child: camera3::base

    # Table
    - add_model:
        name: table
        file: file://""" + str(table_sdf_path.resolve()) + """
    - add_weld:
        parent: world
        child: table::table_link
        X_PC:
            translation: [0.0, 0.0, -0.05]
"""

    # Add bricks with some in a stack
    brick_size = [0.16, 0.08, 0.04]  # x, y, z in meters

    bricks_directives = ""
    brick_positions = [
        # Stack of 3 bricks
        ([0.0, 0.0, 0.02], [0, 0, 0]),  # bottom
        ([0.0, 0.0, 0.06], [0, 0, 90]),  # middle (rotated)
        ([0.0, 0.0, 0.10], [0, 0, 0]),  # top
        # Scattered bricks
        ([0.3, 0.2, 0.02], [0, 0, 30]),
        ([-0.25, 0.15, 0.02], [0, 0, -45]),
        ([0.2, -0.3, 0.02], [0, 0, 60]),
        ([-0.35, -0.2, 0.02], [0, 0, 15]),
        ([0.15, 0.35, 0.02], [0, 0, -30]),
    ]

    for i, (pos, rpy) in enumerate(brick_positions):
        brick_sdf_path = bricks_dir / f"brick_{i}.sdf"
        write_brick_sdf(brick_sdf_path, brick_size, i)

        bricks_directives += f"""
    - add_model:
        name: brick{i}
        file: file://{brick_sdf_path.resolve()}
    - add_weld:
        parent: world
        child: brick{i}::brick_link
        X_PC:
            translation: [{pos[0]}, {pos[1]}, {pos[2]}]
            rotation: !Rpy {{ deg: [{rpy[0]}, {rpy[1]}, {rpy[2]}]}}
"""

    # Add cameras to scenario
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
"""

    scenario_yaml = scenario_yaml_base + bricks_directives + cameras_block

    # Load scenario and create hardware station
    scenario = LoadScenario(data=scenario_yaml)
    station = MakeHardwareStation(scenario, meshcat)
    builder.AddSystem(station)

    # Add point cloud outputs for cameras
    to_point_cloud = AddRgbdSensors(builder, station, scenario)

    for i in range(4):
        builder.ExportOutput(
            to_point_cloud[f"camera{i}"].get_output_port(),
            f"camera_point_cloud{i}"
        )

    # Build diagram
    diagram = builder.Build()

    return diagram, meshcat


def fuse_point_clouds(diagram, context):
    """Fuse point clouds from all 4 cameras."""
    all_points = []
    all_colors = []

    for i in range(4):
        pc = diagram.GetOutputPort(f"camera_point_cloud{i}").Eval(context)
        xyz = pc.xyzs()  # 3xN
        if xyz.shape[1] > 0:
            all_points.append(xyz.T)  # Nx3

            # Get colors if available
            if pc.has_rgbs():
                rgb = pc.rgbs()  # 3xN, values in [0, 255]
                all_colors.append(rgb.T / 255.0)  # Nx3, normalized to [0, 1]
            else:
                all_colors.append(np.ones((xyz.shape[1], 3)) * 0.5)  # gray

    if len(all_points) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    fused_points = np.vstack(all_points)
    fused_colors = np.vstack(all_colors)

    return fused_points, fused_colors


def crop_to_brick_workspace(points, colors, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5), z_min=0.0):
    """Crop point cloud to brick workspace bounding box."""
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_min)
    )
    return points[mask], colors[mask]


def main():
    print("Setting up Drake scene...")
    diagram, meshcat = setup_scene()

    # Create simulator and initialize
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    simulator.AdvanceTo(0.1)

    print("Meshcat URL:", meshcat.web_url())
    print("Open this URL to see the scene visualization")

    # Fuse point clouds from all cameras
    print("\nFusing point clouds from all cameras...")
    fused_points, fused_colors = fuse_point_clouds(diagram, context)
    print(f"Fused point cloud has {fused_points.shape[0]} points")

    # RANSAC plane segmentation
    print("Running RANSAC plane segmentation...")
    inliers, plane_params = ransac_plane_fit(fused_points)
    print(f"Found {np.sum(inliers)} table inliers out of {len(inliers)} points")

    # Remove table and crop to brick workspace
    print("Removing table and cropping to brick workspace...")
    outliers = ~inliers
    outlier_points = fused_points[outliers]
    outlier_colors = fused_colors[outliers]

    cropped_points, cropped_colors = crop_to_brick_workspace(
        outlier_points, outlier_colors,
        x_range=(-0.5, 0.5), y_range=(-0.5, 0.5), z_min=0.0
    )
    print(f"Cropped point cloud has {cropped_points.shape[0]} points")

    # Subsample for visualization
    def subsample(pts, cols, max_pts=5000):
        if pts.shape[0] > max_pts:
            idx = np.random.choice(pts.shape[0], max_pts, replace=False)
            return pts[idx], cols[idx]
        return pts, cols

    # Create plotly figure with subplots
    print("\nGenerating figure...")
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}],
               [{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=("(A) Raw RGB-D View (Camera 0)", "(B) Fused Point Cloud with Table",
                       "(C) RANSAC Plane Segmentation", "(D) Cropped Brick Workspace"),
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    # Panel A: Single camera view
    pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
    xyz0 = pc0.xyzs().T
    if pc0.has_rgbs():
        rgb0 = pc0.rgbs().T / 255.0
    else:
        rgb0 = np.ones((xyz0.shape[0], 3)) * 0.5

    xyz0_sub, rgb0_sub = subsample(xyz0, rgb0)
    fig.add_trace(
        go.Scatter3d(
            x=xyz0_sub[:, 0], y=xyz0_sub[:, 1], z=xyz0_sub[:, 2],
            mode='markers',
            marker=dict(size=2, color=[f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                                       for r, g, b in rgb0_sub]),
            showlegend=False
        ),
        row=1, col=1
    )

    # Panel B: Fused point cloud (gray)
    fused_sub, _ = subsample(fused_points, fused_colors)
    fig.add_trace(
        go.Scatter3d(
            x=fused_sub[:, 0], y=fused_sub[:, 1], z=fused_sub[:, 2],
            mode='markers',
            marker=dict(size=2, color='rgb(150,150,150)'),
            showlegend=False
        ),
        row=1, col=2
    )

    # Panel C: RANSAC (green=table, purple=outliers)
    inlier_pts = fused_points[inliers]
    outlier_pts = fused_points[~inliers]

    inlier_sub, _ = subsample(inlier_pts, np.zeros((inlier_pts.shape[0], 3)))
    outlier_sub, _ = subsample(outlier_pts, np.zeros((outlier_pts.shape[0], 3)))

    fig.add_trace(
        go.Scatter3d(
            x=inlier_sub[:, 0], y=inlier_sub[:, 1], z=inlier_sub[:, 2],
            mode='markers',
            marker=dict(size=2, color='rgb(50,200,50)'),
            name='Table',
            showlegend=True
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter3d(
            x=outlier_sub[:, 0], y=outlier_sub[:, 1], z=outlier_sub[:, 2],
            mode='markers',
            marker=dict(size=2, color='rgb(200,75,125)'),
            name='Bricks',
            showlegend=True
        ),
        row=2, col=1
    )

    # Panel D: Cropped region
    cropped_sub, cropped_col_sub = subsample(cropped_points, cropped_colors)
    fig.add_trace(
        go.Scatter3d(
            x=cropped_sub[:, 0], y=cropped_sub[:, 1], z=cropped_sub[:, 2],
            mode='markers',
            marker=dict(size=2, color=[f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                                       for r, g, b in cropped_col_sub]),
            showlegend=False
        ),
        row=2, col=2
    )

    # Add bounding box to panel D
    x_range, y_range = (-0.5, 0.5), (-0.5, 0.5)
    z_min, z_max = 0.0, 0.2

    # Define box edges
    box_x = [x_range[0], x_range[1], x_range[1], x_range[0], x_range[0], x_range[0], x_range[1], x_range[1], x_range[0], x_range[0], x_range[0], None,
             x_range[1], x_range[1], None, x_range[1], x_range[1], None, x_range[0], x_range[0]]
    box_y = [y_range[0], y_range[0], y_range[1], y_range[1], y_range[0], y_range[0], y_range[0], y_range[0], y_range[0], y_range[1], y_range[1], None,
             y_range[0], y_range[0], None, y_range[1], y_range[1], None, y_range[1], y_range[1]]
    box_z = [z_min, z_min, z_min, z_min, z_min, z_max, z_max, z_min, z_min, z_min, z_max, None,
             z_min, z_max, None, z_min, z_max, None, z_min, z_max]

    fig.add_trace(
        go.Scatter3d(
            x=box_x, y=box_y, z=box_z,
            mode='lines',
            line=dict(color='cyan', width=4),
            name='Workspace',
            showlegend=True
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        height=1200,
        width=1600,
        title_text="RANSAC Point Cloud Processing Pipeline",
        showlegend=True,
        scene=dict(aspectmode='data'),
        scene2=dict(aspectmode='data'),
        scene3=dict(aspectmode='data'),
        scene4=dict(aspectmode='data'),
    )

    # Save as HTML (interactive)
    output_html = Path("/workspaces/bricklaying/ransac_pipeline_figure.html")
    fig.write_html(str(output_html))
    print(f"\nInteractive figure saved to: {output_html}")

    # Save as static image
    try:
        output_png = Path("/workspaces/bricklaying/ransac_pipeline_figure.png")
        fig.write_image(str(output_png), width=1600, height=1200)
        print(f"Static figure saved to: {output_png}")
    except Exception as e:
        print(f"Could not save static image: {e}")
        print("You can still view the interactive HTML file")

    print("\nDone! Keep the Meshcat window open to see the 3D scene.")
    print("Meshcat URL:", meshcat.web_url())
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()