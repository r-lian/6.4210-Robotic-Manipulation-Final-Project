"""
Generate a multi-panel figure showing the RANSAC point cloud processing pipeline.

Panels:
(A) Raw RGB-D view from one camera
(B) Fused point cloud with table still present
(C) RANSAC plane segmentation (table vs outliers)
(D) Cropped brick region after plane removal and bounding box crop
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
from pathlib import Path

# Import 3D plotting after setting backend
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pydrake.all import (
    AddDefaultVisualization,
    DiagramBuilder,
    Meshcat,
    MeshcatVisualizer,
    Parser,
    ProcessModelDirectives,
    RigidTransform,
    RollPitchYaw,
    Simulator,
    StartMeshcat,
    LoadModelDirectives,
)

from manipulation.scenarios import AddRgbdSensors
from manipulation.station import load_scenario


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
            # n @ (p - p1) = 0  =>  n @ p = n @ p1  =>  n @ p - n @ p1 = 0
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
    num_bricks = 8

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

    # Load scenario
    scenario = load_scenario(data=scenario_yaml)
    station = builder.AddSystem(scenario)

    # Add visualization
    visualizer = MeshcatVisualizer.AddToBuilder(builder, station.GetOutputPort("query_object"), meshcat)

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


def plot_point_cloud_3d(ax, points, colors, title, elev=30, azim=45):
    """Plot a 3D point cloud."""
    if points.shape[0] == 0:
        ax.text(0.5, 0.5, 0.5, "No points", ha='center', va='center')
        return

    # Subsample for visualization if too many points
    if points.shape[0] > 10000:
        idx = np.random.choice(points.shape[0], 10000, replace=False)
        points = points[idx]
        colors = colors[idx]

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors, s=1, alpha=0.6)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)

    # Set equal aspect ratio
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def main():
    print("Setting up Drake scene...")
    diagram, meshcat = setup_scene()

    # Create simulator and initialize
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    simulator.AdvanceTo(0.1)

    print("Meshcat URL:", meshcat.web_url())
    print("Open this URL to see the scene visualization")

    # Get RGB image from camera 0 for panel A
    print("\nCapturing RGB image from camera0...")
    station = diagram.GetSubsystemByName("station")
    station_context = station.GetMyContextFromRoot(context)

    # Try to get RGB image
    try:
        rgb_image = station.GetOutputPort("camera0.rgb_image").Eval(station_context)
        rgb_data = rgb_image.data
        # Reshape based on image dimensions
        height, width = rgb_image.height(), rgb_image.width()
        rgb_array = rgb_data.reshape((height, width, 3))
    except:
        print("Could not capture RGB image, will skip panel A")
        rgb_array = None

    # Fuse point clouds from all cameras
    print("Fusing point clouds from all cameras...")
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

    # Create multi-panel figure
    print("\nGenerating figure...")
    fig = plt.figure(figsize=(16, 12))

    # Panel A: Raw RGB-D view (if available)
    if rgb_array is not None:
        ax_a = fig.add_subplot(2, 2, 1)
        ax_a.imshow(rgb_array)
        ax_a.set_title('(A) Raw RGB-D View', fontsize=12, fontweight='bold')
        ax_a.axis('off')
    else:
        ax_a = fig.add_subplot(2, 2, 1, projection='3d')
        # Show a single camera view instead
        pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
        xyz0 = pc0.xyzs().T
        if pc0.has_rgbs():
            rgb0 = pc0.rgbs().T / 255.0
        else:
            rgb0 = np.ones((xyz0.shape[0], 3)) * 0.5
        plot_point_cloud_3d(ax_a, xyz0, rgb0, '(A) Raw RGB-D View (Camera 0)', elev=30, azim=135)

    # Panel B: Fused point cloud (all gray to show structure)
    ax_b = fig.add_subplot(2, 2, 2, projection='3d')
    gray_colors = np.ones((fused_points.shape[0], 3)) * 0.6
    plot_point_cloud_3d(ax_b, fused_points, gray_colors,
                        '(B) Fused Point Cloud with Table', elev=30, azim=45)

    # Panel C: RANSAC segmentation (table=green, outliers=red/purple)
    ax_c = fig.add_subplot(2, 2, 3, projection='3d')
    ransac_colors = np.zeros((fused_points.shape[0], 3))
    ransac_colors[inliers] = [0.2, 0.8, 0.2]  # green for table
    ransac_colors[~inliers] = [0.8, 0.3, 0.5]  # purple/red for bricks
    plot_point_cloud_3d(ax_c, fused_points, ransac_colors,
                        '(C) RANSAC Plane Segmentation', elev=30, azim=45)

    # Panel D: Cropped brick region
    ax_d = fig.add_subplot(2, 2, 4, projection='3d')
    plot_point_cloud_3d(ax_d, cropped_points, cropped_colors,
                        '(D) Cropped Brick Workspace', elev=30, azim=45)

    # Add a translucent bounding box for the brick workspace
    x_range, y_range = (-0.5, 0.5), (-0.5, 0.5)
    z_min, z_max = 0.0, 0.2

    # Define the vertices of the bounding box
    vertices = [
        [x_range[0], y_range[0], z_min],
        [x_range[1], y_range[0], z_min],
        [x_range[1], y_range[1], z_min],
        [x_range[0], y_range[1], z_min],
        [x_range[0], y_range[0], z_max],
        [x_range[1], y_range[0], z_max],
        [x_range[1], y_range[1], z_max],
        [x_range[0], y_range[1], z_max],
    ]

    # Define the 6 faces of the box
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[3], vertices[0], vertices[4], vertices[7]],  # left
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
    ]

    box = Poly3DCollection(faces, alpha=0.15, facecolor='cyan', edgecolor='blue', linewidth=1.5)
    ax_d.add_collection3d(box)

    plt.tight_layout()

    # Save figure
    output_path = Path("/workspaces/bricklaying/ransac_pipeline_figure.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    # Also save as PDF
    output_pdf = output_path.with_suffix('.pdf')
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Figure saved to: {output_pdf}")

    print("\nDone! Keep the Meshcat window open to see the 3D scene.")
    print("Meshcat URL:", meshcat.web_url())
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()