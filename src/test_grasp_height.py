"""Test the grasp height fix"""

import numpy as np
from pydrake.all import RigidTransform, RollPitchYaw
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from brick_pickplace_clean import design_grasp_pose

brick_size = [0.10, 0.08, 0.04]

# Create a brick pose at table height
brick_z = brick_size[2]/2.0 + 0.001  # Brick center height
X_WO = RigidTransform([0.0, 0.0, brick_z])

# Get grasp pose
X_OG, X_WG = design_grasp_pose(X_WO)

print("=" * 70)
print("GRASP HEIGHT TEST")
print("=" * 70)
print(f"\nBrick size: {brick_size}")
print(f"Brick center height (Z): {brick_z:.4f}m = {brick_z*1000:.1f}mm")
print(f"\nGrasp offset from brick center (p_OG): {X_OG.translation()}")
print(f"  X offset: {X_OG.translation()[0]*1000:.1f}mm")
print(f"  Y offset: {X_OG.translation()[1]*1000:.1f}mm")
print(f"  Z offset: {X_OG.translation()[2]*1000:.1f}mm")
print(f"\nGripper position in world (X_WG): {X_WG.translation()}")
print(f"  Gripper Z: {X_WG.translation()[2]:.4f}m = {X_WG.translation()[2]*1000:.1f}mm")

print(f"\nAnalysis:")
if X_OG.translation()[2] > 0.05:
    print(f"  ❌ Z offset is too high ({X_OG.translation()[2]*1000:.1f}mm)!")
    print(f"     Gripper will be {X_OG.translation()[2]*1000:.1f}mm above brick center")
    print(f"     Since brick is only {brick_size[2]*1000:.0f}mm tall, gripper will miss!")
elif X_OG.translation()[2] < 0.02:
    print(f"  ⚠  Z offset might be too low ({X_OG.translation()[2]*1000:.1f}mm)")
else:
    print(f"  ✓ Z offset looks reasonable ({X_OG.translation()[2]*1000:.1f}mm)")
    print(f"    Gripper will be ~{(X_WG.translation()[2] - brick_z)*1000:.1f}mm above brick top surface")

print("=" * 70)
