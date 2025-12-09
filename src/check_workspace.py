"""Check if brick positions are in good workspace"""
import numpy as np

# Robot bases
left_base = np.array([-0.5, -0.5, 0])
right_base = np.array([0.5, -0.5, 0])

# Current positions
brick1 = np.array([-0.50, 0.00, 0.02])
goal1 = np.array([-0.50, 0.30, 0.02])

brick2 = np.array([0.50, 0.00, 0.02])
goal2 = np.array([0.50, 0.30, 0.02])

print("LEFT ARM:")
print(f"  Base: {left_base}")
print(f"  Brick distance: {np.linalg.norm(brick1 - left_base):.3f}m")
print(f"  Goal distance: {np.linalg.norm(goal1 - left_base):.3f}m")
print(f"  Horizontal reach to goal: {goal1[1] - left_base[1]:.3f}m")

print("\nRIGHT ARM:")
print(f"  Base: {right_base}")
print(f"  Brick distance: {np.linalg.norm(brick2 - right_base):.3f}m")
print(f"  Goal distance: {np.linalg.norm(goal2 - right_base):.3f}m")
print(f"  Horizontal reach to goal: {goal2[1] - right_base[1]:.3f}m")

print("\nIIWA 7 typical specs:")
print("  Max reach: ~0.82m")
print("  Good workspace: 0.3m - 0.7m from base")
print("\n⚠️  If goal distance > 0.75m → likely at workspace limit!")
