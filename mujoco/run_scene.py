import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_XML = "scene.xml"


def print_joint_info(model):
    """Print all joint names and their qpos indices once, for debugging."""
    print("======= JOINTS IN MODEL =======")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        qpos_addr = model.jnt_qposadr[j]
        jtype = model.jnt_type[j]
        print(f"{j:2d}: name={name:20s} qpos_idx={qpos_addr:2d} type={jtype}")
    print("================================")


def set_joint_deg(model, data, joint_name, angle_deg):
    """
    Set a hinge or ball joint in *degrees*.
    For slide joints, 'angle_deg' is treated as a linear displacement (meters).
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid == -1:
        print(f"[WARN] joint '{joint_name}' not found, skipping.")
        return

    qpos_addr = model.jnt_qposadr[jid]
    jtype = model.jnt_type[jid]

    if jtype == mujoco.mjtJoint.mjJNT_HINGE:
        data.qpos[qpos_addr] = np.deg2rad(angle_deg)
    elif jtype == mujoco.mjtJoint.mjJNT_BALL:
        # For ball joints, qpos is a quaternion (4 values) starting at qpos_addr.
        # Here we just set a simple rotation around z as an example.
        # You can customize if you want something more specific.
        half_rad = np.deg2rad(angle_deg) / 2.0
        data.qpos[qpos_addr + 0] = np.cos(half_rad)  # w
        data.qpos[qpos_addr + 1] = 0.0              # x
        data.qpos[qpos_addr + 2] = 0.0              # y
        data.qpos[qpos_addr + 3] = np.sin(half_rad) # z
    elif jtype == mujoco.mjtJoint.mjJNT_SLIDE:
        data.qpos[qpos_addr] = angle_deg  # interpret as meters, not degrees
    else:
        print(f"[WARN] joint '{joint_name}' has unsupported type {jtype}, skipping.")


def set_initial_pose(model, data):
    """
    Put the G1 into some reasonable bent-knee pose.
    EDIT THE DICT BELOW with the actual joint names from your model.
    """
    # Example: fill this after you look at the printed joints.
    # Replace these with the real names from print_joint_info().
    target_pose_deg = {
        # "g1_l_hip_pitch": -10.0,
        # "g1_l_knee":       20.0,
        # "g1_r_hip_pitch": -10.0,
        # "g1_r_knee":       20.0,
        # ...
    }

    for jname, angle in target_pose_deg.items():
        set_joint_deg(model, data, jname, angle)


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)

    # Print all joints once so you can see the actual names.
    print_joint_info(model)

    # Set your custom pose before starting simulation
    set_initial_pose(model, data)
    mujoco.mj_forward(model, data)

    # Passive viewer (no fancy control yet)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running():
            # If you want to send torques later, do it here: data.ctrl[...] = ...
            data.ctrl[:] = 0.0

            # Step sim for ~0.01s
            for _ in range(5):
                mujoco.mj_step(model, data)

            viewer.sync()


if __name__ == "__main__":
    main()
