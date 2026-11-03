#!/usr/bin/env python3
"""Hold the Dodo in a fixed pose with a joint PD controller (MuJoCo).

Every joint tracks a constant target angle with

    tau = kp * (q_target - q) - kd * qdot      (clamped to the URDF effort limit)

The law runs in the PD (position) actuators of mjcf/scene.xml, so MuJoCo
integrates the kd term implicitly; computing the same tau in Python and
applying it as a torque is unstable at these gains with a 2 ms step. The
controller only holds joint angles: it does not balance. Expect the robot to
stand still, survive small pushes and fall over from larger ones.

    python3 scripts/pd_stand.py                         # viewer, "home" pose
    python3 scripts/pd_stand.py --pose straight         # legs straight
    python3 scripts/pd_stand.py --push 5                # 5 N forward push for 0.1 s at t = 1 s
    python3 scripts/pd_stand.py --kp 35 35 12 14 --kd 2.96 2.96 1.73 1.87 --armature 0.1
                                                        # Genesis training gains: too soft, falls
    python3 scripts/pd_stand.py --headless              # no window: simulate 5 s, print a report

macOS: run the viewer mode with `mjpython` instead of `python3`.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("hip", "upper_leg", "lower_leg", "foot")  # joint name = <group>_<side>


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pose", default="home", help="keyframe of mjcf/scene.xml to hold: home | straight (default home)")
    p.add_argument("--kp", type=float, nargs=4, metavar=GROUPS, help="stiffness [Nm/rad] per joint group (default: scene.xml)")
    p.add_argument("--kd", type=float, nargs=4, metavar=GROUPS, help="damping [Nm s/rad] per joint group (default: scene.xml)")
    p.add_argument("--armature", type=float, help="motor armature [kg m^2] for all joints (default: scene.xml)")
    p.add_argument("--push", type=float, default=0.0, help="forward push on the body for 0.1 s at t = 1 s [N]")
    p.add_argument("--headless", action="store_true", help="no viewer: simulate --duration seconds and report")
    p.add_argument("--duration", type=float, default=5.0, help="headless simulation time [s] (default 5)")
    args = p.parse_args()

    m = mujoco.MjModel.from_xml_path(str(ROOT / "mjcf" / "scene.xml"))
    d = mujoco.MjData(m)
    names = [m.actuator(a).name for a in range(m.nu)]  # = joint names, URDF order
    group = np.array([GROUPS.index(n.rsplit("_", 1)[0]) for n in names])
    jnt = m.actuator_trnid[:, 0]
    qadr, vadr = m.jnt_qposadr[jnt], m.jnt_dofadr[jnt]

    # MuJoCo position actuator: force = gain * ctrl + bias0 + bias1 * q + bias2 * qdot
    #                                 = kp * ctrl - kp * q - kd * qdot
    if args.kp:
        kp = np.array(args.kp)[group]
        m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = kp, -kp
    if args.kd:
        m.actuator_biasprm[:, 2] = -np.array(args.kd)[group]
    if args.armature is not None:
        m.dof_armature[vadr] = args.armature

    mujoco.mj_resetDataKeyframe(m, d, m.key(args.pose).id)
    q_target = d.ctrl.copy()  # the keyframe's ctrl = its joint angles; held fixed from here on
    body = m.body("body").id

    print(f"pose '{args.pose}', timestep {m.opt.timestep * 1e3:g} ms, "
          f"integrator {mujoco.mjtIntegrator(m.opt.integrator).name}, armature {m.dof_armature[vadr[0]]:g} kg m^2")
    print(f"{'joint':16s} {'target [deg]':>12s} {'kp':>7s} {'kd':>6s} {'effort [Nm]':>11s}")
    for a in range(m.nu):
        print(f"{names[a]:16s} {np.degrees(q_target[a]):12.1f} {m.actuator_gainprm[a, 0]:7.1f} "
              f"{-m.actuator_biasprm[a, 2]:6.2f} {m.jnt_actfrcrange[jnt[a], 1]:11.0f}")

    z0, min_z, max_tilt = d.qpos[2], d.qpos[2], 0.0
    max_tau = np.zeros(m.nu)

    def step():
        nonlocal min_z, max_tilt
        d.xfrc_applied[body, 0] = args.push if (args.push and 1.0 <= d.time < 1.1) else 0.0
        mujoco.mj_step(m, d)
        if d.time > 0.5:  # skip the initial settling
            max_tau[:] = np.maximum(max_tau, np.abs(d.actuator_force))
        min_z = min(min_z, d.qpos[2])
        up_z = d.xmat[body][8]  # world z of the body's z axis
        max_tilt = max(max_tilt, np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))

    if args.headless:
        while d.time < args.duration:
            step()
    else:
        from mujoco import viewer as mj_viewer  # GUI only when asked for

        print("\nviewer: close the window to get the report "
              "(Ctrl + right-drag on the body pushes it; Backspace resets to qpos0, i.e. straight legs)")
        with mj_viewer.launch_passive(m, d) as v:
            while v.is_running():
                t0 = time.time()
                step()
                v.sync()
                time.sleep(max(0.0, m.opt.timestep - (time.time() - t0)))

    fell = min_z < 0.6 * z0 or max_tilt > 45.0
    print(f"\nsimulated {d.time:.2f} s{f' with a {args.push:g} N push' if args.push else ''}: "
          f"base height {z0:.4f} -> {d.qpos[2]:.4f} m (min {min_z:.4f}), max tilt {max_tilt:.1f} deg "
          f"-> {'FELL' if fell else 'stayed up'}")
    print("final tracking error [deg]: " + ", ".join(
        f"{n} {e:+.2f}" for n, e in zip(names, np.degrees(d.qpos[qadr] - q_target))))
    print("max |torque| after 0.5 s [Nm]: " + ", ".join(f"{n} {t:.2f}" for n, t in zip(names, max_tau)))

    # The IMU sensors read in the chip frame (imu_link); axis_remap turns them into body-frame values.
    remap = np.array(json.loads((ROOT / "imu_mount.json").read_text())["axis_remap"], float)
    gyro, acc = d.sensor("imu_gyro").data, d.sensor("imu_acc").data
    print(f"IMU gyro  [rad/s]  chip {np.round(gyro, 3)}  body {np.round(remap @ gyro, 3)}")
    print(f"IMU accel [m/s^2]  chip {np.round(acc, 3)}  body {np.round(remap @ acc, 3)}"
          "  (standing still: about +9.81 along body z)")


if __name__ == "__main__":
    main()
