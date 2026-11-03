#!/usr/bin/env python3
"""Inspect the Dodo in MuJoCo with the physics frozen, and pose it by hand.

Prints a summary of the model (bodies, joints, actuators, sensors, mass and
static balance in the chosen pose), then opens the MuJoCo viewer on a copy of
mjcf/scene.xml without gravity, contacts or actuators, so the robot stays
exactly where it is put.

In the viewer:
  * Space pauses; the sliders of the right-hand "Joint" panel only work while
    paused (MuJoCo's rule). Drag them to set the joint angles [rad].
  * Simulation > Key: load "home" or "straight" again.
  * Keys 0-5 toggle geom groups: 2 = visual meshes, 3 = collision boxes.
    Shift+5 shows the IMU site (box with the chip's axes, see Rendering > Frame).
  * Rendering panel: Frame (body axes), Center of mass, Inertia, Joint (axes).
  * F2 info, F4 sensor plots; Tab / Shift+Tab hide the side panels.

    python3 scripts/view_static.py                  # starts in "home"
    python3 scripts/view_static.py --pose straight
    python3 scripts/view_static.py --no-viewer      # summary only
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "mjcf" / "scene.xml"


def section(title):
    print(f"\n== {title} " + "=" * (72 - len(title)))


def summary(m, d, pose):
    section("Model")
    print(f"{SCENE.relative_to(ROOT)}: {m.nbody - 1} bodies, nq {m.nq} (7 free base + {m.nq - 7} joints), "
          f"nv {m.nv}, {m.nu} actuators, {m.nsensor} sensors, total mass {mujoco.mj_getTotalmass(m):.3f} kg")

    section(f"Bodies (pose '{pose}', world frame)")
    print(f"{'body':17s} {'parent':17s} {'mass [kg]':>9s}   {'COM x, y, z [m]':>24s}")
    for b in range(1, m.nbody):
        x, y, z = d.xipos[b]
        print(f"{m.body(b).name:17s} {m.body(m.body_parentid[b]).name:17s} {m.body_mass[b]:9.4f}   "
              f"{x:8.4f} {y:8.4f} {z:8.4f}")

    section("Joints = actuators, in order (this is the policy's action order)")
    print(f"{'#':>2s} {'joint':16s} {'axis':>6s} {'range [deg]':>16s} {'effort':>6s} {'damping':>7s} "
          f"{'armature':>8s} {'kp':>6s} {'kd':>5s} {'q [deg]':>8s}")
    for a in range(m.nu):
        j = m.actuator_trnid[a, 0]
        dof = m.jnt_dofadr[j]
        lo, hi = np.degrees(m.jnt_range[j])
        axis = "xyz"[int(np.argmax(np.abs(m.jnt_axis[j])))]
        print(f"{a:2d} {m.joint(j).name:16s} {'+' + axis:>6s} {lo:7.1f} .. {hi:6.1f} "
              f"{m.jnt_actfrcrange[j, 1]:6.0f} {m.dof_damping[dof]:7.3f} {m.dof_armature[dof]:8.4f} "
              f"{m.actuator_gainprm[a, 0]:6.0f} {-m.actuator_biasprm[a, 2]:5.2f} "
              f"{np.degrees(d.qpos[m.jnt_qposadr[j]]):8.1f}")
    print("Effort in Nm (URDF limit). Axes are in the parent frame; positive = right-hand rule.")

    section("Sensors (IMU chip frame; for vectors, body frame = imu_mount.json axis_remap @ value)")
    for s in range(m.nsensor):
        kind = mujoco.mjtSensor(m.sensor_type[s]).name.removeprefix("mjSENS_").lower()
        note = "orientation of the chip frame in the world" if kind == "framequat" else "3D vector in the chip frame"
        print(f"{m.sensor(s).name:9s} {kind:14s} dim {m.sensor_dim[s]}  site '{m.site(m.sensor_objid[s]).name}': {note}")

    section(f"Static balance in pose '{pose}'")
    com = d.subtree_com[m.body("body").id]
    soles = [d.xpos[m.body(n).id] for n in ("foot_sole_right", "foot_sole_left")]
    corners = []
    for g in range(m.ngeom):
        name = m.geom(g).name
        if name.startswith("foot_") and name.endswith("_collision"):
            R, size = d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
            corners += [d.geom_xpos[g] + R @ (size * [sx, sy, -1]) for sx in (-1, 1) for sy in (-1, 1)]
    corners = np.array(corners)
    lo, hi = corners[:, :2].min(0), corners[:, :2].max(0)
    margin = min(com[0] - lo[0], hi[0] - com[0], com[1] - lo[1], hi[1] - com[1])
    print(f"base height {d.qpos[2]:.4f} m, whole-robot COM {np.round(com, 4)} m")
    print(f"sole frames: right {np.round(soles[0], 4)}, left {np.round(soles[1], 4)} "
          f"(stance width {soles[1][1] - soles[0][1]:.3f} m)")
    print(f"feet cover x [{lo[0]:.3f}, {hi[0]:.3f}], y [{lo[1]:.3f}, {hi[1]:.3f}] m; COM projection "
          f"{'inside' if margin > 0 else 'OUTSIDE'} that rectangle, {abs(margin) * 1e3:.0f} mm from its nearest edge")


def frozen_copy():
    """scene.xml without gravity, contacts and actuators: nothing moves unless the user moves it."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    for a in list(spec.actuators):
        spec.delete(a)
    for k in spec.keys:
        k.ctrl = []
    spec.option.gravity = [0.0, 0.0, 0.0]
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return spec.compile()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pose", default="home", help="keyframe to start from: home | straight (default home)")
    p.add_argument("--no-viewer", action="store_true", help="print the summary only")
    args = p.parse_args()

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(args.pose).id)
    mujoco.mj_forward(m, d)
    summary(m, d, args.pose)
    if args.no_viewer:
        return

    from mujoco import viewer as mj_viewer  # GUI only when asked for

    mf = frozen_copy()
    df = mujoco.MjData(mf)
    mujoco.mj_resetDataKeyframe(mf, df, mf.key(args.pose).id)
    print("\nOpening the viewer (gravity, contacts and actuators off). Press Space to pause, then drag the "
          "sliders in the right-hand 'Joint' panel.")
    mj_viewer.launch(mf, df)


if __name__ == "__main__":
    main()
