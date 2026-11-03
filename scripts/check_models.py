#!/usr/bin/env python3
"""Check that the URDF, the MJCF and imu_mount.json describe the same robot.

Both models are compiled with MuJoCo and compared body by body: tree, poses,
inertials, joints (axis, range, damping, friction, armature, effort limit),
collision boxes, visual meshes and colours. The IMU frame is checked against
imu_mount.json; mjcf/scene.xml's armature, PD actuators (order, ranges, effort
limits, left/right gains) and keyframes (limits, symmetry, feet on the floor)
against the URDF. Exits non-zero on any mismatch.

    python3 scripts/check_models.py
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "urdf" / "dodo_daimao.urdf"
MJCF = ROOT / "mjcf" / "dodo_daimao.xml"
SCENE = ROOT / "mjcf" / "scene.xml"
IMU_JSON = ROOT / "imu_mount.json"

ATOL = 1e-9  # values are copied verbatim, so only float noise is allowed
errors = []


def check(ok, msg):
    if not ok:
        errors.append(msg)


def close(a, b, atol=ATOL, rtol=0.0):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=atol, rtol=rtol)


def quat_close(a, b):
    return close(a, b) or close(a, -np.asarray(b))


def body_inertia(m, b):
    """Inertia tensor about the COM, expressed in the body frame."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.body_iquat[b])
    R = R.reshape(3, 3)
    return R @ np.diag(m.body_inertia[b]) @ R.T


def rpy_to_quat(rpy):
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.asarray(rpy, float), "XYZ")  # URDF rpy = extrinsic XYZ
    return q


urdf = mujoco.MjModel.from_xml_path(str(URDF))
mjcf = mujoco.MjModel.from_xml_path(str(MJCF))

# --- bodies -------------------------------------------------------------------
u_bodies = [urdf.body(i).name for i in range(1, urdf.nbody)]
m_bodies = [mjcf.body(i).name for i in range(1, mjcf.nbody)]
check(sorted(u_bodies) == sorted(m_bodies), f"body sets differ: {sorted(set(u_bodies) ^ set(m_bodies))}")

for name in u_bodies:
    if name not in m_bodies:
        continue
    bu, bm = urdf.body(name).id, mjcf.body(name).id
    pu, pm = urdf.body(urdf.body_parentid[bu]).name, mjcf.body(mjcf.body_parentid[bm]).name
    if name != "body":  # root: URDF is welded to the world, MJCF floats at standing height
        check(pu == pm, f"{name}: parent {pu} (URDF) vs {pm} (MJCF)")
        check(close(urdf.body_pos[bu], mjcf.body_pos[bm]), f"{name}: pos {urdf.body_pos[bu]} vs {mjcf.body_pos[bm]}")
        check(quat_close(urdf.body_quat[bu], mjcf.body_quat[bm]), f"{name}: quat {urdf.body_quat[bu]} vs {mjcf.body_quat[bm]}")
    check(close(urdf.body_mass[bu], mjcf.body_mass[bm]), f"{name}: mass {urdf.body_mass[bu]} vs {mjcf.body_mass[bm]}")
    check(close(urdf.body_ipos[bu], mjcf.body_ipos[bm]), f"{name}: COM {urdf.body_ipos[bu]} vs {mjcf.body_ipos[bm]}")
    check(close(body_inertia(urdf, bu), body_inertia(mjcf, bm), atol=1e-14, rtol=1e-9), f"{name}: inertia tensor differs")

# --- joints -------------------------------------------------------------------
free = [mjcf.joint(j).name for j in range(mjcf.njnt) if mjcf.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
check(len(free) == 1 and mjcf.body(mjcf.jnt_bodyid[mjcf.joint(free[0]).id]).name == "body",
      f"MJCF should have exactly one freejoint, on 'body' (found {free})")
u_joints = [urdf.joint(j).name for j in range(urdf.njnt)]
m_joints = [mjcf.joint(j).name for j in range(mjcf.njnt) if mjcf.joint(j).name not in free]
check(u_joints == m_joints, f"joint order differs: {u_joints} vs {m_joints}")

for name in u_joints:
    if name not in m_joints:
        continue
    ju, jm = urdf.joint(name), mjcf.joint(name)
    du, dm = ju.dofadr[0], jm.dofadr[0]
    for field, a, b in [
        ("type", ju.type, jm.type),
        ("body", urdf.body(ju.bodyid[0]).name, mjcf.body(jm.bodyid[0]).name),
        ("axis", ju.axis, jm.axis),
        ("pos", ju.pos, jm.pos),
        ("limited", ju.limited, jm.limited),
        ("range", ju.range, jm.range),
        ("actfrclimited", urdf.jnt_actfrclimited[ju.id], mjcf.jnt_actfrclimited[jm.id]),
        ("actfrcrange (effort)", urdf.jnt_actfrcrange[ju.id], mjcf.jnt_actfrcrange[jm.id]),
        ("damping", urdf.dof_damping[du], mjcf.dof_damping[dm]),
        ("frictionloss", urdf.dof_frictionloss[du], mjcf.dof_frictionloss[dm]),
        ("armature", urdf.dof_armature[du], mjcf.dof_armature[dm]),
    ]:
        same = a == b if isinstance(a, str) else close(a, b)
        check(same, f"joint {name}: {field} {a} (URDF) vs {b} (MJCF)")

# --- collision geoms ----------------------------------------------------------
def collision_geoms(m, body_id):
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == body_id and (m.geom_contype[g] or m.geom_conaffinity[g]):
            out.append((int(m.geom_type[g]), m.geom_size[g].copy(), m.geom_pos[g].copy(), m.geom_quat[g].copy()))
    return sorted(out, key=lambda t: tuple(t[2]))


for name in u_bodies:
    if name not in m_bodies:
        continue
    gu, gm = collision_geoms(urdf, urdf.body(name).id), collision_geoms(mjcf, mjcf.body(name).id)
    check(len(gu) == len(gm), f"{name}: {len(gu)} collision geoms (URDF) vs {len(gm)} (MJCF)")
    for a, b in zip(gu, gm):
        check(a[0] == b[0] and close(a[1], b[1]) and close(a[2], b[2]) and quat_close(a[3], b[3]),
              f"{name}: collision geom differs: URDF {a} vs MJCF {b}")

# --- visual meshes and colours (MuJoCo's URDF import drops visuals: parse XML) --
u_xml, m_xml = ET.parse(URDF).getroot(), ET.parse(MJCF).getroot()
m_mesh_file = {e.get("name"): e.get("file") for e in m_xml.iter("mesh")}
m_mat_rgba = {e.get("name"): e.get("rgba") for e in m_xml.iter("material")}
m_body_xml = {e.get("name"): e for e in m_xml.iter("body")}
for link in u_xml.findall("link"):
    name, vis = link.get("name"), link.find("visual")
    body = m_body_xml.get(name)
    m_vis = [] if body is None else [g for g in body.findall("geom") if g.get("class") == "visual"]
    if vis is None:
        check(not m_vis, f"{name}: MJCF has a visual geom, URDF has none")
        continue
    if len(m_vis) != 1:
        check(False, f"{name}: expected 1 visual geom in MJCF, found {len(m_vis)}")
        continue
    g, origin = m_vis[0], vis.find("origin")
    u_file = Path(vis.find("geometry/mesh").get("filename")).name
    m_file = m_mesh_file.get(g.get("mesh"))
    check(u_file == m_file, f"{name}: visual mesh {u_file} (URDF) vs {m_file} (MJCF)")
    check((ROOT / "meshes" / u_file).is_file(), f"{name}: meshes/{u_file} missing")
    u_rgba = [float(x) for x in vis.find("material/color").get("rgba").split()]
    m_rgba = [float(x) for x in (m_mat_rgba.get(g.get("material")) or "nan nan nan nan").split()]
    check(close(u_rgba, m_rgba), f"{name}: colour {u_rgba} (URDF) vs {m_rgba} (MJCF)")
    u_pos = [float(x) for x in origin.get("xyz").split()]
    m_pos = [float(x) for x in g.get("pos", "0 0 0").split()]
    u_quat = rpy_to_quat([float(x) for x in origin.get("rpy").split()])
    m_quat = [float(x) for x in g.get("quat", "1 0 0 0").split()]
    check(close(u_pos, m_pos) and quat_close(u_quat, m_quat), f"{name}: visual origin differs")

# --- IMU vs imu_mount.json ----------------------------------------------------
imu = json.loads(IMU_JSON.read_text())
R_rec = np.array(imu["axis_remap"], float)
mass, box = imu["mass_kg"], np.array(imu["box_m"], float)
I_box = mass / 12 * np.array([box[1] ** 2 + box[2] ** 2, box[0] ** 2 + box[2] ** 2, box[0] ** 2 + box[1] ** 2])
for label, m in [("URDF", urdf), ("MJCF", mjcf)]:
    b = m.body("imu_link").id
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.body_quat[b])
    check(m.body(m.body_parentid[b]).name == "body", f"{label}: imu_link parent is not 'body'")
    check(close(m.body_pos[b], imu["xyz_m"]), f"{label}: imu_link pos {m.body_pos[b]} vs xyz_m {imu['xyz_m']}")
    check(close(R.reshape(3, 3), R_rec, atol=1e-8), f"{label}: imu_link rotation differs from axis_remap")
    check(close(m.body_mass[b], mass), f"{label}: imu_link mass {m.body_mass[b]} vs mass_kg {mass}")
    check(close(np.sort(m.body_inertia[b]), np.sort(I_box), atol=1e-10), f"{label}: imu_link inertia vs box_m")
s = mjcf.site("imu").id
check(mjcf.site_bodyid[s] == mjcf.body("imu_link").id, "MJCF: site 'imu' is not on imu_link")
check(close(mjcf.site_pos[s], 0) and quat_close(mjcf.site_quat[s], [1, 0, 0, 0]), "MJCF: site 'imu' is offset from imu_link")
check(close(mjcf.site_size[s], box / 2), f"MJCF: site 'imu' size {mjcf.site_size[s]} vs box_m/2")
imu_sensors = [mjcf.sensor(i).name for i in range(mjcf.nsensor) if mjcf.sensor_objid[i] == s]
check(len(imu_sensors) == 3, f"MJCF: expected gyro/accelerometer/framequat on site 'imu', found {imu_sensors}")

# --- scene.xml: armature, PD actuators, keyframes -----------------------------
scene = mujoco.MjModel.from_xml_path(str(SCENE))
hinge_dofs = [scene.jnt_dofadr[scene.joint(n).id] for n in u_joints]
arm = scene.dof_armature[hinge_dofs]
check(np.all(arm > 0) and close(arm, arm[0]), f"scene: joint armature should be one positive value, got {arm}")


def mirror(name):
    return name.replace("_right", "_left") if name.endswith("_right") else name.replace("_left", "_right")


act_joints = [scene.joint(scene.actuator_trnid[a, 0]).name for a in range(scene.nu)]
check(act_joints == u_joints, f"scene: actuator order {act_joints} vs URDF joint order {u_joints}")
for a in range(scene.nu):
    name, js, ju = act_joints[a], scene.actuator_trnid[a, 0], urdf.joint(act_joints[a]).id
    kp, bias = scene.actuator_gainprm[a, 0], scene.actuator_biasprm[a, :3]
    check(scene.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE and close(bias[:2], [0, -kp]) and bias[2] <= 0,
          f"scene: actuator {name} is not a position (PD) actuator")
    check(scene.actuator_ctrllimited[a] and close(scene.actuator_ctrlrange[a], urdf.jnt_range[ju]),
          f"scene: actuator {name} ctrlrange {scene.actuator_ctrlrange[a]} vs joint range {urdf.jnt_range[ju]}")
    check(close(scene.jnt_actfrcrange[js], urdf.jnt_actfrcrange[ju]), f"scene: joint {name} effort limit changed")
    check(close(scene.actuator_gear[a, 0], 1.0), f"scene: actuator {name} gear != 1")
    am = scene.actuator(mirror(name)).id
    check(close(scene.actuator_gainprm[a, :3], scene.actuator_gainprm[am, :3]) and close(bias, scene.actuator_biasprm[am, :3]),
          f"scene: gains of {name} and {mirror(name)} differ")

data = mujoco.MjData(scene)
robot_geoms = [g for g in range(scene.ngeom) if scene.geom_bodyid[g] != 0 and scene.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
               and (scene.geom_contype[g] or scene.geom_conaffinity[g])]  # the robot's collision boxes
for k in range(scene.nkey):
    mujoco.mj_resetDataKeyframe(scene, data, k)
    mujoco.mj_forward(scene, data)
    key = scene.key(k).name
    q = {n: data.qpos[scene.jnt_qposadr[scene.joint(n).id]] for n in u_joints}
    for n in u_joints:
        lo, hi = scene.jnt_range[scene.joint(n).id]
        check(lo <= q[n] <= hi, f"scene: key '{key}' {n}={q[n]} outside [{lo}, {hi}]")
        sign = -1 if n.startswith("hip_") else 1  # roll flips sign under left/right mirroring, pitch does not
        check(close(q[n], sign * q[mirror(n)]), f"scene: key '{key}' is not left/right symmetric at {n}")
    check(close(data.ctrl, [q[n] for n in act_joints]), f"scene: key '{key}' ctrl differs from its joint angles")
    zmin = min(np.min([(data.geom_xpos[g] + data.geom_xmat[g].reshape(3, 3) @ (scene.geom_size[g] * [sx, sy, sz]))[2]
                       for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) for g in robot_geoms)
    check(0 <= zmin < 1e-3, f"scene: key '{key}' lowest robot point at z={zmin:.5f} m, expected on the floor (0..1 mm)")

# --- report -------------------------------------------------------------------
print(f"URDF {URDF.relative_to(ROOT)}: {urdf.nbody - 1} bodies, {urdf.njnt} joints, "
      f"mass {mujoco.mj_getTotalmass(urdf):.4f} kg")
print(f"MJCF {MJCF.relative_to(ROOT)}: {mjcf.nbody - 1} bodies, {mjcf.njnt} joints (incl. freejoint), "
      f"mass {mujoco.mj_getTotalmass(mjcf):.4f} kg")
print(f"scene {SCENE.relative_to(ROOT)}: armature {arm[0]} kg m^2, "
      + ", ".join(f"{scene.actuator(a).name} kp {scene.actuator_gainprm[a, 0]:g} kv {-scene.actuator_biasprm[a, 2]:g}"
                  for a in range(4))
      + f" (left = right), keys {[scene.key(k).name for k in range(scene.nkey)]}")
if errors:
    print(f"\n{len(errors)} mismatch(es):")
    for e in errors:
        print("  -", e)
    sys.exit(1)
print("OK: URDF, MJCF, scene.xml and imu_mount.json are consistent.")
