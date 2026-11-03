# Dodo assets

Robot description of **Dodo** (`dodo_daimao`), the 8-DoF bipedal robot of the course
**Dodo Alive! – Resurrecting the Dodo with Robotics and AI: Simulation & Control**.
In the course you will train control policies for this robot with reinforcement
learning. This repository holds the models those simulations start from (URDF,
MuJoCo MJCF, meshes, IMU mount) and two small MuJoCo scripts to get to know the
robot.

```
urdf/dodo_daimao.urdf     robot description (source of truth), for Isaac Lab, Genesis, PyBullet, ...
mjcf/dodo_daimao.xml      the same robot as MuJoCo MJCF (no actuators: for mjlab and your own scenes)
mjcf/scene.xml            floor + PD actuators + keyframes around it: open this one in the MuJoCo viewer
meshes/*.STL              visual meshes (binary STL, metres), shared by URDF and MJCF
imu_mount.json            IMU position / orientation on the body, as measured on the robot
scripts/view_static.py    model summary + viewer with frozen physics: pose the robot with sliders
scripts/pd_stand.py       PD controller holding a fixed pose, with pushes and a report
```

## Quick start

```bash
git clone git@github.com:Atarilab/dodo_assets.git
cd dodo_assets
pip install mujoco                      # brings numpy; tested with MuJoCo 3.10, Python 3.13

python3 scripts/view_static.py          # print the model summary, then pose the robot by hand
python3 scripts/pd_stand.py             # watch the robot stand under PD control
python3 -m mujoco.viewer --mjcf mjcf/scene.xml   # plain MuJoCo viewer
```

On macOS, run `pd_stand.py` with `mjpython` instead of `python3` (MuJoCo's passive viewer needs it).

## The robot

| | |
|---|---|
| Degrees of freedom | 6 (floating base) + 8 actuated joints, 4 per leg |
| Mass | 4.72 kg: body 3.31 kg, each leg 0.70 kg, IMU 5 g |
| Height of the base origin | 0.447 m with straight legs, 0.417 m in the `home` pose |
| Stance width | 0.30 m between the sole centres |
| Base frame (`body`) | x forward, y left, z up; origin on the hip-pitch axis, midway between the legs |

### Joints

The order below is the URDF order. It is also the actuator order in `scene.xml` and
the order of the joints in MuJoCo's `qpos[7:15]` / `qvel[6:14]`, i.e. a sensible
order for a policy's actions.

| # | joint | function | axis | range [rad] | effort [Nm] |
|---|---|---|---|---|---|
| 0 | `hip_right` | hip roll | x | −0.35 … 0.35 | 27 |
| 1 | `upper_leg_right` | hip pitch | y | −1.57 … 1.57 | 27 |
| 2 | `lower_leg_right` | knee | y | −3.1416 … 1.3963 | 9 |
| 3 | `foot_right` | ankle pitch | y | −1.05 … 1.57 | 9 |
| 4–7 | `hip_left`, `upper_leg_left`, `lower_leg_left`, `foot_left` | same as 0–3 | | | |

Every joint has a 6 rad/s velocity limit and 0.01 Nm·s/rad damping in the URDF.
Link and joint names are the same in the URDF and the MJCF (each joint is named
after the link it moves).

**Signs.** A positive angle rotates the child link about the axis by the right-hand
rule. All axes are the parent's +x or +y, which at zero are the base's axes.

* Hip pitch +: the leg swings backward.
* Knee −: the shank swings forward. With hip pitch + and knee −, the knee points
  backward, like a bird's leg (see the `home` pose).
* Ankle +: toes down.
* Hip roll +: the foot moves to the left. That pulls the right leg inward and
  pushes the left leg outward.

**Left/right mirroring** (for symmetric data augmentation or rewards): swap the
legs, negate the two hip roll angles, and keep the pitch joints. For base
quantities, negate the y component of vectors such as linear velocity and
projected gravity, and the x and z components of angular velocity.

### Poses (keyframes in `mjcf/scene.xml`)

| key | joints (per leg: hip, upper_leg, lower_leg, foot) | base height |
|---|---|---|
| `home` | 0, 0.45, −1.0, 0.55 rad: the bent-leg standing pose used for training upstream | 0.4168 m |
| `straight` | all 0: legs straight down (the URDF's zero configuration) | 0.4468 m |

In both poses hip pitch + knee + ankle = 0, so the feet stay parallel to the body.
The base heights put the feet on the floor.

## The model files

### URDF: `urdf/dodo_daimao.urdf`

This is the SolidWorks export of the robot. Its kinematics, inertias, limits and
collision boxes are unchanged from
[liambansbach/legged-robot_rl_genesis](https://github.com/liambansbach/legged-robot_rl_genesis)
@ `1cfa733` (`ressources/robots/dodo_daimao/`), where it was used to train a
walking policy in Genesis. Since then two things were added:

* a `<mujoco><compiler fusestatic="false"/></mujoco>` block. MuJoCo merges links
  attached by fixed joints when it imports a URDF; this keeps them. Other
  simulators ignore the block.
* the IMU (`imu_link` on the fixed `imu_joint`), generated from `imu_mount.json`
  by `imu_locate.py`, a tool that is not part of this repository. Do not edit that
  block by hand.

Fixed links: `foot_sole_right` / `foot_sole_left` (origin on the bottom face of the
sole) and `imu_link`. Load the URDF with a floating base; MuJoCo's own URDF import
fixes the base to the world, so use the MJCF in MuJoCo.

### MJCF: `mjcf/dodo_daimao.xml`

The same robot for MuJoCo: same body tree, names, inertials, joint axes, ranges,
damping, effort limits (as `actuatorfrcrange`) and collision boxes, with every
number copied from the URDF. It adds:

* a free joint `floating_base_joint` on `body`;
* the visual meshes (geom group 2; MuJoCo's URDF import drops them);
* names for the collision boxes, `<link>_collision` (geom group 3);
* an `imu` site in `imu_link` with three sensors, `imu_gyro`, `imu_acc` and
  `imu_quat` (see [IMU](#imu)).

It has no actuators and no motor armature, like the URDF. These are simulation
choices: set them in your framework's actuator configuration, as mjlab does.

### Scene: `mjcf/scene.xml`

Includes the robot and adds a floor, light, a 2 ms `implicitfast` time step,
the two keyframes above, and simulation settings chosen so that a PD controller
alone keeps the robot standing:

| | hip | upper_leg | lower_leg | foot |
|---|---|---|---|---|
| kp [Nm/rad] | 150 | 150 | 50 | 50 |
| kd [Nm·s/rad] | 1.5 | 1.5 | 0.8 | 0.8 |
| armature [kg·m²] | 0.005 | 0.005 | 0.005 | 0.005 |

The actuators are MuJoCo position actuators: `force = kp (ctrl − q) − kd q̇`,
clamped to the effort limit, where `ctrl` is the joint target in rad. In the
viewer, the sliders of the right-hand "Control" panel set these targets.

The armature is a **placeholder**, of the same order as a Unitree Go1 hip motor's
reflected inertia (0.004 kg·m² in mjlab's Go1 model). Replace it with *rotor inertia × gear ratio²* from the Dodo motors'
datasheets.

## Scripts

All scripts locate the files relative to the repository, so run them from anywhere.

### `scripts/view_static.py`: look at the robot and pose it by hand

Prints the model summary (bodies with masses and COM positions, the joint table with
limits and gains, the sensors, and static balance in the chosen pose). It then
opens the MuJoCo viewer with gravity, contacts and actuators switched off, so the
robot stays exactly where you put it.

* Press **Space** to pause. MuJoCo enables the sliders in the right-hand
  **Joint** panel only while paused; drag them to set the joint angles.
* **Simulation > Key** reloads `home` or `straight`.
* Keys **2** / **3** toggle the visual meshes and the collision boxes. **Shift+5**
  shows the IMU site.
* The **Rendering** panel draws body frames, centres of mass, inertia boxes and
  joint axes.

```bash
python3 scripts/view_static.py --pose straight
python3 scripts/view_static.py --no-viewer        # summary only
```

### `scripts/pd_stand.py`: stand under PD control

Every joint holds a fixed target (the keyframe's angles) with the PD actuators of
`scene.xml`. The script prints the gains, then simulates (in the viewer, or for 5 s
with `--headless`) and reports:

* base height and tilt, and whether the robot fell;
* the tracking error and peak torque per joint;
* the IMU readings in the chip frame and in the body frame.

```bash
python3 scripts/pd_stand.py --pose straight
python3 scripts/pd_stand.py --push 5 --headless   # forward push on the body, 0.1 s at t = 1 s
python3 scripts/pd_stand.py --kp 35 35 12 14 --kd 2.96 2.96 1.73 1.87 --armature 0.1   # Genesis gains
python3 scripts/pd_stand.py --armature 0          # the bare URDF model
```

Expected results:

* `home` and `straight` both stand. The body leans forward by a few degrees (max tilt
  about 6° and 3°), because the knees and ankles give way slightly under the weight.
* A 5 N push is survived in `home`. Pushes of about 10 N tip the robot over, because
  a joint PD controller holds angles and does not balance. Balancing is the
  policy's job.
* The Genesis training gains are too soft to stand on their own.
* Without armature, the feet chatter and the robot falls even with the default gains.

## IMU

An Arduino Nicla Sense ME (Bosch BHI260AP) sits on the body at
(0.0142, 0, 0.095) m, measured by hand on 2026-11-02 (`imu_mount.json`).
`imu_link` carries the **chip's own axes**, and its rotation in the body frame is
`axis_remap`:

```
v_body = axis_remap @ v_chip,   axis_remap = [[0, 1, 0],
                                              [1, 0, 0],
                                              [0, 0, -1]]
```

That is, body x = chip y, body y = chip x, and body z = −chip z: the board is
mounted upside down with x and y swapped.

The MJCF sensors read in the chip frame, exactly like the real IMU, so apply the
same `axis_remap` in simulation as on the robot (`pd_stand.py` shows both).
Standing still, the accelerometer measures about +9.81 m/s² along body z; it
measures the reaction to gravity, not gravity itself.

## Before training policies: what the model does not tell you

* **No actuator model.** The URDF has no rotor inertia, motor friction or
  torque–speed curve; the MJCF follows it. `scene.xml` uses a placeholder armature
  (see above). Genesis applies `default_armature = 0.1` to every joint unless told
  otherwise, so the upstream training used 0.1. Match the real motors before sim-to-real.
* **Gains.** `scene.xml`'s gains are tuned to stand still. The upstream Genesis
  training used kp 35/35/12/14, kd = 0.5·√kp, action scale 0.25 around the `home`
  pose, with the policy at 50 Hz. Neither set is necessarily what the real motor
  drivers run; use those for sim-to-real.
* **Velocity limit.** The URDF's 6 rad/s has no MuJoCo equivalent. Enforce it in
  your environment, by clipping, a penalty or termination.
* **Contacts.** All collision shapes are boxes. Each `foot_sole_*_collision` box
  lies inside its `foot_*_collision` box, whose bottom is 0.7 mm lower, so the
  ground touches `foot_*_collision`. Use those geoms (or both) for foot-contact
  sensing and rewards.
* **Model accuracy.** Masses and inertias come from CAD, and the joint damping
  (0.01) and zero friction from the export, not from measurements. Randomise
  mass, friction, gains, armature and latency during training.

