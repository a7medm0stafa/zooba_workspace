# Plan: `controll_sim` — Closed-Loop Simulation Stack

## Overview

A **new** launch file `controll_sim.launch.py` will be created inside `zooba_simulation/launch/`.  
Two new Python nodes will live in `zooba_simulation/zooba_simulation/` (alongside `sim_bridge_node.py`).  
No node from `mid_level_controller` is used.

---

## Nodes Involved

| Node | Package | File | Role |
|---|---|---|---|
| `sim_bridge_node` | `zooba_simulation` | *(existing)* | Reads `/joint_states` + `/tf`, writes `/vehicle/state`. Receives `/vehicle/cmd` and drives Gazebo |
| **`speed_control_node`** | `zooba_simulation` | `sim_speed_control_node.py` *(new)* | PI controller → publishes `/sim/speed_cmd` (Float64, m/s) |
| **`lateral_control_node`** | `zooba_simulation` | `sim_lateral_control_node.py` *(new)* | Stanley controller → publishes `/sim/lateral_cmd` (Float64, deg) |
| **`sim_cmd_merger_node`** *(inline in launch or tiny node)* | — | merged in node OR simple timer | Combines Float64 speed + Float64 heading → `/vehicle/cmd` (VehicleCmd) |

> **Note:** Instead of a separate merger node, the two new nodes will each hold the latest state of the _other_ axis and one of them will publish the merged `VehicleCmd`. Alternatively, a very small third node `sim_cmd_merger_node.py` (new, in `zooba_simulation`) merges them cleanly. **A third small merger node is preferred for clarity.**

---

## Topic Graph

```
Gazebo Simulator
     │  /joint_states, /tf
     ▼
┌──────────────────────┐
│   sim_bridge_node    │ ──► /vehicle/state (VehicleState)
│   (zooba_simulation) │ ◄── /vehicle/cmd  (VehicleCmd)
│                      │ ──► /steering_angle, /velocity
└──────────────────────┘
          ▲
          │  /vehicle/cmd (VehicleCmd)
          │
┌────────────────────────┐
│   sim_cmd_merger_node  │ ◄── /sim/speed_cmd   (Float64, m/s)
│   (zooba_simulation)   │ ◄── /sim/lateral_cmd (Float64, deg)
└────────────────────────┘
          ▲                    ▲
          │                    │
┌──────────────────┐  ┌───────────────────────┐
│speed_control_node│  │ lateral_control_node  │
│ PI Controller    │  │ Stanley Controller     │
│ Sub: /vehicle/   │  │ Sub: /vehicle/state   │
│      state       │  │ Pub: /sim/lateral_cmd │
│ Pub: /sim/       │  │      (Float64, deg)   │
│      speed_cmd   │  └───────────────────────┘
│ (Float64, m/s)   │
└──────────────────┘
```

---

## Files to Create

### 1. `zooba_simulation/zooba_simulation/sim_speed_control_node.py` *(NEW)*

**PI speed controller.**

- **Subscribes:** `/vehicle/state` (VehicleState)
- **Publishes:** `/sim/speed_cmd` (Float64, m/s)
- **Parameters (all settable in launch file):**
  - `desired_speed` — goal speed [m/s]
  - `kp`, `ki` — PI gains
  - `max_velocity` — saturation [m/s]
  - `control_rate` — Hz
- **Console output (pretty-printed every cycle, throttled):**
  ```
  ╔══════════ SPEED CONTROL ══════════╗
  ║  Target    :  0.50 m/s
  ║  True vel  :  0.32 m/s
  ║  Error     : +0.18 m/s
  ║  Ctrl eff  :  0.21 m/s
  ╚═══════════════════════════════════╝
  ```

---

### 2. `zooba_simulation/zooba_simulation/sim_lateral_control_node.py` *(NEW)*

**Stanley lateral controller.**

- **Subscribes:** `/vehicle/state` (VehicleState)
- **Publishes:** `/sim/lateral_cmd` (Float64, degrees)
- **Parameters (all settable in launch file):**
  - `desired_y` — target lateral lane/distance [m]
  - `desired_heading` — target heading [rad]
  - `k_stanley` — cross-track gain
  - `k_soft` — softening constant
  - `max_steering_angle` — saturation [deg]
  - `control_rate` — Hz
- **Console output (pretty-printed, throttled):**
  ```
  ╔══════════ LATERAL CONTROL ════════╗
  ║  Target Y  :  2.00 m
  ║  Actual Y  :  1.74 m   X:  3.21 m
  ║  Heading ψ : +5.3°     Target: 0.0°
  ║  CTE       : +0.26 m
  ║  Ctrl eff  : -8.4°
  ╚═══════════════════════════════════╝
  ```

---

### 3. `zooba_simulation/zooba_simulation/sim_cmd_merger_node.py` *(NEW)*

**Merges Float64 speed + Float64 heading → VehicleCmd.**

- **Subscribes:** `/sim/speed_cmd` (Float64), `/sim/lateral_cmd` (Float64)
- **Publishes:** `/vehicle/cmd` (VehicleCmd)
- No parameters needed (just pass-through at fixed rate).

---

### 4. `zooba_simulation/launch/controll_sim.launch.py` *(NEW)*

**Single launch file for the full closed-loop stack.**

All editable arguments:

| Arg | Default | Description |
|---|---|---|
| `world` | `empty.sdf` | Gazebo world |
| `x` | `0.0` | Initial X position [m] |
| `y` | `0.0` | Initial Y position [m] |
| `z` | `0.1` | Initial Z (height) [m] |
| `Y` | `0.0` | Initial Yaw (heading) [rad] |
| `desired_speed` | `0.5` | Target speed [m/s] |
| `desired_y` | `0.0` | Target lateral lane [m] |
| `desired_heading` | `0.0` | Target heading [rad] |
| `kp` | `1.5` | PI proportional gain |
| `ki` | `0.2` | PI integral gain |
| `max_velocity` | `2.0` | Speed saturation [m/s] |
| `k_stanley` | `2.5` | Stanley cross-track gain |
| `k_soft` | `1.0` | Stanley softening constant |
| `max_steering_angle` | `35.0` | Steering saturation [deg] |

**Nodes launched:**
1. Gazebo (via `simulation.launch.py` include)
2. `sim_bridge_node`
3. `sim_speed_control_node`
4. `sim_lateral_control_node`
5. `sim_cmd_merger_node`

---

## `setup.py` Changes

Add three new entry points in `zooba_simulation/setup.py`:
```python
'sim_speed_control_node = zooba_simulation.sim_speed_control_node:main',
'sim_lateral_control_node = zooba_simulation.sim_lateral_control_node:main',
'sim_cmd_merger_node = zooba_simulation.sim_cmd_merger_node:main',
```

---

## Example Usage

```bash
# Default run (speed=0.5 m/s, straight lane y=0):
ros2 launch zooba_simulation controll_sim.launch.py

# Custom pose + goals:
ros2 launch zooba_simulation controll_sim.launch.py \
    x:=1.0 y:=0.5 Y:=0.0 \
    desired_speed:=0.8 desired_y:=2.0 \
    kp:=2.0 ki:=0.3 k_stanley:=3.0
```

---

> [!IMPORTANT]
> **No node from `mid_level_controller` is used.** The pipeline goes:
> `speed_control_node` + `lateral_control_node` → `sim_cmd_merger_node` → `sim_bridge_node` → Gazebo

> [!NOTE]
> `closed_loop_sim.launch.py` is **not touched** — the new launch file is `controll_sim.launch.py`.
