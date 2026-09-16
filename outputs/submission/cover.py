"""Render a 16:9 cover frame at the moment both arms hold the plate clear of the table.

Not a mock-up: this drives the real teacher and captures the frame where the episode's
own gate says the plate is airborne, level, and held by all four pads.
"""
import mujoco
from PIL import Image

from tabletop_vla.sim.bimanual_plate import BimanualPlateLift, prepare_bimanual_workcell
from tabletop_vla.sim.runtime import Simulation

W, H = 1280, 720
sim = Simulation()
sim.reset(50)
prepare_bimanual_workcell(sim, 50)
sim.task = task = BimanualPlateLift(sim)
sim.running = True

best = None  # (lift height, pixels) at the most airborne, most level instant
# The scene's `overview` camera sits far back and leaves the arms tiny in a mostly
# black frame. Frame the plate itself: close, low, and off-axis so the gap between the
# plate and the table is visible, which is the whole point of the shot.
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance = 0.70
cam.azimuth = 0.0   # set per-variant below
cam.elevation = -12

with mujoco.Renderer(sim.model, height=H, width=W) as renderer:
    for _ in range(60):
        def watch(s):
            global best
            if task.phase < 0:
                return
            name = task.phases[task.phase][0]
            if name not in {"Lift clear of table", "Carry to mat"}:
                return
            geoms = task.contacts()
            airborne = (task.all_pads <= geoms and not (geoms - task.robot_geoms)
                        and task.lift > 0.045 and task.tilt_deg() < 15.0)
            if airborne and (best is None or task.lift > best[0]):
                best = (task.lift, s.data.qpos.copy(), task.tilt_deg())
        sim.step(200, observer=watch)
        if task.status != "running":
            break

print("episode:", task.status, "| peak lift", round(task.peak_lift, 4))
if best is None:
    raise SystemExit("no airborne frame captured")
lift, qpos, tilt = best
# Re-render the captured instant from several angles: both arms approach from -y, so a
# front view puts an arm column across the plate. Pick the angle that shows the gap
# under the plate, which is the whole subject of the shot.
sim.data.qpos[:] = qpos
mujoco.mj_forward(sim.model, sim.data)
with mujoco.Renderer(sim.model, height=H, width=W) as renderer:
    for az, el in ((60, -14), (90, -12), (240, -14), (270, -12), (300, -16)):
        cam.azimuth, cam.elevation = az, el
        cam.lookat[:] = sim.data.xpos[task.body]
        renderer.update_scene(sim.data, camera=cam)
        Image.fromarray(renderer.render()).save(f"outputs/submission/cover-az{az}.png")
print(f"captured at lift {lift*1000:.0f} mm, tilt {tilt:.1f} deg; variants written")
sim.close()
