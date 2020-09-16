import pyglet
import pymunk

from scipy import linalg
import numpy as np

from trajectory import MinimumSnapTrajectory

SCREEN_HEIGHT = 700
window = pyglet.window.Window(1000, SCREEN_HEIGHT, vsync=False, caption='Quadcopter Simulator')

# setup the space
space = pymunk.Space()
space.gravity = 0, -9.8

# quadcopter body
qc_mass = 0.2
qc_size = 0.05, 0.05
qc_moment = 0.001
qc_body = pymunk.Body(mass=qc_mass, moment=qc_moment)
qc_body.position = 0.0, 0.0
qc_shape = pymunk.Poly.create_box(qc_body, qc_size)
space.add(qc_body, qc_shape)
arm_length = 0.15

print(f"quadcopter mass = {qc_body.mass:0.1f} kg, quadcopter moment = {qc_body.moment:0.3f} kg*m^2")

# simulation stuff
f1 = 0.0
f2 = 0.0
MAX_FORCE = 2
DT = 1 / 60.0
ref = (0.0, 0.0)
currtime = 0.0

# drawing stuff
# pixels per meter
PPM = 200.0

color = (200, 200, 200, 200)
label_pos = pyglet.text.Label(text='', font_size=18, color=color, x=10, y=SCREEN_HEIGHT - 28)
label_ang = pyglet.text.Label(text='', font_size=18, color=color, x=10, y=SCREEN_HEIGHT - 58)
label_force = pyglet.text.Label(text='', font_size=18, color=color, x=10, y=SCREEN_HEIGHT - 88)
label_time = pyglet.text.Label(text='', font_size=18, color=color, x=10, y=SCREEN_HEIGHT - 118)

labels = [label_pos, label_ang, label_force, label_time]

# invert our A matrix once so we can quickly calculate forces
A = np.array([[1, 1], [arm_length, -arm_length]])
A_inv = linalg.inv(A)

# PD gains
# manually tuned these in the simulation since i don't have the discrete model setup yet
pdy, pdz, pdtheta = (6.0, 4.5), (12, 6.5), (3.5, 0.14)

points = [(0.0, 0.0), (1.0, 1.0), (0, 2.0), (-1.0, 1.0),  (0.0, 1.0), (0.0, 2.0), (0.0, 0.0)]
times = [ii*2 for ii in range(len(points))]
print(times)
minsnap = MinimumSnapTrajectory()
minsnap.generate(points, times)


def draw_body(offset, body):
    for shape in body.shapes:
        if isinstance(shape, pymunk.Poly):
            # get vertices in world coordinates
            vertices = [v.rotated(body.angle) + body.position for v in shape.get_vertices()]

            # convert vertices to pixel coordinates
            points = []
            for v in vertices:
                points.append(int(v[0] * PPM) + offset[0])
                points.append(int(v[1] * PPM) + offset[1])

            data = ('v2i', tuple(points))
            pyglet.graphics.draw(len(vertices), pyglet.gl.GL_LINE_LOOP, data)


def draw_arms(offset, body):
    left_arm = pymunk.Vec2d(-arm_length, 0.0)
    right_arm = pymunk.Vec2d(arm_length, 0.0)
    vertices = [left_arm.rotated(body.angle) + body.position, right_arm.rotated(body.angle) + body.position]

    # convert vertices to pixel coordinates
    points = []
    for v in vertices:
        points.append(int(v[0] * PPM) + offset[0])
        points.append(int(v[1] * PPM) + offset[1])

    data = ('v2i', tuple(points))
    pyglet.graphics.draw(len(vertices), pyglet.gl.GL_LINES, data)


def draw_points(offset):
    vertices = []
    for p in points:
        vertices.append(int(p[0] * PPM) + offset[0])
        vertices.append(int(p[1] * PPM) + offset[1])

    data = ('v2i', tuple(vertices))
    pyglet.graphics.draw(len(points), pyglet.gl.GL_POINTS, data)


@window.event
def on_draw():
    window.clear()

    # center view x around 0
    offset = (500, 100)
    draw_body(offset, qc_body)
    draw_arms(offset, qc_body)
    draw_points(offset)

    for label in labels:
        label.draw()


def simulate(_):
    # ensure we get a consistent simulation step - ignore the input dt value
    dt = DT

    # simulate the world
    # NOTE: using substeps will mess up gains
    space.step(dt)

    global currtime
    currtime += dt

    if currtime > (times[-1] + 2.5):
        currtime = 0

    # get our desired state at this time
    values = minsnap.getvalues(currtime)
    yvals = values[0]
    zvals = values[1]

    # populate the current state
    posy = qc_body.position[0]
    posz = qc_body.position[1]
    vely = qc_body.velocity[0]
    velz = qc_body.velocity[1]
    ang = qc_body.angle
    angv = qc_body.angular_velocity

    u1 = qc_mass * (-space.gravity[1] + zvals[2] + (zvals[1] - velz) * pdz[1] + (zvals[0] - posz) * pdz[0])
    thetac = (-1 / -space.gravity[1]) * (yvals[2] + (yvals[1] - vely) * pdy[1] + (yvals[0] - posy) * pdy[0])
    u2 = -pdtheta[1] * angv + pdtheta[0] * (thetac - ang)

    f = A_inv.dot(np.array([[u1], [u2]]))

    global f1, f2
    # clamp the forces to the allowable range [0, MAX_FORCE]
    f1 = max(0, min(f[0][0], MAX_FORCE))
    f2 = max(0, min(f[1][0], MAX_FORCE))

    # apply force to cart center of mass
    qc_body.apply_force_at_local_point((0.0, f1), (arm_length, 0))
    qc_body.apply_force_at_local_point((0.0, f2), (-arm_length, 0))


# function to store the current state to draw on screen
def update_state_label(_):
    label_pos.text = f'Position: ({qc_body.position[0]:0.3f}, {qc_body.position[1]:0.3f}) m'
    label_ang.text = f'Angle: {qc_body.angle:0.3f} radians'
    label_force.text = f'Force: ({f2:0.3f}, {f1:0.3f}) newtons'
    label_time.text = f'Time: {currtime:0.1f}'


def update_reference(_, newref):
    global ref
    ref = newref


# callback for simulation
pyglet.clock.schedule_interval(simulate, DT)
pyglet.clock.schedule_interval(update_state_label, 0.25)

pyglet.app.run()
