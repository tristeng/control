import logging
import math

import pyglet
import pymunk

from scipy import linalg
import numpy as np

from trajectory import MininumTrajectory, TrajectoryType

logging.basicConfig(level=logging.INFO)

SCREEN_HEIGHT = 700
SCREEN_WIDTH = 1000
window = pyglet.window.Window(SCREEN_WIDTH, SCREEN_HEIGHT, vsync=False, caption='Quadcopter Simulator')

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

logging.info(f'quadcopter mass = {qc_body.mass:0.1f} kg, quadcopter moment = {qc_body.moment:0.3f} kg*m^2')

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

# center view x around 0
offset = (500, 100)

label_color = (200, 200, 200, 200)
curr_label_offset = 28


def create_label():
    global curr_label_offset
    label = pyglet.text.Label(text='', font_size=18, color=label_color, x=10, y=SCREEN_HEIGHT - curr_label_offset)
    curr_label_offset += 30
    return label


label_traj = create_label()
label_pos = create_label()
label_ang = create_label()
label_force = create_label()
label_time = create_label()

labels = [label_pos, label_ang, label_force, label_time, label_traj]

# invert our A matrix once so we can quickly calculate forces
A = np.array([[1, 1], [arm_length, -arm_length]])
A_inv = linalg.inv(A)

# PD gains
# manually tuned these in the simulation since i don't have the discrete model setup yet
pdy, pdz, pdtheta = (6.0, 4.5), (12, 6.5), (3.5, 0.14)

# generate min trajectories based off of an average velocity and set of waypoints
avg_velocity = 0.75  # m/s
points = [(0.0, 0.0),  (1.0, 1.0), (0.0, 2.0), (-1.0, 1.0),  (0.0, 1.0), (0.0, 2.5), (0.0, 0.0)]
# points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 2.0),  (-1.0, 1.0), (-1.0, 0.0), (0.0, 0.0)]

# use our desired average velocity to create our desired waypoint times
times = [0.0]
for idx, start in enumerate(points[:-1]):
    end = points[idx+1]
    times.append(times[-1] + math.sqrt((start[0]-end[0])**2 + (start[1]-end[1])**2) / avg_velocity)

# create various min trajectories so we can compare them
minsnap = MininumTrajectory(TrajectoryType.SNAP)
minsnap.generate(points, times)

minjerk = MininumTrajectory(TrajectoryType.JERK)
minjerk.generate(points, times)

minacc = MininumTrajectory(TrajectoryType.ACCELERATION)
minacc.generate(points, times)

minvel = MininumTrajectory(TrajectoryType.VELOCITY)
minvel.generate(points, times)

# we will iterate over the generated trajectories
trajectory_gens = [minsnap, minjerk, minacc, minvel]
traj_index = 0
traj_labels = ['Minimum Snap Trajectory', 'Minimum Jerk Trajectory', 'Minimum Acceleration Trajectory',
               'Minimum Velocity Trajectory']

# create the trajectories for drawing
samples = 25
tvals = np.linspace(times[0], times[-1], samples*(len(points) - 1))


def gen_traj_points(traj_generator):
    return [(value[0][0], value[1][0]) for value in [traj_generator.getvalues(tval) for tval in tvals]]


snap_points = gen_traj_points(minsnap)
jerk_points = gen_traj_points(minjerk)
acc_points = gen_traj_points(minacc)
vel_points = gen_traj_points(minvel)


def generate_gl_desired_trajectory(traj_points):
    """
    Generate the desired trajectory for display - do this once to make it more efficient
    :return: a graphics vertex list
    """
    traj_vertices = []
    colors = []
    for v in traj_points:
        traj_vertices.append(int(v[0] * PPM) + offset[0])
        traj_vertices.append(int(v[1] * PPM) + offset[1])
        colors += [200, 200, 200]

    data = ('v2i', tuple(traj_vertices))
    colors = ('c3B', tuple(colors))

    return pyglet.graphics.vertex_list(samples*(len(points) - 1), data, colors)


gl_snap_points = generate_gl_desired_trajectory(snap_points)
gl_jerk_points = generate_gl_desired_trajectory(jerk_points)
gl_acc_points = generate_gl_desired_trajectory(acc_points)
gl_vel_points = generate_gl_desired_trajectory(vel_points)

gl_trajectories = [gl_snap_points, gl_jerk_points, gl_acc_points, gl_vel_points]


def generate_gl_desired_points():
    """
    Generate the desired waypoints for display - do this once to make it more efficient
    :return: a graphics vertex list
    """
    point_vertices = []
    colors = []
    cross_length = 5  # pixels
    for p in points:
        point_vertices.append(int(p[0] * PPM) + offset[0] - cross_length)
        point_vertices.append(int(p[1] * PPM) + offset[1])
        colors += [255, 0, 0]
        point_vertices.append(int(p[0] * PPM) + offset[0] + cross_length)
        point_vertices.append(int(p[1] * PPM) + offset[1])
        colors += [255, 0, 0]

        point_vertices.append(int(p[0] * PPM) + offset[0])
        point_vertices.append(int(p[1] * PPM) + offset[1] - cross_length)
        colors += [255, 0, 0]
        point_vertices.append(int(p[0] * PPM) + offset[0])
        point_vertices.append(int(p[1] * PPM) + offset[1] + cross_length)
        colors += [255, 0, 0]

    data = ('v2i', tuple(point_vertices))
    colors = ('c3B', tuple(colors))

    return pyglet.graphics.vertex_list(4*len(points), data, colors)


gl_desired_points = generate_gl_desired_points()

# record the actual y,z points to plot them to see how close to the desired trjectoy we get
actual_trajectory = []


def draw_body(body):
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


def draw_arms(body):
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


def draw_desired_trajectory():
    gl_trajectories[traj_index].draw(pyglet.gl.GL_LINES)


def draw_points():
    gl_desired_points.draw(pyglet.gl.GL_LINES)


def draw_actual_trajectory():
    actual_traj_vertices = []
    colors = []
    num_vertices = 0
    for v in actual_trajectory:
        actual_traj_vertices.append(int(v[0] * PPM) + offset[0])
        actual_traj_vertices.append(int(v[1] * PPM) + offset[1])
        colors += [128, 128, 0]
        num_vertices += 1

    data = ('v2i', tuple(actual_traj_vertices))
    colors = ('c3B', tuple(colors))
    pyglet.graphics.draw(num_vertices, pyglet.gl.GL_LINES, data, colors)


@window.event
def on_draw():
    window.clear()

    draw_body(qc_body)
    draw_arms(qc_body)
    draw_desired_trajectory()
    draw_points()
    draw_actual_trajectory()

    for label in labels:
        label.draw()


def simulate(_):
    # ensure we get a consistent simulation step - ignore the input dt value
    dt = DT

    # simulate the world
    # NOTE: using substeps will mess up gains
    space.step(dt)

    global currtime, actual_trajectory, traj_index
    currtime += dt

    # restart the simulation once we have gone a couple seconds past our final waypoint
    if currtime > (times[-1] + 2.5):
        currtime = 0
        actual_trajectory = []

        # iterate over each trajectory generator
        traj_index += 1
        if traj_index >= len(trajectory_gens):
            traj_index = 0

    # get our desired state at this time
    # values = minsnap.getvalues(currtime)
    values = trajectory_gens[traj_index].getvalues(currtime)
    yvals = values[0]  # a list of desired y pos, vel and acc
    zvals = values[1]  # a list of desired z pos, vel and acc

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
    label_time.text = f'Time: {currtime:0.1f} s'
    label_traj.text = traj_labels[traj_index]


def update_reference(_, newref):
    global ref
    ref = newref


def record_curr_pos(_):
    posy = qc_body.position[0]
    posz = qc_body.position[1]
    actual_trajectory.append((posy, posz))


# callback for simulation
pyglet.clock.schedule_interval(simulate, DT)
pyglet.clock.schedule_interval(update_state_label, 0.25)
pyglet.clock.schedule_interval(record_curr_pos, 0.1)

pyglet.app.run()
