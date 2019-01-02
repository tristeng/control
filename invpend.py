import csv
import math

import pyglet
import pymunk

SCREEN_HEIGHT = 700
window = pyglet.window.Window(1000, SCREEN_HEIGHT, vsync=False, caption='Inverted Pendulum Simulator')

# setup the space
space = pymunk.Space()
space.gravity = 0, -9.8

fil = pymunk.ShapeFilter(group=1)

# ground
ground = pymunk.Segment(space.static_body, (-4, -0.1), (4, -0.1), 0.1)
ground.friction = 0.1
ground.filter = fil
space.add(ground)

# cart
cart_mass = 0.5
cart_size = 0.3, 0.2
cart_moment = pymunk.moment_for_box(cart_mass, cart_size)
cart_body = pymunk.Body(mass=cart_mass, moment=cart_moment)
cart_body.position = 0.0, cart_size[1] / 2
cart_shape = pymunk.Poly.create_box(cart_body, cart_size)
cart_shape.friction = ground.friction
space.add(cart_body, cart_shape)

# pendulum
pend_length = 0.6  # to center of mass
pend_size = 0.1, pend_length * 2  # to get CoM at 0.6 m
pend_mass = 0.2
pend_moment = 0.001
pend_body = pymunk.Body(mass=pend_mass, moment=pend_moment)
pend_body.position = cart_body.position[0], cart_body.position[1] + cart_size[1] / 2 + pend_length
pend_shape = pymunk.Poly.create_box(pend_body, pend_size)
pend_shape.filter = fil
space.add(pend_body, pend_shape)

# joint
joint = pymunk.constraint.PivotJoint(cart_body, pend_body, cart_body.position + (0, cart_size[1] / 2))
joint.collide_bodies = False
space.add(joint)

print(f"cart mass = {cart_body.mass:0.1f} kg")
print(f"pendulum mass = {pend_body.mass:0.1f} kg, pendulum moment = {pend_body.moment:0.3f} kg*m^2")

# K gain matrix and Nbar found from modelling via Jupyter
K = [-57.38901804, -36.24133932, 118.51380879, 28.97241832]
Nbar = -57.25

# simulation stuff
force = 0.0
MAX_FORCE = 25
DT = 1 / 60.0
ref = 0.0

# drawing stuff
# pixels per meter
PPM = 200.0

color = (200, 200, 200, 200)
label_x = pyglet.text.Label(text='', font_size=18, color=color, x=10, y=SCREEN_HEIGHT - 28)
label_ang = pyglet.text.Label(text='', font_size=18, color=color, x=10, y=SCREEN_HEIGHT - 58)
label_force = pyglet.text.Label(text='', font_size=18, color=color, x=10, y=SCREEN_HEIGHT - 88)

labels = [label_x, label_ang, label_force]

# data recorder so we can compare our results to our predictions
f = open('data/invpend.csv', 'w')
out = csv.writer(f)
out.writerow(['time', 'x', 'theta'])
currtime = 0.0
record_data = False


def draw_body(offset, body):
    for shape in body.shapes:
        if isinstance(shape, pymunk.Circle):
            # TODO
            pass
        elif isinstance(shape, pymunk.Poly):
            # get vertices in world coordinates
            vertices = [v.rotated(body.angle) + body.position for v in shape.get_vertices()]

            # convert vertices to pixel coordinates
            points = []
            for v in vertices:
                points.append(int(v[0] * PPM) + offset[0])
                points.append(int(v[1] * PPM) + offset[1])

            data = ('v2i', tuple(points))
            pyglet.graphics.draw(len(vertices), pyglet.gl.GL_LINE_LOOP, data)


def draw_ground(offset):
    vertices = [v + (0, ground.radius) for v in (ground.a, ground.b)]

    # convert vertices to pixel coordinates
    points = []
    for v in vertices:
        points.append(int(v[0] * PPM) + offset[0])
        points.append(int(v[1] * PPM) + offset[1])

    data = ('v2i', tuple(points))
    pyglet.graphics.draw(len(vertices), pyglet.gl.GL_LINES, data)


@window.event
def on_draw():
    window.clear()

    # center view x around 0
    offset = (500, 5)
    draw_body(offset, cart_body)
    draw_body(offset, pend_body)
    draw_ground(offset)

    for label in labels:
        label.draw()


def simulate(_):
    # ensure we get a consistent simulation step - ignore the input dt value
    dt = DT

    # simulate the world
    # NOTE: using substeps will mess up gains
    space.step(dt)

    # populate the current state
    posx = cart_body.position[0]
    velx = cart_body.velocity[0]
    ang = pend_body.angle
    angv = pend_body.angular_velocity

    # dump our data so we can plot
    if record_data:
        global currtime
        out.writerow([f"{currtime:0.4f}", f"{posx:0.3f}", f"{ang:0.3f}"])
        currtime += dt

    # calculate our gain based on the current state
    gain = K[0] * posx + K[1] * velx + K[2] * ang + K[3] * angv

    # calculate the force required
    global force
    force = ref * Nbar - gain

    # kill our motors if we go past our linearized acceptable angles
    if math.fabs(pend_body.angle) > 0.35:
        force = 0.0

    # cap our maximum force so it doesn't go crazy
    if math.fabs(force) > MAX_FORCE:
        force = math.copysign(MAX_FORCE, force)

    # apply force to cart center of mass
    cart_body.apply_force_at_local_point((force, 0.0), (0, 0))


# function to store the current state to draw on screen
def update_state_label(_):
    label_x.text = f'Cart X: {cart_body.position[0]:0.3f} m'
    label_ang.text = f'Pendulum Angle: {pend_body.angle:0.3f} radians'
    label_force.text = f'Force: {force:0.1f} newtons'


def update_reference(_, newref):
    global ref
    ref = newref


# callback for simulation
pyglet.clock.schedule_interval(simulate, DT)
pyglet.clock.schedule_interval(update_state_label, 0.25)

# schedule some small movements by updating our reference
pyglet.clock.schedule_once(update_reference, 2, 0.2)
pyglet.clock.schedule_once(update_reference, 7, 0.6)
pyglet.clock.schedule_once(update_reference, 12, 0.2)
pyglet.clock.schedule_once(update_reference, 17, 0.0)

pyglet.app.run()

# close the output file
f.close()
