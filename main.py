import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

SUN_MASS = 1.989e30
EARTH_MASS = 1.989e30
SIMULATION_DURATION_FACTOR = 1
AXIS_SIZE_FACTOR = 2
ANIMATION_SPEED = 5 #lower is faster
DISTANCE_FROM_SUN = 1.496e11

# Calculate the rate of change in position and velocity
def equations(y, t, G, m1, m2):
    r1, v1, r2, v2 = y[:2], y[2:4], y[4:6], y[6:8]
    r = np.linalg.norm(r2 - r1) # Calculate the distance between the two objects
    dr1dt = v1
    dv1dt = G * m2 * (r2 - r1) / r**3 # Acceleration of the first object due to gravitational force from the second object
    dr2dt = v2 
    dv2dt = G * m1 * (r1 - r2) / r**3 # Acceleration of the second object due to gravitational force from the first object
    return np.concatenate([dr1dt, dv1dt, dr2dt, dv2dt])

# Initial conditions and parameters
G = 6.67430e-11  # gravitational constant
m1 = SUN_MASS # mass of the Sun
m2 = EARTH_MASS # mass of the Earth

# Initial positions and velocities
r1 = np.array([0, 0])
v1 = np.array([0, 0])
r2 = np.array([DISTANCE_FROM_SUN, 0])
v2 = np.array([0, 29.78e3])

y0 = np.concatenate([r1, v1, r2, v2])
# Simulate years where 1000 steps is one year
t = np.linspace(0, 365*24*3600*SIMULATION_DURATION_FACTOR, 1000*SIMULATION_DURATION_FACTOR)

# Solve the differential equations
solution = odeint(equations, y0, t, args=(G, m1, m2))

r1_sol = solution[:, :2]
r2_sol = solution[:, 4:6]

# Create the animation
fig, ax = plt.subplots()
ax.set_xlim(-2e11 * AXIS_SIZE_FACTOR, 2e11 * AXIS_SIZE_FACTOR)
ax.set_ylim(-2e11 * AXIS_SIZE_FACTOR, 2e11 * AXIS_SIZE_FACTOR)
ax.set_aspect('equal')

sun, = ax.plot([], [], 'yo', markersize=5)  # sun size
earth, = ax.plot([], [], 'bo', markersize=5)
sun_trail, = ax.plot([], [], 'y-', alpha=0.5)  # sun trail
earth_trail, = ax.plot([], [], 'b-', alpha=0.5)

def init():
    sun.set_data([], [])
    earth.set_data([], [])
    sun_trail.set_data([], [])
    earth_trail.set_data([], [])
    return sun, earth, sun_trail, earth_trail

def update(frame):
    sun_x, sun_y = r1_sol[frame]
    earth_x, earth_y = r2_sol[frame]
    trail_x = r2_sol[:frame+1, 0]
    trail_y = r2_sol[:frame+1, 1]

    sun.set_data(np.array([sun_x]), np.array([sun_y]))
    earth.set_data(np.array([earth_x]), np.array([earth_y]))
    
    # Update Sun trail
    sun_trail.set_data(r1_sol[:frame+1, 0], r1_sol[:frame+1, 1])
    
    # Update Earth trail
    earth_trail.set_data(trail_x, trail_y)
    
    return sun, earth, sun_trail, earth_trail

# run animation
ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=ANIMATION_SPEED)

plt.show()
