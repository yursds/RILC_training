import numpy as np
import matplotlib.pyplot as plt

# Lissajous figure parameters
f0 = 1.0
ny = 3
nz = 1
ampY = 0.2
ampZ = 0.2
k = 2
offy = 0.05
offz = 0.5
dy = 0
dz = k * np.pi / 4 / ny
a = 2 * np.pi * ny * f0
b = 2 * np.pi * nz * f0


# Time interval
t = np.linspace(0, 2 * np.pi, 1000)

y = ampY * np.cos(a * t + dy) + offy
z = ampZ * np.cos(b * t + dz) + offz

vy = - 2 * a * ampY * np.sin(a * t + dy)
vz = - 2 * b * ampZ * np.sin(b * t + dz)

ay = - 4 * a * a * ampY * np.cos(a * t + dy)
az = - 4 * b * b * ampZ * np.cos(b * t + dz)

plt.figure(figsize=(8, 8))
plt.plot(t, vy, label='vy')
plt.plot(t, vz, label='vz')

# Lissajous figure plot
plt.figure(figsize=(8, 8))
plt.plot(y, z, label='Lissajous Figure')


plt.title(f'Lissajous Figure with zero velocity points, ratio {ny}:{nz}')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()


def MJT_1D(t: float, tf: float, x0: float, xf: float, dx0: float, dxf: float, ddx0: float, ddxf: float):
    """
    Compute position, velocity, and acceleration for a 1D minimum jerk trajectory.
    
    Args:
        t (float): Current time.
        tf (float): Total duration of the trajectory.
        x0, xf (float): Initial and final position.
        dx0, dxf (float): Initial and final velocity.
        ddx0, ddxf (float): Initial and final acceleration.
    
    Returns:
        np.ndarray: Vector of 3 elements [position, velocity, acceleration].
    """
    if t < 0:
        t = 0.0
    if t > tf:
        t = tf

    t2 = t**2
    t3 = t**3
    t4 = t**4
    t5 = t**5

    tf2 = tf**2
    tf3 = tf**3
    tf4 = tf**4
    tf5 = tf**5

    C5 = (12*x0 - 12*xf + 6*dxf*tf + 6*dx0*tf - ddxf*tf2 + ddx0*tf2) / (2*tf5)
    C3 = (20*x0 - 20*xf + 8*dxf*tf + 12*dx0*tf - ddxf*tf2 + 3*ddx0*tf2) / (2*tf3)
    C4 = (30*x0 - 30*xf + 14*dxf*tf + 16*dx0*tf - 2*ddxf*tf2 + 3*ddx0*tf2) / (2*tf4)

    # Position computation
    x = x0 + dx0*t + (ddx0*t2)/2 - C5*t5 - C3*t3 + C4*t4
    
    # Velocity computation
    dx = dx0 + ddx0*t - 5*C5*t4 - 3*C3*t2 + 4*C4*t3
    
    # Acceleration computation
    ddx = ddx0 - 20*C5*t3 - 6*C3*t + 12*C4*t2

    return np.array([x, dx, ddx])


