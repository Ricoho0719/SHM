import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# PARAMETERS FOR THE ANIMATION
# ---------------------------
R = 1.0                # Radius of the circular motion
omega = 2 * np.pi / 5  # Angular velocity (period = 5 sec)
t_max = 10             # Maximum time [sec]
fps = 30               # Frames per second
n_frames = int(t_max * fps)

# Create the full time array
t_full = np.linspace(0, t_max, n_frames)

# Precompute circular motion quantities
x_full = R * np.cos(omega * t_full)
y_full = R * np.sin(omega * t_full)
z_full = np.zeros_like(t_full)

# Compute the SHM projection along x:
d_full = x_full.copy()                # displacement (x-projection)
v_full = -R * omega * np.sin(omega * t_full)  # velocity
a_full = -R * omega**2 * np.cos(omega * t_full) # acceleration
F_full = a_full.copy()  # force (for a unit mass)

# ---------------------------
# SETTING UP THE FIGURE AND AXES
# ---------------------------
fig = plt.figure(figsize=(12, 10))
fig.suptitle("Circular Motion, SHM Projections, and Real-Time Force", fontsize=16)

grid = plt.GridSpec(4, 4, figure=fig, wspace=0.5, hspace=0.6)

# 3D animation panel (left two columns)
ax3d = fig.add_subplot(grid[:, :2], projection='3d')
ax3d.set_title("3D Circular Motion")
lim = R * 1.2
ax3d.set_xlim([-lim, lim])
ax3d.set_ylim([-lim, lim])
ax3d.set_zlim([-lim, lim])
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")
ax3d.view_init(elev=20, azim=45)

# Time-series plots on the right
ax_f = fig.add_subplot(grid[0, 2:])
ax_f.set_title("Force vs Time")
ax_f.set_xlim(0, t_max)
force_lim = R * omega**2 * 1.2
ax_f.set_ylim(-force_lim, force_lim)
ax_f.set_ylabel("F (N)")

ax_d = fig.add_subplot(grid[1, 2:])
ax_d.set_title("Displacement vs Time")
ax_d.set_xlim(0, t_max)
ax_d.set_ylim(-lim, lim)
ax_d.set_ylabel("d (m)")

ax_v = fig.add_subplot(grid[2, 2:])
ax_v.set_title("Velocity vs Time")
ax_v.set_xlim(0, t_max)
vel_lim = abs(R * omega) * 1.2
ax_v.set_ylim(-vel_lim, vel_lim)
ax_v.set_ylabel("v (m/s)")

ax_a = fig.add_subplot(grid[3, 2:])
ax_a.set_title("Acceleration vs Time")
ax_a.set_xlim(0, t_max)
ax_a.set_ylim(-force_lim, force_lim)
ax_a.set_ylabel("a (m/s²)")
ax_a.set_xlabel("Time (s)")

# ---------------------------
# INITIAL PLOTS
# ---------------------------
ax3d.plot(x_full, y_full, z_full, "gray", lw=0.5, alpha=0.5)

point_3d, = ax3d.plot([], [], [], "ro", markersize=8, label="Particle")
line_center_to_point, = ax3d.plot([], [], [], "r:", lw=2, label="Radius")
proj_line, = ax3d.plot([], [], [], "g--", lw=1, label="Projection (SHM)")
force_quiver = None  # placeholder for the force arrow

line_f, = ax_f.plot([], [], "orange", lw=2, label="F(t)")
line_d, = ax_d.plot([], [], "r-", lw=2, label="d(t)")
line_v, = ax_v.plot([], [], "m-", lw=2, label="v(t)")
line_a, = ax_a.plot([], [], "c-", lw=2, label="a(t)")

# ---------------------------
# ANIMATION FUNCTION
# ---------------------------
def update(frame):
    global force_quiver
    t = t_full[frame]
    x = x_full[frame]
    y = y_full[frame]
    z = z_full[frame]
    
    # SHM quantities at time t
    d = d_full[frame]
    v = v_full[frame]
    a = a_full[frame]
    
    point_3d.set_data([x], [y])
    point_3d.set_3d_properties([z])
    
    line_center_to_point.set_data([0, x], [0, y])
    line_center_to_point.set_3d_properties([0, z])
    
    proj_line.set_data([x, x], [0, 0])
    proj_line.set_3d_properties([0, z])
    
    # Update force vector: remove previous quiver if exists.
    if force_quiver is not None:
        force_quiver.remove()
    
    fx = -R * omega**2 * np.cos(omega * t)
    fy = -R * omega**2 * np.sin(omega * t)
    fz = 0
    force_quiver = ax3d.quiver(x, y, z, fx, fy, fz, color='orange', arrow_length_ratio=0.2)
    
    current_slice = slice(0, frame + 1)
    line_f.set_data(t_full[current_slice], F_full[current_slice])
    line_d.set_data(t_full[current_slice], d_full[current_slice])
    line_v.set_data(t_full[current_slice], v_full[current_slice])
    line_a.set_data(t_full[current_slice], a_full[current_slice])
    
    return (point_3d, line_center_to_point, proj_line, line_f, line_d, line_v, line_a)

ani = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

# Instead of using plt.show(), convert the animation to HTML5 video.
html_video = ani.to_html5_video()

# ---
# STREAMLIT CODE TO EMBED THE VIDEO
# ---
st.title("Circular Motion and SHM Animation")
st.markdown("This animation shows circular motion and its corresponding SHM projections.")
st.markdown(html_video, unsafe_allow_html=True)