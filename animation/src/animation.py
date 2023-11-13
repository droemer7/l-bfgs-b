# Copyright (c) 2023 Dane Roemer droemer7@gmail.com
# Distributed under the terms of the MIT License

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import animation
from time import sleep

xb = [-2, 0]
yb = [0, 5]
zb = [0, 100]

# Attach 3D axes to the figure
fig = plt.figure(constrained_layout=False, figsize=(5,5))
ax = fig.add_subplot(projection="3d")
ax.view_init(elev=45, azim=-63, roll=0)
fig.subplots_adjust(left=-0.01)

# x lower bound data
Y_lbx = yb
Z_lbx = zb
X_lbx = [xb[0], xb[0]]
Y_lbx_mesh, Z_lbx_mesh = np.meshgrid(Y_lbx, Z_lbx)

# x upper bound data
Y_ubx = yb
Z_ubx = zb
X_ubx = [xb[1], xb[1]]
Y_ubx_mesh, Z_ubx_mesh = np.meshgrid(Y_ubx, Z_ubx)

# y lower bound data
X_lby = xb
Z_lby = zb
Y_lby = [yb[0], yb[0]]
X_lby_mesh, Z_lby_mesh = np.meshgrid(X_lby, Z_lby)

# y upper bound data
X_uby = xb
Z_uby = zb
Y_uby = [yb[1], yb[1]]
X_uby_mesh, Z_uby_mesh = np.meshgrid(X_uby, Z_uby)

# Function data
f = lambda x, y: 5*(y - x**2)**2 + (x - 1)**2
Xf = np.arange(-2, 2, 0.001)
Yf = np.arange(-2, 5, 0.001)
Xf, Yf = np.meshgrid(Xf, Yf)
Zf = f(Xf, Yf)
Zf[Zf > zb[1]] = np.nan

# Solver data
pts = []
pts.append([-2, 0, 89])
pts.append([-1, 2.5, 15.25])
pts.append([-1.14872561190384, 1.86464213610067, 6.10253702626735])
pts.append([-1.21495837728307, 1.49715896009661, 4.90825299058596])
pts.append([-1.18950570205663, 1.47295298005842, 4.81077213919468])
pts.append([-0.312844196312099, 2.18822539504618e-16, 1.7714540277058])
pts.append([-0.136547279870641, 3.71682848349416e-18, 1.29347792927137])
pts.append([-0.136547279870641, 3.71682848349416e-18, 1.29347792927137])
pts.append([-3.03195868124642e-17, 0.186451596400712, 1.17382098900187])
pts.append([-6.7323006752638e-33, 0.0471744108204238, 1.01112712518127])
pts.append([-1.49487104367547e-48, 1.04748234131921e-17, 1])
pts = np.array(pts)

# Set axis properties
ax.set(xlim3d=(-2, 2), xlabel='x0')
ax.set(ylim3d=(-2, 5), ylabel='x1')
ax.set(zlim3d=(0, 100), zlabel='f(x)')

# Show starting point with sphere
ax.scatter3D(pts[0][0], pts[0][1], pts[0][2], color="black", linewidth=2)

# Constraint lines at axis minimum
ax.plot(X_lbx, Y_lbx, [0, 0], color="black", linewidth=1.5)  # x lower bound
ax.plot(X_ubx, Y_ubx, [0, 0], color="black", linewidth=1.5)  # x upper bound
ax.plot(X_lby, Y_lby, [0, 0], color="black", linewidth=1.5)  # y lower bound
ax.plot(X_uby, Y_uby, [0, 0], color="black", linewidth=1.5)  # y upper bound

# Constraint planes
ax.plot_surface(X_lbx, Y_lbx_mesh, Z_lbx_mesh, color="gray", linewidth=0, alpha=0.5)  # x lower bound
ax.plot_surface(X_ubx, Y_ubx_mesh, Z_ubx_mesh, color="gray", linewidth=0, alpha=0.5)  # x upper bound
ax.plot_surface(X_lby_mesh, Y_lby, Z_lby_mesh, color="gray", linewidth=0, alpha=0.5)  # y lower bound
ax.plot_surface(X_uby_mesh, Y_uby, Z_uby_mesh, color="gray", linewidth=0, alpha=0.5)  # y upper bound

# Function surface
ax.plot_surface(Xf, Yf, Zf, cmap=cm.rainbow, alpha=0.8)

# Initialize line for solver path
line = ax.plot([], [], [], color="black", linewidth=2)[0]

# Figure initialization function
def init():
  ax.set_title("Iteration: {} | f(x) = {:.2f}".format(0, pts[0][2]), fontsize=11, x=0.57, y=1.1)
  line.set_data([], [])
  line.set_3d_properties([])
  return line

# Figure animation update function
start_delay = 0
end_delay = 2
loops = 3
frames_per_loop = len(pts) + start_delay + end_delay
total_frames = loops * frames_per_loop

def update(frame):
  i = frame - start_delay
  if i >= 0 and (i % frames_per_loop) < len(pts):
    i = i % frames_per_loop
    ax.set_title("Iteration: {} | f(x) = {:.2f}".format(i, pts[i][2]), fontsize=11, x=0.57, y=1.1)
    line.set_data(pts[:i+1, :2].T)
    line.set_3d_properties(pts[:i+1, 2])
  return line

# Create the animations
gif_anim = animation.FuncAnimation(fig, func=update, init_func=init, frames=frames_per_loop, interval=750)
mp4_anim = animation.FuncAnimation(fig, func=update, init_func=init, frames=total_frames, interval=750)

# Save the animations
ffmpeg = animation.FFMpegWriter(fps=1, codec="libx264", bitrate=5000)
pillow = animation.PillowWriter(fps=1, codec="libx264", bitrate=5000)
mp4_anim.save('../animation.mp4', writer=ffmpeg)
gif_anim.save('../animation.gif', writer=pillow)

# Display animation
plt.show()
