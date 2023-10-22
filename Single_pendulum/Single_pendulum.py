import numpy as np
import sympy as sym
from scipy.fft import fft, fftfreq
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


import matplotlib
print(matplotlib.matplotlib_fname())


# Define the ODE function
def ode(X, t):
    theta, vtheta = X
    return [vtheta, k * np.sin(theta)]

# Constants and Symbols for sympy
m, r, i, g, t = sym.symbols('m r i g t')
theta = sym.Function('theta')(t)

# Relations between variables
x, y = r * sym.sin(theta), -r * sym.cos(theta)

# File Writing Section
with open("SinglePendulum.txt", "w") as f:
    # Write relations between the Cartesian coordinates and the generalized coordinate
    f.write('a. Relations between the Cartesian coordinates, x and y, and the generalized coordinate, theta(t). \n')
    f.write(f'x = {x.simplify()} \n')
    f.write(f'y = {y.simplify()} \n \n')
    
    # Velocities and energies
    vtheta, vx, vy = theta.diff(t), x.diff(t), y.diff(t)
    f.write('b. Velocities on x,y axis in terms of theta(t) \n')
    f.write(f'v_x = {vx.simplify()} \n')
    f.write(f'v_y = {vy.simplify()} \n \n')
    
    T_rec = sym.Rational(1, 2) * m * (vx**2 + vy**2)
    f.write('Rectilinear kinetic energy \n')
    f.write(f'T_rec = {T_rec.simplify()} \n \n')
    
    T_rot = sym.Rational(1, 2) * i * vtheta**2
    f.write('Rotational kinetic energy \n')
    f.write(f'T_rot = {T_rot.simplify()} \n \n')
    
    T = T_rec + T_rot
    f.write('Total kinetic energy \n')
    f.write(f'T = {T.simplify()} \n \n')
    
    U = m * g * y - m * g * -r
    f.write('c. Potential energy \n')
    f.write(f'U = {U.simplify()} \n \n')
    
    L = T - U
    f.write('Lagrangian \n')
    f.write(f'L = T - U = {L.simplify()} \n \n')
    
    Q = L.diff(vtheta).diff(t) - L.diff(theta)
    f.write('d. equations of the system. Q represents the torque \n')
    f.write(f'Q = {Q.simplify()} \n \n')

# System parameters
m_val = 1.0
l = 1.0
i_val = 0.025
g_val = 9.81
k = -m_val * g_val * l / (i_val + m_val * l * l)

# Solving the equation of motion
T_sample = 1/100
N = 1000
t_array = np.linspace(0, N * T_sample, N, endpoint=False)
X01 = [np.pi / 2, 0.0]
sol1 = odeint(ode, X01, t_array)

# Plotting theta and its derivative over time
plt.figure(figsize=(10, 5))
plt.plot(t_array, sol1[:, 0], 'r', label=r'\u03B8(t) (rad)')
plt.plot(t_array, sol1[:, 1], 'b', label=r"\u03B8'(t) (rad/s)")
plt.legend(loc='upper right', fontsize=13)
plt.title('Single Pendulum: Theta and its derivative as time functions', fontsize=15)
plt.xlabel('Time / s', fontsize=13)
plt.ylabel('Amplitude', fontsize=13)
plt.grid()
plt.savefig("Single_task.png", dpi=300)
plt.show()

# Plotting phase diagrams
plt.figure(figsize=(18, 9))
for i in range(-10, 10, 1):
    for j in range(-6, 6, 1):
        X01 = [i, j]
        sol1 = odeint(ode, X01, t_array)
        theta_vals = sol1[:, 0]
        v_theta_vals = sol1[:, 1]
        plt.plot(theta_vals, v_theta_vals)
        plt.plot(theta_vals, -v_theta_vals)
plt.xlim([-2 * np.pi, 2 * np.pi])
plt.ylim([-9, 9])
plt.title('Phase diagram', fontsize=30)
plt.xlabel(r'\u03B8(t) (rad)', fontsize=25)
plt.ylabel(r'\u03B8(t) (rad/s)', fontsize=25)
plt.savefig("Single_Phase_Diagram.png", dpi=300)
plt.show()

# Fourier transform
frequencies = fftfreq(N, T_sample)[:N // 2]
sol1_fft = fft(sol1[:, 0])
absol1f = 2.0 / N * np.abs(sol1_fft[0:N // 2])
plt.plot(frequencies, absol1f, 'r')
plt.title('Single Pendulum: Frequency domain Theta', fontsize=15)
plt.xlabel('f / Hz', fontsize=13)
plt.ylabel('fft Amplitude', fontsize=13)
plt.grid()
plt.savefig("Single_fft.png", dpi=300)
plt.show()

# Animation
fig, ax = plt.subplots(figsize=(10.5, 5.5))
ax.set_xlim([-1 - .05, 1 + .05])
ax.set_ylim([-1 - .05, .05])
ax.set_title('Single pendulum animation')
line, = ax.plot([], [], 'r-', lw=2)
ax.plot(0, 0, 'kp')

# Animation function for FuncAnimation
def animate(i):
    state = sol1[:, 0]
    line.set_data([0, np.sin(state[i])], [0, -np.cos(state[i])])
    ax.set_xlabel(f't = {(i * T_sample):.2f} sec')
    return line,

ani = FuncAnimation(fig, animate, frames=len(sol1[:, 0]), blit=True, repeat=False, interval=10)
video_path_mod = "C:\\Users\\Owner\\OneDrive\\Documents\\2023-2024\\Other code\\pendulum_animation.mp4"
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Owner\\OneDrive\\Documents\\ffmpeg-2023-10-18-git-e7a6bba51a-full_build\\bin\\ffmpeg.exe'
ani.save(video_path_mod, writer=writer)
