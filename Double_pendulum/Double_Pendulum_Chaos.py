import numpy as np
import sympy as sym
from sympy.matrices import Matrix
from scipy.fft import fft, fftfreq
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Functions

# ODE: transform 2nd degree ODE on 1st degree ODE
def ode(X, t):
    theta_1, v_theta_1, theta_2, v_theta_2  = X
    Y = [v_theta_1, l_1*l_2**2*m_2**2*(g*np.sin(theta_2) - l_1*np.sin(theta_1 - theta_2)*v_theta_1**2)*np.cos(theta_1 - theta_2)/(4*(i_1 + l_1**2*m_1/4 + l_1**2*m_2)*(i_2 - l_1**2*l_2**2*m_2**2*np.cos(theta_1 - theta_2)**2/(4*(i_1 + l_1**2*m_1/4 + l_1**2*m_2)) + l_2**2*m_2/4)) - l_1*(l_1**2*l_2**2*m_2**2*np.cos(theta_1 - theta_2)**2/(4*(i_1 + l_1**2*m_1/4 + l_1**2*m_2)*(i_2 - l_1**2*l_2**2*m_2**2*np.cos(theta_1 - theta_2)**2/(4*(i_1 + l_1**2*m_1/4 + l_1**2*m_2)) + l_2**2*m_2/4)) + 1)*(g*m_1*np.sin(theta_1) + 2*g*m_2*np.sin(theta_1) + l_2*m_2*np.sin(theta_1 - theta_2)*v_theta_2**2)/(2*(i_1 + l_1**2*m_1/4 + l_1**2*m_2)),v_theta_2, l_1**2*l_2*m_2*(g*m_1*np.sin(theta_1) + 2*g*m_2*np.sin(theta_1) + l_2*m_2*np.sin(theta_1 - theta_2)*v_theta_2**2)*np.cos(theta_1 - theta_2)/(4*(i_1 + l_1**2*m_1/4 + l_1**2*m_2)*(i_2 - l_1**2*l_2**2*m_2**2*np.cos(theta_1 - theta_2)**2/(4*(i_1 + l_1**2*m_1/4 + l_1**2*m_2)) + l_2**2*m_2/4)) - l_2*m_2*(g*np.sin(theta_2) - l_1*np.sin(theta_1 - theta_2)*v_theta_1**2)/(2*(i_2 - l_1**2*l_2**2*m_2**2*np.cos(theta_1 - theta_2)**2/(4*(i_1 + l_1**2*m_1/4 + l_1**2*m_2)) + l_2**2*m_2/4))]
    return Y

# System parameters
m_1 = 2.0  # Mass (kg)
m_2 = 1.0  # Mass (kg)
l_1 = 1.0  # Length (m)
l_2 = 1.5  # Length (m)
i_1 = 0.025  # (kg m^2)
i_2 = 0.075  # (kg m^2)
g = 9.81  # acceleration due to gravity (m s^-2)

# Solving the equation of movement

# Creating time interval
# 100 samples per second in 1 seconds. Sample spacing:
T = 1 / 200

# Taking 10 seconds of samples so, the number of samples is
N = 2000

# Building the interval with that omitting the last point
t = np.linspace(0, N * T, N, endpoint=False)

# Initial conditions X0 on theta_i and its time derivative
# X0 = [theta, v_theta]= [theta_1, v_theta_1, theta_2, v_theta_2]
X0 = [np.pi / 3, 0.0, np.pi * 2 / 3, 0.0]

# Finding the solution vector sol=X(t)
sol = odeint(ode, X0, t)

# Question F

# Plotting theta_i and their time derivatives over time
plt.plot(t, sol[:, 0], 'c', label=f'\u03B8₁(t) (rad)')
plt.plot(t, sol[:, 1], 'r--', label="\u03B8₁'(t) (rad / s)")
plt.plot(t, sol[:, 2], 'b', label=f'\u03B8₂(t) (rad)')
plt.plot(t, sol[:, 3], 'm--', label="\u03B8₂'(t) (rad / s)")
plt.legend(loc='upper left', prop={'size': 10})
plt.title('Double Pendulum: Time domain', fontsize=15)
plt.xlabel('Time / s', fontsize = 13)
plt.ylabel('Amplitude', fontsize = 13)
plt.grid()
plt.savefig(r"C:\\Users\\Owner\\OneDrive\\Documents\\2023-2024\\Other code\\Pendulum\\Double_pendulum\\time_domain.png", dpi=300)
plt.show()

# Question G

# Fourier transform
# Calculating the frequencies with which we are going to work
f = fftfreq(N, T)[:N // 2]

# Fourier transform of theta
sol0f_1 = fft(sol[:, 0])
sol0f_2 = fft(sol[:, 2])

# Obtaining absolute value and rescaling it
absol0f_1 = 2.0 / N * np.abs(sol0f_1[0:N // 2])
absol0f_2 = 2.0 / N * np.abs(sol0f_2[0:N // 2])

# Plotting absolute value rescaled of theta fft over positive frequencies
plt.plot(f, absol0f_1, 'c', label='\u03B8₁(f)')
plt.plot(f, absol0f_2, 'b--', label='\u03B8₂(f)')
plt.legend(loc='best', prop={'size': 13})
plt.xlim([0, 4])
plt.title('Double Pendulum: Frequency domain', fontsize = 15)
plt.xlabel('f / Hz', fontsize = 13)
plt.ylabel('fft Amplitude', fontsize = 13)
plt.grid()
plt.savefig(r"C:\\Users\\Owner\\OneDrive\\Documents\\2023-2024\\Other code\\Pendulum\\Double_pendulum\\Freq_domain.png", dpi=300)
plt.show()

# Question H

# Create figure in which animation will be show
plt.figure(figsize=(10.5, 10.5))

# Inputting the length of the array to know how many iterations there are
leng = len(sol[:, 0])

# Perform animation

# Animate the path of the extreme of the second pendulum
# We use a loop to animate the plot through time
for i in range(1, leng - 10, 3):  # We do not use the last terms (-10) because they can cause problems

    # X0 = [theta, v_theta]= [theta_1, v_theta_1, theta_2, v_theta_2]
    X0_1 = [np.pi / 3, 0.0, np.pi * 2 / 3, 0.0]
    X0_2 = [np.pi / 3, 0.0, np.pi * 2 / 3+0.1, 0.0]

    # Finding the solution vector sol=X(t)
    sol_1 = odeint(ode, X0_1, t)
    sol_2 = odeint(ode, X0_2, t)

    # Set limits of plot
    plt.xlim([-2.5 - .05, 2.5 + .05])
    plt.ylim([-2.5 - .05, 2.5 + .05])


    # Recovering the part of the solution associated to theta_i
    state_1_1 = sol_1[:, 0]  # 1st upper pendulum theta_1
    state_1_2 = sol_1[:, 2]  # 1st lower pendulum theta_2

    state_2_1 = sol_2[:, 0]  # 2nd upper pendulum theta_1
    state_2_2 = sol_2[:, 2]  # 2nd lower pendulum theta_2

    # Plot the 2nd pendulum extreme path
    plt.plot(l_1 * np.sin(state_1_1[i]) + l_2 * np.sin(state_1_2[i]), - l_1 * np.cos(state_1_1[i]) - l_2 * np.cos(state_1_2[i]), 'r.', label='I.C. 1')
    plt.plot(l_1 * np.sin(state_2_1[i]) + l_2 * np.sin(state_2_2[i]), - l_1 * np.cos(state_2_1[i]) - l_2 * np.cos(state_2_2[i]), 'b.', label='I.C. 1')

    # Styling
    plt.title(f'Path of 2nd pendulum extreme', fontsize = 15)
    plt.xlabel(f't = {(i * T):.3} sec', fontsize = 13)
    plt.draw()
    plt.pause(0.001)