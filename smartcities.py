# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem

Kp = 1e4       


Va = (1.5*1.85+2.56*3.62)*2.5          # m³ volume of air
ACH = 1             # air changes per hour
Va_dot = ACH * Va / 3600    # m³/s air infiltration

#propriétés air
d_air=.2                      # kg/m³
c_air=1000               # J/kg.K
#propriétés bois
d_bois=1400
c_bois=2050
cond_bois=0.065
e_porte=0.05
S_porte=2
#propriétés concrete
d_c=2300
c_c=880
cond_c=1.4
e_c=0.16
S_c=6.590*2.5

#propriété Insulation
d_i=25
c_i=1030
cond_i=0.03
e_i=0.08
S_i=S_c

#propriété placo
d_pla=825
c_pla=1225
cond_pla=0.3
e_pla=0.01
S_pla=S_c

#propriété verre
d_v=2500
c_v=750
cond_v=1.4
e_v=0.02
S_v=3.620*2.5


ε_wLW = 0.9     # long wave wall emmisivity (concrete)
α_wSW = 0.2     # absortivity white surface
ε_gLW = 0.9     # long wave glass emmisivity (glass pyrex)
τ_gSW = 0.83    # short wave glass transmitance (glass)
α_gSW = 0.1     # short wave glass absortivity

σ = 5.67e-8     # W/m².K⁴ Stefan-Bolzmann constant

Fwg = 1 / 5     # view factor wall - glass

Tm = 20 + 273   # mean temp for radiative exchange

GLW1 = ε_wLW / (1 - ε_wLW) * S_c * 4 * σ * Tm**3
GLW2 = Fwg * S_c * 4 * σ * Tm**3
GLW3 = ε_gLW / (1 - ε_gLW) * S_c* 4 * σ * Tm**3
# long-wave exg. wall-glass
GLW = 1 / (1 / GLW1 + 1 / GLW2 + 1 / GLW3)

Gv = Va_dot * d_air * c_air

g_c =  cond_c/ e_c * S_c
g_pla =  cond_pla/ e_pla * S_pla
g_i =  cond_i/ e_i * S_i
g_v =  cond_v/ e_v * S_v
g_bois =  cond_bois/ e_porte * S_porte



A = np.zeros([14, 12])
A[0, 0] = 1
A[1, 0], A[1, 1] = -1, 1
A[2, 1], A[2, 2] = -1, 1
A[3, 2], A[3, 3] = -1, 1
A[4, 3], A[4, 4] = -1, 1
A[5, 4], A[5, 5] = -1, 1
A[6, 5], A[6, 6] = -1, 1
A[7, 6], A[7, 7] = -1, 1
A[8, 7], A[8, 8] = 1, -1
A[9, 8] = 1
A[10, 7] = 1
A[10, 9], A[11, 9] = -1, 1
A[13, 7], A[11, 10] = 1, -1
A[12, 10], A[12, 11] = 1, -1
A[13, 11] =- 1

np.set_printoptions(suppress=False)
#print(A)


C=np.zeros([12])
C[1]=d_c*c_c*e_c*S_c
C[3]=d_i*c_i*e_i*S_i
C[5]=d_pla*c_pla*e_pla*S_pla
C[8]=d_v*c_v*e_v*S_v
C[10]=d_c*c_c*e_c*S_c
Cdiag=np.diag(C)

#print(Cdiag)


G=np.zeros([14])

G[1]=25*S_c
G[0]=g_c/2
G[2]=g_c/2
G[3]=g_i/2
G[4]=g_i/2
G[5]=g_pla/2
G[6]=g_pla/2

G[8]=g_v
G[9]=25*S_v
G[10]=g_c/2
G[11]=g_c/2
G[12]=8*(2.02*2.5+1.85*2.50)
G[13]=g_bois
G[7]=GLW+Gv
Gdiag=np.diag(G)


#print(Gdiag)



b = np.zeros(14)
b[[0,9,11]] = 10+10 + np.array([20, 100, 110])

f = np.zeros(12)
f[[0, 6, 7,8]] = 10 + np.array([0, 80, 100, 110])


y=np.ones(12)

u = np.hstack([b[np.nonzero(b)], f[np.nonzero(f)]])

#print(u)
[As, Bs, Cs, Ds] = dm4bem.tc2ss(A, Gdiag, b, Cdiag, f, y)

#print(As, Bs, Cs, Ds)


yss = (-Cs @ np.linalg.inv(As) @ Bs + Ds) @ u
ytc = np.linalg.inv(A.T @ Gdiag @ A) @ (A.T @ Gdiag @ b + f)

print(np.array_str(yss, precision=3, suppress_small=True))
print(np.array_str(ytc, precision=3, suppress_small=True))
print(f'Max error in steady-state between thermal circuit and state-space:\
 {max(abs(yss - ytc)):.2e}')

#DYNAMIC

b = np.zeros(14)
b[[0,9,11]] = 1

f = np.zeros(12)
f[[0, 6, 7,8]] = 1

y = np.zeros(12)
y[[7]] = 1




[As, Bs, Cs, Ds] = dm4bem.tc2ss(A, Gdiag, b, Cdiag, f, y)


dtmax = min(-2. / np.linalg.eig(As)[0])
print(f'Maximum time step: {dtmax:.2f} s')


# dt = 5
dt = 360
duration = 3600 * 24 * 2        # [s]
n = int(np.floor(duration / dt))
t = np.arange(0, n * dt, dt)    # time

# Vectors of state and input (in time)
n_tC = As.shape[0]              # no of state variables (temps with capacity)
# u = [To To To Tsp Phio Phii Qaux Phia]
u = np.zeros([7, n])
u[0:2, :] = np.ones([2, n])


temp_exp = np.zeros([n_tC, t.shape[0]])
temp_imp = np.zeros([n_tC, t.shape[0]])


I = np.eye(n_tC)
for k in range(n - 1):
    temp_exp[:, k + 1] = (I + dt * As) @\
        temp_exp[:, k] + dt * Bs @ u[:, k]
    temp_imp[:, k + 1] = np.linalg.inv(I - dt * As) @\
        (temp_imp[:, k] + dt * Bs @ u[:, k])
        

y_exp = Cs @ temp_exp + Ds @  u
y_imp = Cs @ temp_imp + Ds @  u

fig, ax = plt.subplots()
ax.plot(t / 3600, y_exp.T, t / 3600, y_imp.T)
ax.set(xlabel='Time [h]',
       ylabel='$T_i$ [°C]',
       title='Step input: To = 1°C')
plt.show()

b = np.zeros(14)
b[[0,9,11]] = 1
f = np.zeros(12)
ytc = np.linalg.inv(A.T @ Gdiag @ A) @ (A.T @ Gdiag @ b + f)
print('Steady-state indoor temperature obtained with:')
print(f'- DAE model: {ytc[6]:.4f} °C')
print(f'- response to step input:{float(y_exp[:, -2]):.4f} °C')


filename = 'FRA_Lyon.074810_IWEC.epw'
start_date = '2000-01-01 12:00:00'
end_date = '2000-01-30 18:00:00'

# Read weather data from Energyplus .epw file
[data, meta] = dm4bem.read_epw(filename, coerce_year=None)
weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
del data
weather.index = weather.index.map(lambda t: t.replace(year=2000))
weather = weather[(weather.index >= start_date) & (
    weather.index < end_date)]


surface_orientation = {'slope': 90,
                       'azimuth': 0,
                       'latitude': 45}
albedo = 0.2
rad_surf1 = dm4bem.sol_rad_tilt_surf(weather, surface_orientation, albedo)
rad_surf1['Φt1'] = rad_surf1.sum(axis=1)

data = pd.concat([weather['temp_air'], rad_surf1['Φt1']], axis=1)
data = data.resample(str(dt) + 'S').interpolate(method='linear')
data = data.rename(columns={'temp_air': 'To'})


data['Ti'] = 20 * np.ones(data.shape[0])
data['Qa'] = 0 * np.ones(data.shape[0])

t = dt * np.arange(data.shape[0])


u = pd.concat([data['To'], data['To'], data['Ti'],
               α_wSW * S_c * data['Φt1'],
               τ_gSW * α_wSW * S_v * data['Φt1'],
               data['Qa'],
               α_gSW * S_v * data['Φt1']], axis=1)

temp_exp = 20 * np.ones([As.shape[0], u.shape[0]])

for k in range(u.shape[0] - 1):
    temp_exp[:, k + 1] = (I + dt * As) @ temp_exp[:, k]\
        + dt * Bs @ u.iloc[k, :]
        
y_exp = Cs @ temp_exp + Ds @ u.to_numpy().T
q_HVAC = Kp * (data['Ti'] - y_exp[0, :])

fig, axs = plt.subplots(2, 1)
# plot indoor and outdoor temperature
axs[0].plot(t / 3600, y_exp[0, :], label='$T_{indoor}$')
axs[0].plot(t / 3600, data['To'], label='$T_{outdoor}$')
axs[0].set(xlabel='Time [h]',
           ylabel='Temperatures [°C]',
           title='Simulation for weather')
axs[0].legend(loc='upper right')

# plot total solar radiation and HVAC heat flow
axs[1].plot(t / 3600,  q_HVAC, label='$q_{HVAC}$')
axs[1].plot(t / 3600, data['Φt1'], label='$Φ_{total}$')
axs[1].set(xlabel='Time [h]',
           ylabel='Heat flows [W]')
axs[1].legend(loc='upper right')

fig.tight_layout()







