#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:30:54 2021

@author: richard
Solve the differential equation for magnetisation precession.

Given by: dm/dt = -g*(np.cross(m, B))-(g*alpha/Ms)*(np.cross(m0, np.cross(m0, B)))
"""
# import streamlit as st
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import streamlit as st


plt.close('all')

g = 2 # gyromagnetic ratio
alpha = 0.16 # damping constant
t0 = 0
t=[t0]

fig = plt.figure(figsize=(14,7), tight_layout=True)
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_axis_off()


ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(324)
ax4 = fig.add_subplot(326)

xx, yy = np.meshgrid(range(-6, 6), range(-6, 6))


with st.sidebar:
    start = st.button('Start simulation')
    tab1, tab2 = st.tabs(["External field", "Magnetisation"])
    with tab1:
        st.markdown('#### Magnetic field direction:')
        Bx = st.slider('Bx', min_value=0.0,max_value=1.0, value=0.0, format='')
        By = st.slider('By', min_value=0.0,max_value=1.0, value=0.0, format='')
        Bz = st.slider('Bz', min_value=0.0,max_value=1.0, value=1.0, format='')
        st.markdown('#### Magnetic field amplitude:')
        Bs = st.slider('', min_value=1.0,max_value=5.0, value=5.0, step=0.5)
      

        
    with tab2:
        st.markdown('#### Magnetisation direction',)
        mx0 = st.slider('Mx', min_value=0.0,max_value=1.0, value=1.0, format='')
        my0 = st.slider('My', min_value=0.0,max_value=1.0, value=0.0, format='')
        mz0 = st.slider('Mz', min_value=0.0,max_value=1.0, value=0.0, format='')      
 #   with col3:
        st.markdown('#### Magnetisation amplitude:')
        Ms = st.slider('', min_value=1.0,max_value=5.0, value=3.0, step=0.5)
        st.markdown('#### Magnetic damping:')
        alpha = st.slider('', min_value=0.02,max_value=0.2, value=0.08)


B = np.array([Bx, By, Bz])
B = Bs*(B/np.linalg.norm(B))

m0 = np.array([mx0, my0, mz0])
m0 = Ms*(m0/np.linalg.norm(m0))

empty = st.empty()
with empty:
    ax1.cla()
    ax1.plot_surface(xx, yy, xx*0, alpha=0.1)
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-5, 5])
    ax1.set_zlim([-5, 5])
    ax1.quiver(0, 0, 0, B[0], B[1], B[2], length=1, color ='r', linewidth=2.5, label='B')
    ax1.quiver(0, 0, 0, m0[0], m0[1], m0[2], length=1, color='b', linewidth=2.5, label='M')
    # Hide grid lines
    ax1.grid(False)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    plt.pause(1e-6)
    ax1.legend()
    st.write(fig)


delta_t = 1e-3 #simulation step size (not real time)
n = 6000 # total simulation steps

mx = [m0[0]]
my = [m0[1]]
mz = [m0[2]]

if start:
    plt.close('all')
    with empty:
        for j in range(n+1):
            # Euler step method:
            dm_p = -g*(np.cross(m0, B)) # the change in m due to precession
            dm_d = -(g*alpha/Ms)*(np.cross(m0, np.cross(m0, B)))# the change in m due to damping
            dm = dm_p + dm_d # the total change in m
            m1 = (m0 + (delta_t*dm))
            # m1 = m1/np.linalg.norm(m1)
            t1 = t0 + delta_t
            
            mx.append(m1[0])
            my.append(m1[1])
            mz.append(m1[2])
            t.append(t1)
            #print('Time = {0}, Moment = {1}'.format(t1, m1))
            t0 = t1
            m0 = m1
            if j%250 == 0:
                #print(np.linalg.norm(m1))
                ax1.cla()
                ax1.plot_surface(xx, yy, xx*0, alpha=0.1)
                ax1.set_title(
                    'B = {0}, Ms = {1}, Damping = {2}, \n Simulation step = {3}'.format(
                        B, Ms, alpha, j))
                ax1.plot(mx, my, mz)
                ax1.quiver(0, 0, 0, B[0], B[1], B[2], length=1, color ='r', linewidth=2.5,
                           label='B')
                ax1.quiver(0, 0, 0, m1[0], m1[1], m1[2], length=1, color='b', linewidth=2.5,
                           label='M')
                # Hide grid lines
                ax1.grid(False)
                ax1.legend()

                # Hide axes ticks
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.set_zticks([])
                
                ax2.cla()
                ax2.plot(t,mx)
                ax3.cla()
                ax3.plot(t,my)
                ax4.cla()
                ax4.plot(t,mz)
                
                ax2.set_ylabel('Mx')
                ax3.set_ylabel('My')
                ax4.set_ylabel('Mz')
                ax4.set_xlabel('time step')
                
                ax1.set_xlim([-Bs, Bs])
                ax1.set_ylim([-Bs, Bs])
                ax1.set_zlim([-Bs, Bs])
                
                plt.pause(1e-6)
                st.write(fig)
print(mx) 
            