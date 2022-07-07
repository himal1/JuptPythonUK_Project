#!/usr/bin/env python
# coding: utf-8

# # This program obtimizes the vaules of parameters used in the magnet systems
# 

# The idea is to run the simulation to obtain polarization for many configurations and compre the polarization.<br>
# The details of the simulation is explianed in simulation program.

# In[1]:


import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy.integrate as spi
from scipy.integrate import solve_ivp
import warnings


# In[2]:


#suppress warnings
warnings.filterwarnings('ignore')


# First define a function to obtain magnetic field

# In[3]:


def ABC_Coil_CalculateB(x0,y0,z0,B2,B4,B5,B6):
    z=z0
    x=x0#0.55
    y=y0#0.55
    #zrange=zmax-zmin
    #z=np.arange(zmin,zmax,zrange/2500)
    #quadrupole and shim
    z1=63.6;   dz1=0.53;    r1=0.75;                # center,roll-off,radius [cm]
    zs=z1-5.25*2.54; zt=z1-.25*2.54; dzs=3.0;       # start,end,roll-off [cm]
    B1= 7500; Bs=B1*0.30;                           # field [G]  (Bs=0: no shim)
    vg=22000;
    # small,large ring
    z2=z1+7.0                      
    r2=10;          
    #B2=35;
    z3=z2;                          
    r3=15;          
    B3=-B2*pow((r2/r3),3)
    # top,bot of MSR
    za=z2+74.4;     zb=za+40.4;
    # small,large costheta
    z4=z2+8.4;      dz4=3.2;        r4=10;          #B4=60;
    z6=z2+60;       dz6=3.2;        r6=30;          #B6=2.5;
    #more parameters
    z5=250;                         r5=1.9;         #B5=0.03;
    Bx5=z*0+B5;
    Qxy=B1/r1/2*(1-np.tanh((z-z1)/dz1)) + Bs/r1/2*(np.tanh((z-zs)/dzs)-np.tanh((z-zt)/dzs));
    dQxy=-B1/r1/2*pow((1/np.cosh((z-z1)/dz1)),2)/dz1 + Bs/r1/2*(pow((1/np.cosh((z-zs)/dzs)),2) - pow(1/np.cosh((z-zt)/dzs),2))/dzs
    Bz2=B2*pow(r2,3)/pow((pow(r2,2)+pow((z-z2),2)),(3/2));
    Bz3=B3*pow(r3,3)/pow((pow(r3,2)+pow((z-z3),2)),(3/2));
    dBz2=-3*Bz2*(z-z2)/(pow(r2,2)+pow((z-z2),2));         #dBz/dz
    dBz3=-3*Bz3*(z-z3)/(pow(r3,2)+pow((z-z3),2));         #dBz/dz
    
    Bx4=B4*pow(r4,5)/pow((pow((pow(r4,2)+pow((z-z4),2)),(3/2)) + pow((pow(r4,2)-pow(dz4,2)),(3/2))),(5/3));
    Bx6=B6*pow(r6,5)/pow((pow((pow(r6,2)+pow((z-z6),2)),(3/2)) + pow((pow(r6,2)-pow(dz6,2)),(3/2))),(5/3));
    dBx4=-5*Bx4/(pow((pow(r4,2)+pow((z-z4),2)),(3/2)) + pow((pow(r4,2)-pow(dz4,2)),(3/2)))*np.sqrt(pow(r4,2)+ pow((z-z4),2))*(z-z4);
    dBx6=-5*Bx6/(pow((pow(r6,2)+pow((z-z6),2)),(3/2)) + pow((pow(r6,2)-pow(dz6,2)),(3/2)))*np.sqrt(pow(r6,2)+ pow((z-z6),2))*(z-z6);
    Bx = Qxy*y - 1/2*(dBz2+dBz3)*x + Bx4 + Bx6 + B5;
    By = Qxy*x - 1/2*(dBz2+dBz3)*y;
    Bz = dQxy*x*y + (Bz2+Bz3) + (dBx4+dBx6)*x;
    
    B  = [Bx,By,Bz];
    Bt=np.linalg.norm(B,axis=0);
    
    return B


# Now defiene a function to obtain magnetic field in certain time

# In[4]:


def Btm(x0,v,t,B2,B4,B5,B6):
    xt=x0+(v*t)
    B1tt=ABC_Coil_CalculateB(xt[0],xt[1],xt[2],B2,B4,B5,B6);
    return B1tt


# In[5]:


def PolSim(z0,z1,B2,B4,B5,B6,nev):
    #B2=35;
    #B4=60;
    #B5=0.03;
    #B6=2.5;
    g=20378.9; v3=22000;
    #z0=62.5;
    r0=0.629;
    #z1=250;
    r1=5;
    #nev=10;
    div=(r1+r0)/(z1-z0);
    Pol=[];
    for n in range(1, nev):
        #variables
        rho=np.sqrt(random.random())*r0; 
        phi=random.random()*2*math.pi; # random position on entrance aperture
        x0=[ rho*np.cos(phi), rho*np.sin(phi), z0 ]; 
        B0=ABC_Coil_CalculateB(x0[0],x0[1],x0[2],B2,B4,B5,B6);
        #obtain norm of the magnetic field
        B0t=np.linalg.norm(B0,axis=0);
    
        r=99999; 
        while(r>r1):  # choose random direction within exit aperture
            ctheta=1-(1-np.cos(div))*random.random(); phi=random.random()*2*math.pi;
            v1=(v3*math.sqrt(1-pow(ctheta,2))*np.cos(phi), v3*math.sqrt(1-pow(ctheta,2))*np.sin(phi), v3*ctheta );
         
            x1=np.array(x0)
            v=np.array(v1)
            t1=(z1-z0)/v[2];
            x1=x0+(v*t1); 
            r=math.sqrt(x1[0]*x1[0] + x1[1]*x1[1]);
        
        B1  = ABC_Coil_CalculateB(x1[0],x1[1],x1[2],B2,B4,B5,B6);
        B1t = np.linalg.norm(B1,axis=0);
    
        #solve bloch equation using build-in function 
        sol = solve_ivp(lambda t, y: g*np.cross(y, Btm(x0,v,t,B2,B4,B5,B6)), [0,t1], B0/B0t, method="RK45", rtol = 1e-5)
    
        #obtain last elemet of the array from the ode solution
        Mx=sol.y[0][-1]
        My=sol.y[1][-1]
        Mz=sol.y[2][-1]
        #norm of output vector of the ode
        Mn=np.sqrt(Mx**2+My**2+Mz**2)
    
        #polarization is given by 
        pol=(B1[0]*Mx+B1[1]*My+B1[2]*Mz)/(Mn*B1t)
        Pol.append(pol)
    return Pol


# In[ ]:


nev=100;
Mean=[]; B2=[]; B4=[]; B5=[]; B6=[]; Sigma=[];

for i in range(30,40):
    for j in range (55,65):
        for k in range (0,10):
            for l in range (20,30):
                k1=float(k/100);
                l1=float(l/10);
                him=0
                P=PolSim(62.5,250,i,j,k1,l1,nev)
                print("Running B2="+str(i)+", B4="+str(j)+", B5="+str(k1)+", B6="+str(l1)+" with "+str(nev)+" events in a loop")
                Mean.append(np.mean(P));
                Sigma.append(np.std(P)/np.sqrt(nev));
                B2.append(i)
                B4.append(j)
                B5.append(k1)
                B6.append(l1)
                him+1
                #Output=np.insert([Output],[Out],axis=0)
data = {'Mean':Mean, 'Sigma':Sigma, 'B2':B2, 'B4':B4, 'B5':B5, 'B6':B6}
df_1=pd.DataFrame(data)
numpy_array = df_1.to_numpy()
np.savetxt("file.txt", numpy_array)


# In[ ]:


print(df_1)


# In[ ]:





# In[ ]:




