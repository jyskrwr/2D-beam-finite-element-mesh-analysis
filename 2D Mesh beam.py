#!/usr/bin/env python
# coding: utf-8

# In[1]:


# DEPENDENCIES
import math #..............................Math functionality
import copy #..............................Create copies of objects in memory
import numpy as np #.......................Numpy for working with arrays
from glob import glob #....................Allows check that file exists before import
import matplotlib.colors #.................Colormap functionality
import ipywidgets as widgets #.............Widget control functionality
from numpy import genfromtxt #.............Importing structure data from csv
import matplotlib.pyplot as plt #..........Plotting functionality 
import matplotlib.patches as patch #.......For plotting 2D shape 
from scipy.interpolate import griddata #...Interpolation of stress/strain values to grid


# ### Automatic structure & loading data import
#     Import .csv data defining the structure and loading (generated in Blender)

# In[2]:


if glob('data/Vertices.csv'): 
    nodes = genfromtxt('data/Vertices.csv', delimiter=',') 
    print('1. 🟢 Vertices.csv imported')
else: 
    print('1. 🛑 STOP: Vertices.csv not found')

    
if glob('data/Elements.csv'): 
    elements = genfromtxt('data/Elements.csv', delimiter=',')
    elements = np.int_(elements) 
    print('2. 🟢 Elements.csv imported')
else: 
    print('2. 🛑 STOP: Elements.csv not found')
    


if glob('data/Restraint-Nodes.csv'):
    restrainNodes = genfromtxt('data/Restraint-Nodes.csv', delimiter=',')
    print('3. 🟢 Restraint-Nodes.csv imported')
else: 
    print('3. 🛑 STOP: Restraint-Nodes.csv not found')
    
if glob('data/Restraint-DoF.csv'): 
    restraintData = genfromtxt('data/Restraint-DoF.csv', delimiter=',')
    restraintData = np.int_(restraintData) #Convert members definitions from float to int
    flatData = restraintData.flatten() #Flatten DoF data
    restrainedDoF = flatData[np.nonzero(flatData)[0]].tolist() #Remove zeros from DoF data
    print('4. 🟢 Restraint-DoF.csv imported')
else: 
    print('4. 🛑 STOP: Restraint-DoF.csv not found')
    

if glob('data/Force-Data.csv'):
    forceLocationData = genfromtxt('data/Force-Data.csv', delimiter=',')
    forceLocationData = np.int_(forceLocationData)
    nforces = len(np.array(forceLocationData.shape))
    
    if nforces < 2:
        forceLocationData = np.array([forceLocationData])
    print('5. 🟢 Force-Data.csv imported')              
else:
    forceLocationData = []
    print('5. 🛑 Force-Data.csv not found')
        
                  
                                 
                                   



# ### Manual data entry and parameter setting
# 

# In[3]:


#=================================START OF MANUAL DATA ENTRY================================
#CONSTANTS
E = 200*10**9 #(N/m^2) Young's modulus
nu = 0.3 #Poisson's ratio

#PLANE STRESS
C = (E/(1-nu**2))*np.array([[1, nu, 0],[nu, 1, 0],[0, 0, (1-nu)/2]]) #Plane stress material matrix
t = 0.1 #(m) Element thickness (1 if plane strain problem)

#PLANE STRAIN
# C = ((E*(1-nu))/((1+nu)*(1-2*nu))) * np.array([[1, nu/(1-nu), 0],[nu/(1-nu), 1,0],[0,0, (1-2*nu)/(2*(1-nu))]])
# t = 1 #Always set to 1 for plane strain

#GAUSS SCHEME PARAMETERS
alpha = [1,1] #Weights
sp = [-0.5773502692,0.5773502692] #Sampling points

#ASSIGN POINT LOADS
P = -10000 #(N) Point load magnitude (and direction via sign)
pointLoadAxis = 'y' #The GLOBAL axis along which point loads are applied

#=================================END OF MANUAL DATA ENTRY================================


#  ### Plot the structure before proceeding

# In[4]:


fig = plt.figure() 
axes = fig.add_axes([0.1,0.1,2,1.5]) 
fig.gca().set_aspect('equal', adjustable='box')

for n, ele in enumerate(elements):
    
    #Identify node numbers for this element
    node_1 = ele[0] #Node number for node 'local 1' (top-right)
    node_2 = ele[1] #Node number for node 'local 2' (top-left)
    node_3 = ele[2] #Node number for node 'local 3' (bottom-left)
    node_4 = ele[3] #Node number for node 'local 4' (bottom-right)
    
    #Build an array of x and y coords for for this element    
    x = np.array([nodes[node_1-1,0], 
                  nodes[node_2-1,0],
                  nodes[node_3-1,0],
                  nodes[node_4-1,0]])
    
    y = np.array([nodes[node_1-1,1], 
                  nodes[node_2-1,1],
                  nodes[node_3-1,1],
                  nodes[node_4-1,1]])   
    
    #Plot 2D patch for each element
    axes.add_patch(patch.Polygon(xy=list(zip(x,y)), facecolor ='gray', alpha=0.2)) #Patch fill
    axes.add_patch(patch.Polygon(xy=list(zip(x,y)), fill=None, edgecolor='gray', alpha=1)) #Patch outline
    axes.plot(x,y,'go', markersize=1) #Nodes

maxX = nodes.max(0)[0]
maxY = nodes.max(0)[1]
minX = nodes.min(0)[0]
minY = nodes.min(0)[1]
x_margin = 0.2
y_margin = 0.2
axes.set_xlim([minX-x_margin,maxX+x_margin])
axes.set_ylim([minY-y_margin,maxY+y_margin])
axes.set_xlabel('X-coordinate (m)')
axes.set_ylabel('Y-coordinate (m)')
axes.set_title('Structure')
plt.show()


# ### Build the global force vector and add point forces

# In[5]:


forceVector = np.array([np.zeros(len(nodes)*2)]).T
if (len(forceLocationData) > 0):
    ForceNodes = forceLocationData[:, 0]
    xForceIndices = forceLocationData[:, 1]
    yForceIndices = forceLocationData[:, 2]
    
    if(pointLoadAxis == 'x'):
        forceVector[xForceIndices] = P
        
    elif(pointLoadAxis == 'y'):
        forceVector[yForceIndices] = P
    
    


# ### Define a function to calculate element stiffness matrix
# Function to build 8x8 element stiffness matrix

# In[6]:


def calculateKE(alpha, sp, x, y):
    KE = np.zeros([8, 8])
    
    for i, r in enumerate(sp):
        
        for j, s in enumerate(sp):
            
            dh1dr = 0.25*(1+s)
            dh2dr = -0.25*(1+s)
            dh3dr = -0.25*(1-s)
            dh4dr = 0.25*(1-s)
            
            dh1ds = 0.25*(1+r)
            dh2ds = 0.25*(1-r)
            dh3ds = -0.25*(1-r)
            dh4ds = -0.25*(1+r)
            
            dxdr = x[0]*dh1dr + x[1]*dh2dr + x[2]*dh3dr + x[3]*dh4dr
            dydr = y[0]*dh1dr + y[1]*dh2dr + y[2]*dh3dr + y[3]*dh4dr
            dxds = x[0]*dh1ds + x[1]*dh2ds + x[2]*dh3ds + x[3]*dh4ds
            dyds = y[0]*dh1ds + y[1]*dh2ds + y[2]*dh3ds + y[3]*dh4ds
            
            J = np.matrix([[dxdr, dydr], [dxds, dyds]])
            invj = J.I
            detJ = np.linalg.det(J)
            
            dh1 = np.array([[dh1dr, 0, dh2dr, 0, dh3dr, 0, dh4dr, 0], 
                           [dh1ds, 0, dh2ds, 0, dh3ds, 0, dh4ds, 0]])
            
            
            dh2 = np.array([[0, dh1dr,0, dh2dr, 0, dh3dr, 0, dh4dr], 
                           [0, dh1ds, 0, dh2ds, 0, dh3ds, 0, dh4ds]])
            
            B = np.matrix([[1, 0], [0, 0], [0, 1]])*invj*dh1 + np.matrix([[0, 0], [0, 1], [1, 0]])*invj*dh2
            
            KE = KE + alpha[i]*alpha[j]*t*B.T*C*B*detJ
            
    return KE
            


# ### Build the primary stiffness matrix, $Kp$

# In[7]:


nDoF = np.amax(elements)*2
Kp = np.zeros([nDoF, nDoF])

for n, ele in enumerate(elements):
    node_1 = ele[0]
    node_2 = ele[1]
    node_3 = ele[2]
    node_4 = ele[3]
    
    x = np.array([nodes[node_1 - 1, 0], 
                 nodes[node_2 - 1, 0], 
                 nodes[node_3 - 1, 0], 
                 nodes[node_4 - 1, 0]])
    
    y = np.array([nodes[node_1 - 1, 1], 
                 nodes[node_2 - 1, 1], 
                 nodes[node_3 - 1, 1], 
                 nodes[node_4 - 1, 1]])
    
    KE = calculateKE(alpha, sp, x, y)
    
    i1x = 2*node_1-2 #index for horizontal DoF for node 1 (top right)
    i1y = 2*node_1-1 #index for vertical DoF for node 1 (top right)
    i2x = 2*node_2-2 #index for horizontal DoF for node 2 (top left)
    i2y = 2*node_2-1 #index for vertical DoF for node 2 (top left)
    i3x = 2*node_3-2 #index for horizontal DoF for node 3 (top left)
    i3y = 2*node_3-1 #index for vertical DoF for node 3 (top left)
    i4x = 2*node_4-2 #index for horizontal DoF for node 4 (top left)
    i4y = 2*node_4-1 #index for vertical DoF for node 4 (top left)v
    
    indices = [i1x,i1y,i2x,i2y,i3x,i3y,i4x,i4y]
    Indexarray = np.ix_(indices, indices)
    Kp[Indexarray] =  Kp[Indexarray] + KE
    


# ### Extract the structure stiffness matrix, $Ks$

# In[8]:


restrainedIndex = [x-1 for x in restrainedDoF]
Ks = np.delete(Kp, restrainedIndex, 0)
Ks = np.delete(Ks, restrainedIndex, 1)
Ks = np.matrix(Ks)


# ### Solve for displacements

# In[9]:


forceVectorRed = copy.copy(forceVector)
forceVectorRed = np.delete(forceVectorRed, restrainedIndex, 0)
U = Ks.I*forceVectorRed


# ### Solve for reactions

# In[10]:


UG = np.zeros(nDoF)
c = 0
for i in np.arange(nDoF):
    if i in restrainedIndex:
        
        UG[i] = 0
    else:
        
        UG[i] = U[c]
        c = c + 1
        
UG = np.array([UG]).T
FG = np.matmul(Kp, UG)
        
        


# ### Plot deflected shape

# In[11]:


fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 2, 1.5])
fig.gca().set_aspect('equal', adjustable='box')

dFac = 2
x_margin = 0.2
y_margin = 0.2

for n, ele in enumerate(elements):
    
    node_1 = ele[0]
    node_2 = ele[1]
    node_3 = ele[2]
    node_4 = ele[3]

    x = np.array([nodes[node_1 - 1, 0],
                 nodes[node_2 - 1, 0], 
                 nodes[node_3 - 1, 0], 
                 nodes[node_4 - 1, 0]])
    
    y = np.array([nodes[node_1 - 1, 1],
                 nodes[node_2 - 1, 1], 
                 nodes[node_3 - 1, 1], 
                 nodes[node_4 - 1, 1]])
    
    axes.add_patch(patch.Polygon(xy = list(zip(x,y)), facecolor='gray', alpha=0.1))
    axes.add_patch(patch.Polygon(xy = list(zip(x,y)), fill = None , edgecolor='gray', alpha=1))
    i1x = 2*node_1-2 #index for horizontal DoF for node 1 (top right)
    i1y = 2*node_1-1 #index for vertical DoF for node 1 (top right)
    i2x = 2*node_2-2 #index for horizontal DoF for node 2 (top left)
    i2y = 2*node_2-1 #index for vertical DoF for node 2 (top left)
    i3x = 2*node_3-2 #index for horizontal DoF for node 3 (top left)
    i3y = 2*node_3-1 #index for vertical DoF for node 3 (top left)
    i4x = 2*node_4-2 #index for horizontal DoF for node 4 (top left)
    i4y = 2*node_4-1 #index for vertical DoF for node 4 (top left)v
    
    xd = np.array([nodes[node_1 - 1, 0] + UG[i1x,0]*dFac,
                 nodes[node_2 - 1, 0] + UG[i2x,0]*dFac, 
                 nodes[node_3 - 1, 0] + UG[i3x,0]*dFac, 
                 nodes[node_4 - 1, 0] + UG[i4x,0]*dFac])
    
    yd = np.array([nodes[node_1 - 1, 1] + UG[i1y,0]*dFac,
                 nodes[node_2 - 1, 1] + UG[i2y,0]*dFac, 
                 nodes[node_3 - 1, 1] + UG[i3y,0]*dFac, 
                 nodes[node_4 - 1, 1] + UG[i4y,0]*dFac])
    
    axes.add_patch(patch.Polygon(xy=list(zip(xd, yd)), facecolor='red', alpha=0.3))
    axes.add_patch(patch.Polygon(xy=list(zip(xd, yd)), fill=None, edgecolor='red',alpha=1))
    axes.plot(xd,yd, 'go', markersize=1)
    
    
maxX = nodes.max(0)[0]
maxY = nodes.max(0)[1]
minX = nodes.min(0)[0]
minY = nodes.min(0)[1]
axes.set_xlim([minX-x_margin,maxX+x_margin])
axes.set_ylim([minY-y_margin,maxY+y_margin])
axes.set_xlabel('X-coordinate (m)')
axes.set_ylabel('Y-coordinate (m)')
axes.set_title('Deflected shape')
plt.show()     


# In[ ]:




