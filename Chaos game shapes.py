
# coding: utf-8

# In[1]:

#This script steps through chaos game with n vertices to generate fractal images
#Will Cheney May 2019

import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:

#Defines (x,y) vertices coordinates
# sides corresponds to number of points e.g. square = 4
def calculate_vertices(sides, center = (0,0)):
    i = 1
    vertices = []
    while (i * 360/sides) <= 360:
        vertices.append((round((center[0] + math.sin(math.radians(i * 360/sides))),4), round((center[1] + math.cos(math.radians(i * 360/sides))),4)))
        i += 1

    return vertices


# In[4]:

#Runs Chaos game with specified number of vertices and steps
#returns list of points from each iteration
#Restrict determines whether the same vertex can be chosen twice sequentially. A value of True prevents the same vertex being chosen in the next step
def generate_cgr(vertices, steps = 10000, restrict = False):
    vertices = calculate_vertices(vertices)
    points = [(0,0)]
    interval = 1/len(vertices)
    last_vertex = -1
    
    for i in range(1,steps):
        randomnumber = np.random.rand()
        
        
           
        for threshold in range(1,(len(vertices) + 1 )):           


            if randomnumber >= (1 - (interval*threshold)):
                if restrict == True:
                 
                    if threshold == last_vertex:
                        break
                points.append((((points[-1][0] + vertices[(threshold - 1)][0])/2), ((points[-1][1] + vertices[(threshold - 1)][1])/2)))
                last_vertex = threshold    
                break
   
    return points


# In[26]:

#scatter points generated to create image
points = generate_cgr(5,35000,restrict = True)
plt.scatter(*zip(*points), color = '#192a56', s = 0.1)
fig = plt.gcf()
plt.show()


# In[27]:

#fig.savefig('title',dpi = 300, bbox_inches='tight')


# In[200]:




# In[230]:




# In[ ]:



