#!/usr/bin/env python
# coding: utf-8

# Part 3
# 
# Create an interactive plot which allows the user to vary the degree of a Lagrange interpolating polynomial (not piecewise) for a certain set of knots and evaluation points. The plot should show the interpolating function and the knots. For demonstration purposes use the function from part 1, i.e. 
# f
# (
# x
# )
# =
# e
# x
# cos
# (
# 10
# x
# )
# 
#  but now plot the interval 
# 
# x
# ∈
# [
# 1
# ,
# 2
# ]
# ,
#  choose a modest number of knots and centre your interpolant (i.e. as the degree increases and more knots are required work from the centre out, as seen in the video).

# In[31]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from ipywidgets import interact,interactive
from ipywidgets import fixed


# In[32]:


# defining function and knots

f = lambda X: np.e**X*np.cos(10*X)

M = 11
x0 = np.linspace(1,2,M)
y0 = f(x0)

N = 101
x = np.linspace(1,2,N) 


# In[33]:


### Less efficient code than your code - once changed over can delete

mid = (max(x0)+min(x0))/2

diff_from_centre = np.abs(x0 - mid)
order = np.argsort(diff_from_centre)           # this gives index of positions from centre of x0 to extremes


L=[]
for i in range(8):
    L.append(x0[order[i]])
L
xorder = np.array(L)

yorder=f(xorder)


# In[34]:


def make_poly(x,d):
    list = []
    for i in range(d):
        list.append(x0[order[i]])
    xorder = np.array(list)
    yorder = f(xorder)
    
    
    A = np.vander(xorder)                  # construct the Vandermode matrix
    coeff = np.linalg.solve(A,yorder)      # the first term is the coefficient of the highest order
    
    J = len(xorder)
    
    
    pows = (J-1-np.arange(J)).reshape(J,1)         # these are the exponents required
    xnew = np.reshape(x,(1,N))                     # reshape for the broadcast
    y = np.sum((xnew**pows)*coeff.reshape(J,1),axis=0)
    
    return y


# In[35]:


def plot_poly(x0,y0,x,d):              
    # x0, y0 are the knots; x is the continuous domain; d is the degree of the polynomial
    plt.plot(x0,y0,label = 'data')
    plt.plot(x,make_poly(x,d+1),label = 'poly interpolated data')
    plt.legend()


# In[36]:


interactive(plot_poly,
            x0 = fixed(x0), y0 = fixed(y0),x = fixed(x), d = (1, 10))


# In[ ]:




