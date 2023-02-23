#!/usr/bin/env python
# coding: utf-8

# # Interactivity code from notebook
# 
# Jupyter Notebooks are excellent ways of communicating code and results. One of the nice features that makes them stand out in this regard is how easy it is to embed interactive "widgets" into the notebook.
# 
# This is easiest seen by example.

# In[1]:


### EXAMPLE INTERACTIVITY

from ipywidgets import interact,interactive
import numpy as np

def f(x):
    print(f"x\u00b2 = {x**2}")
    return 

interact(f, x=(0.0, 10.0));


# The idea is to provide a function which produces an output for each value of the slider. Here's a more interesting example, where we plot a phase-plane for different values of a parameter $k$. This ODE arises from a population model for spruce budworm, outbreaks of which have caused major deforestation in North America.

# In[2]:


### EXAMPLE CODE

get_ipython().run_line_magic('matplotlib', 'inline')
# interactive plots work better "inline" rather than as "notebook" figures

import matplotlib.pyplot as plt
from ipywidgets import fixed


def vec_field(X, Y, r, k, c, a):
    U = np.ones((len(X), len(Y)))               # the flow of time
    V = r*(1 - Y/k) * Y - c * Y**2/(a + Y**2)   # the population change
    return U, V
    
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100) 
X, Y = np.meshgrid(x, y)

# interact will start with the 
def phase_plane(r = 0.3, k = 10, c = 1, a = 0.02):
    fig, ax = plt.subplots()
    U, V = vec_field(X, Y, r, k, c, a)
    ax.streamplot(X, Y, U, V)
    plt.title(f'Spruce budworm population with r={r}, k={k}')
    plt.xlabel('time')
    plt.ylabel('population')
    plt.show()

# Investigate "continuous_update" in the documentation linked below
# to avoid the plots lagging as the slider is dragged!
interactive(phase_plane,
            r = (0.2, 1.0),
            k = (8.0, 12.0),
            c = fixed(1),     # used "fixed" so there are no sliders for c or a
            a = fixed(0.02))


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

# In[8]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from ipywidgets import fixed


# In[9]:


def practise_plot(d = 3,e=2):
    
    g = lambda X: d*X**d*2+e
    
    N = 101
    x = np.linspace(1,2,N)  
    plt.plot(x,g(x))
    plt.title('model title')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

interactive(practise_plot,
            d = (3, 10), e=fixed(2))


# In[10]:


# defining function 

f = lambda X: np.e**X*np.cos(10*X)


# In[11]:


M = 11
x0 = np.linspace(1,2,M)
y0 = f(x0)

plt.plot(x0,y0, label = "knots")



# We have only shown a small number of the interactivity features in Jupyter Notebooks. More details on the `interact` function can be found [here](https://ipywidgets.readthedocs.io/en/stable/examples/Using%20Interact.html). Note that `ipywidgets` only work in Jupyter Notebooks, and not in other Python environments. More complex interactivity is possible using the other functions from the `ipywidgets` module, which we leave you to investigate for yourself if interested.
# 
# Similar interactive plots can be produced outside Jupyter Notebooks; the easiest way is through `matplotlib` widgets.
