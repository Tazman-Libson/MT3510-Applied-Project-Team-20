#!/usr/bin/env python
# coding: utf-8

# In[25]:


def f(x):
    return np.tanh(x)**2

n = 10
x_knots = np.linspace(0, 10, 100)
y_knots = f(x_knots)


evalpts = np.linspace(0, 10, 10)

interpolated_poly(n, x_knots[0], x_knots[1], x_knots[2], y_knots[0], y_knots[1], y_knots[2])




# In[ ]:


def interpolated_poly(n, newX, newY):
    A = np.vander(newX)
    B = np.linalg.solve(A, newY)
    
    poly = np.poly1d(B)
    
    return poly 


# In[61]:


import sympy as sym

x = sym.Symbol('x')

print(interpolated_poly(5, np.linspace(0, 7), f(np.linspace(0, 7))))


# In[58]:


# MT3510 group project

# piecewise polynomial function

# take xarray , yarray - data
# take degree polynomial
# new eval points

def interpolated_poly(n, newX, newY):
    A = np.vander(newX)
    B = np.linalg.solve(A, newY)
    
    poly = np.poly1d(B)
    
    return poly 

def PiecewisePoly(n, x_knots, y_knots, evalpts):
    
    poly = []
    
    X = x_knots
    Y = y_knots
    
    for i in range(2, len(X)):
        
        newX = [X[i-2], X[i-1], X[i]]
        newY = [Y[i-2], Y[i-1], Y[i]]
    
        
        poly = interpolated_poly(n, newX, newY)
        
    return poly
        
        
# for each set of 3 points: 

# work out polynomial:
    # take x array = x0, x1, x2 
    # np.vander(xarray)
    # solve : vander, y knots = coeffs
    # poly = coeff * x^n 
    # cut off at correct n 
    
# 
    
# return interpolated polynomial evaluated eval points


# In[59]:


def f(x):
    return np.tanh(x)**2

n = 10
x_knots = np.linspace(0, 10, 100)
y_knots = f(x_knots)


evalpts = np.linspace(0, 10, 100)

#interpolatedpoly(n, x_knots[0], x_knots[1], x_knots[2], y_knots[0], y_knots[1], y_knots[2])

PiecewisePoly(n, x_knots, y_knots, evalpts)

