import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from sympy import pprint, Symbol, Function, Derivative, dsolve, solve, diff
import sympy as sp
from sympy.interactive import printing
printing.init_printing(use_latex = True)

f = Function('f');
x = Symbol('x');

# ODE model 1
def model1(y, x):
    dydx = 1 + x -y;
    return dydx;

model = f(x).diff(x) + 1 + x - f(x);
y_general = dsolve(model, f(x));
print('The general solution: ')
pprint(y_general);

y_special = dsolve(model,f(x), ics={f(0):0});
print('The special solution: ')
pprint(y_special);

# solution = solve(y_special.subs([(f(x), 1)]), x)
# print(solution)

# model 1
plt.figure(1)
# Direction field parameters
X_start = -15;
X_end = 15;
N_X_points = 30; # number of points in X direction
Y_start = -15;
Y_end = 15;
N_Y_points = N_X_points; # number of points in Y direction

X, Y = np.meshgrid(np.linspace(X_start, X_end, N_X_points), np.linspace(Y_start, Y_end, N_Y_points));
delta_X = np.ones((N_X_points, N_Y_points)); # direction vector x component
delta_Y = model1(Y, X); # direction vector y component
vector_normalization = np.sqrt(np.power(delta_X, 2) + np.power(delta_Y, 2));
delta_X = delta_X / vector_normalization; # normalized direction vector x component
delta_Y = delta_Y / vector_normalization; # normalized direction vector y component

# Plot direction field
plt.quiver(X, Y, delta_X, delta_Y, angles="xy");#, scale_units='xy', scale = 1);

# Plot integral curves
x_start = -8;
x_end = 12;
y_start = -15;
y_end = 15;
N_int_curves = 15; # number of solutions to plot
x = np.linspace(x_start, x_end, N_X_points * 100); # evaluate solution y(x) at x
y0_vector = np.linspace(y_start, y_end, N_int_curves); # vector of initial solutions
for solution_no in range(N_int_curves):
    y0 = y0_vector[solution_no]; # initial condition
    y = odeint(model1, y0, x, args=());
    plt.plot(x, y)

plt.xlim([X_start, X_end])
plt.ylim([Y_start, Y_end])
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid()
plt.show()

