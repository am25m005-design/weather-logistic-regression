# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:58:00 2026

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ---------------- DATA ----------------
y = np.array([40,35,30,25,20,18,16,14,12,10,8,6,4,2,0,-2,-4,-6,-8,-10,
              -12,-14,-16,-18,-20,-25,-30,-35,-40])

pe = np.array([504,500,506,507,508,503,500,483,424,327,248,132,105,48,40,35,
               62,85,225,260,385,437,486,506,503,508,510,513,516])

# ---------------- CONSTANTS ----------------
q_inf = 450      # p0 - pc (Pa)
d = 12.5         # cylinder diameter (mm)

# ---------------- CALCULATIONS ----------------
y_by_d = y / d

# velocity ratio (correct formula)
u_by_U = np.sqrt(pe / q_inf)

# momentum deficit term
momentum = u_by_U * (1 - u_by_U)

# absolute value for drag calculation
momentum_abs = np.abs(momentum)

# sort data
idx = np.argsort(y_by_d)
x = y_by_d[idx]
z = momentum[idx]
z_abs = momentum_abs[idx]

# ---------------- FULL SPREAD PLOT ----------------
x_smooth = np.linspace(x.min(), x.max(), 400)
z_smooth = make_interp_spline(x, z, k=3)(x_smooth)

plt.figure(figsize=(7,5))
plt.plot(x_smooth, z_smooth, linewidth=2, label="Experimental curve")
plt.scatter(x, z, s=35)

plt.axhline(0, linewidth=0.8)
plt.xlabel(r"$y/d$")
plt.ylabel(r"$\frac{u}{U}\left(1-\frac{u}{U}\right)$")
plt.title("Full Spread of Wake Traverse Results (Circular Cylinder)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig("fig7_full.png", dpi=300)
plt.show()

# ---------------- SELECTED REGION ----------------
mask = (y_by_d >= -2) & (y_by_d <= 2)

x2 = y_by_d[mask]
z2 = momentum_abs[mask]

# sort selected data
idx2 = np.argsort(x2)
x2 = x2[idx2]
z2 = z2[idx2]

# smooth curve
x2_smooth = np.linspace(x2.min(), x2.max(), 400)
z2_smooth = make_interp_spline(x2, z2, k=3)(x2_smooth)

# area calculation (drag coefficient)
area = np.trapezoid(z2, x2)

plt.figure(figsize=(7,5))
plt.plot(x2_smooth, z2_smooth, linewidth=2, label="Selected region")
plt.scatter(x2, z2, s=35)
plt.fill_between(x2_smooth, z2_smooth, alpha=0.25)

plt.axhline(0, linewidth=0.8)
plt.xlabel(r"$y/d$")
plt.ylabel(r"$\left|\frac{u}{U}\left(1-\frac{u}{U}\right)\right|$")
plt.title("Selected Wake Region for Drag Estimation (Circular Cylinder)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig("fig7_selected.png", dpi=300)
plt.show()

print("Drag Coefficient (Cd) ≈", round(area, 3))