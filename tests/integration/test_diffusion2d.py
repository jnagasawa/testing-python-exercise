"""
Tests for functionality checks in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D


import pytest
import numpy as np
from diffusion2d import SolveDiffusion2D


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain + initialize_physical_parameters
    """
    solver = SolveDiffusion2D()
    w = 10.0
    h = 10.0
    dx = 0.5
    dy = 0.5
    d = 2.0
    
    # expected_dt = (0.5^2 * 0.5^2) / (2 * 2.0 * (0.5^2 + 0.5^2))
    # expected_dt = 0.0625 / (4.0 * 0.5) = 0.03125
    expected_dt = 0.03125


    
    solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)
    solver.initialize_physical_parameters(d=d)
    
    assert solver.dt == pytest.approx(expected_dt)


def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.initialize_domain + set_initial_condition
    """
    solver = SolveDiffusion2D()
    w = 10.0
    h = 10.0
    dx = 0.5
    dy = 0.5
    T_cold = 300.0
    T_hot = 700.0
    
    solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)
    # We call initialize_physical_parameters only to set T_cold and T_hot
    solver.initialize_physical_parameters(T_cold=T_cold, T_hot=T_hot)
    
    u = solver.set_initial_condition()
    
    # Manually compute expected u for verification
    nx = int(w / dx)
    ny = int(h / dy)
    expected_u = T_cold * np.ones((nx, ny))
    r, cx, cy = 2, 5, 5
    r2 = r ** 2
    for i in range(nx):
        for j in range(ny):
            p2 = (i * dx - cx) ** 2 + (j * dy - cy) ** 2
            if p2 < r2:
                expected_u[i, j] = T_hot
    
    assert np.allclose(u, expected_u)

