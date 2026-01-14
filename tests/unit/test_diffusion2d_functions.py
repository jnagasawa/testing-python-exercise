"""
Tests for functions in class SolveDiffusion2D
"""

import pytest
import numpy as np
from diffusion2d import SolveDiffusion2D




def test_initialize_domain():
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    w = 20.0
    h = 10.0
    dx = 0.2
    dy = 0.2
    
    expected_nx = 100
    expected_ny = 50
    
    solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)
    
    assert solver.nx == expected_nx
    assert solver.ny == expected_ny


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_physical_parameters
    """
    solver = SolveDiffusion2D()
    solver.dx = 0.1
    solver.dy = 0.1
    
    d = 4.0
    T_cold = 300.0
    T_hot = 700.0
    
    # expected_dt = 0.1^2 * 0.1^2 / (2 * 4.0 * (0.1^2 + 0.1^2))
    # expected_dt = 0.01 * 0.01 / (8.0 * (0.01 + 0.01))
    # expected_dt = 0.0001 / (8.0 * 0.02) = 0.0001 / 0.16 = 0.000625
    expected_dt = 0.000625
    
    solver.initialize_physical_parameters(d=d, T_cold=T_cold, T_hot=T_hot)
    
    assert solver.dt == pytest.approx(expected_dt)



def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.set_initial_condition
    """
    solver = SolveDiffusion2D()
    solver.nx = 10
    solver.ny = 10
    solver.dx = 1.0
    solver.dy = 1.0
    solver.T_cold = 300.0
    solver.T_hot = 700.0
    
    u = solver.set_initial_condition()

    
    # Manual check for a few points
    # r=2, cx=5, cy=5. r2=4.
    # p2 = (i*dx - 5)^2 + (j*dy - 5)^2
    
    # Point (5,5) should be hot
    assert u[5, 5] == 700.0
    # Point (3,5) should be cold ( (3-5)^2 + (5-5)^2 = 4, which is not < 4)
    assert u[3, 5] == 300.0
    # Point (4,4) should be hot ( (4-5)^2 + (4-5)^2 = 2 < 4)
    assert u[4, 4] == 700.0
    
    # Check shape
    assert u.shape == (10, 10)

