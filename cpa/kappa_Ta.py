# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:17:01 2024

@author: jlu
"""
import numpy as np
import sympy as sp
from scipy.optimize import fsolve
from sympy import symbols, nsolve

def Ta_to_kappa(Ta, flag_overcoupled = True):
    """
    Transform the Ta to kappa, depends on the flag, the return is different
    (without global loss)

    """
    a=np.array(Ta)
    b=2*np.array(Ta)-4
    c=np.array(Ta)
    kappa_all=[(-b+np.sqrt(b**2-4*a*c))/2/a, (-b-np.sqrt(b**2-4*a*c))/2/a]
    if flag_overcoupled:
        return kappa_all[0]
    else:
        return kappa_all[1]
    
def kappa_to_Ta(kappa):
    """
    Transform the kappa to Ta (without global loss)

    """
    kappa=np.array(kappa)
    return 4*kappa/((1+kappa)**2)

def g(N, M, Gamma):
    m = M/N
    lambda0 = N/np.pi
    return -Gamma/4/lambda0+np.sqrt((1-m)+Gamma**2/16/(lambda0**2))

def solve_kappa_app(N, M, Gamma, Ta, flag_over=True):
    """
    When the system with global loss, the kappa and Ta do not follow standard relation

    """
    
    kappa_eff = symbols('kappa_eff')
    
    g0 = g(N, M, Gamma)
    
    saa = (1 - g0 * kappa_eff) / (1 + g0 * kappa_eff)
    Ta_equation = 1 - saa**2 - Ta
    if Gamma == 37.26:
        if flag_over:
            if N <= 20:  
                initial_guess = 10  
            elif N <= 100:  
                initial_guess = 4  
            else:
                initial_guess = 2
        else:
            initial_guess = 0.5
    elif Gamma == 20:
        if flag_over:
            if N <= 15:  
                initial_guess = 10  
            elif N <= 100:  
                initial_guess = 4  
            else:
                initial_guess = 2
        else:
            initial_guess = 0.5
            
    elif Gamma == 0:
        if flag_over:
            initial_guess = 1.5
        else:
            initial_guess = 0.7
        
    kappa_eff_value = nsolve(Ta_equation, kappa_eff, initial_guess) 
    return float(kappa_eff_value)


def solve_Ta_app(N, M, Gamma, kappa):
    """
    When the system with global loss, the kappa and Ta do not follow standard relation

    """
    g0 = g(N, M, Gamma)
    saa = (1 - g0 * kappa) / (1 + g0 * kappa)
    Ta = 1 - saa**2 
    return Ta

def solve_kappa_cubic(N, M, d_value, T_a_value, flag_over=True):
    """
    Use the cubic equation to solve the kappa, exact solution

    """
    m_value = M/N
    lambda_value = N/np.pi
    g, kappa_a = sp.symbols('g kappa_a', real=True, positive=True)

    eq1 = sp.Eq(g - 1/g + m_value * kappa_a / (1 + kappa_a * g) + d_value / (2 * lambda_value), 0)

    S_aa = (1 - kappa_a * g) / (1 + kappa_a * g)
    Ta_eq = sp.Eq(T_a_value, 1 - S_aa**2)

    solutions = sp.solve((eq1, Ta_eq), (g, kappa_a))
    
    kappa_a_solutions = [float(sol[1]) for sol in solutions]
    if flag_over:
        return np.sort(kappa_a_solutions)[1]
    else:
        return np.sort(kappa_a_solutions)[0]

def solve_Ta_cubic(N, M, d_value, kappa_a_value):
    """
    Use the cubic equation to solve the Ta, exact solution

    """
    m_value = M / N
    lambda_value = N / np.pi

    def g_equation(g):
        return g - 1 / g + m_value * kappa_a_value / (1 + kappa_a_value * g) + d_value / (2 * lambda_value)

    initial_guesses = np.linspace(0.1, 10, 10) 
    g_solution = None

    for guess in initial_guesses:
        try:
            g_solution = fsolve(g_equation, guess)
            if abs(g_equation(g_solution[0])) < 1e-6:  
                break
        except (RuntimeError, ValueError):
            continue

    if g_solution is None:
        raise ValueError("no solution found")

    g_solution = g_solution[0]
    
    S_aa = (1 - kappa_a_value * g_solution) / (1 + kappa_a_value * g_solution)
    T_a = 1 - S_aa**2

    return T_a

def solve_Ta_cubic_g(N, M, d_value, kappa_a_value):
    """
    Use the cubic equation to solve the Ta, exact solution

    """
    m_value = M / N
    lambda_value = N / np.pi

    def g_equation(g):
        return g - 1 / g + m_value * kappa_a_value / (1 + kappa_a_value * g) + d_value / (2 * lambda_value)

    initial_guesses = np.linspace(0.1, 10, 10) 
    g_solution = None

    for guess in initial_guesses:
        try:
            g_solution = fsolve(g_equation, guess)
            if abs(g_equation(g_solution[0])) < 1e-6:  
                break
        except (RuntimeError, ValueError):
            continue

    if g_solution is None:
        raise ValueError("no solution found")

    g_solution = g_solution[0]

    return g_solution