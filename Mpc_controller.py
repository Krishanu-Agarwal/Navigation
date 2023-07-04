import numpy as np

from scipy.optimize import minimize

class MPC_controller():
    def __init__(self):
        pass

    def controller():
        def objective(): #objective function
            pass

        def constraint_ineq(): #inequality contraints(non-negative {>=0})
            pass

        def constraint_eq(): #equality constraints (equals 0 {=0})
            pass
        
        u = np.zeros() #initial guess

        constraints = [{'type':'eq','fun':constraint_eq},{'type':'eq','fun':constraint_ineq}]

        return minimize(objective, u,method='SLSQP',constraints=constraints).x[0] #return first value of optimal control
        

