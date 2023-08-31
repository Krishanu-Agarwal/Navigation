import numpy as np

from scipy.optimize import minimize

class MPC_controller():
    def __init__(self,x0, x_final, M,P,delT,model, *args,**kwargs):
        self.x0 = x0
        self.x_final = x_final #goal position
        self.P = P #prediciton horizon
        self.model = model
        self.args = args
        self.kwargs = kwargs
        pass

    def controller():
        def objective(u,x_final,x0,h): #objective function
            return np.sum(np.abs(model(u,x0,h)-x_final))
            pass

        def constraint_ineq(): #inequality contraints(non-negative {>=0})
            pass

        def constraint_eq(): #equality constraints (equals 0 {=0})
            pass
        
        u = np.zeros() #initial guess

        constraints = [{'type':'eq','fun':constraint_eq},{'type':'eq','fun':constraint_ineq}]

        return minimize(objective, u,method='SLSQP',constraints=constraints).x[0] #return first value of optimal control
        

def model(u,x0,h):
    return x0 + h*(np.asarray([[0,1],[0,0]])@x0 +np.asarray([0,1])*u)
