
import matplotlib.pyplot as plt
import numpy as np
import polytope as pc

from scipy.optimize import minimize
from scipy.linalg import block_diag

class Zonotope:
    def __init__(self, center, generator):
        """
        Creates a Zonotope with given Centers and Generators
        """
        self.center =np.asarray(center)
        self.generator = np.asarray(generator)
        try:
            self.p = pc.Polytope(self.A,self.b) #creates a Polytope representation of the Zonotope
        except:
            print("Zonotope exceeds dimension 2, some methods may not be useable")

    def plot(self): 
        """
        plots the Zonotpes using the H-repsentation
        """
        fig,ax = plt.subplots()
        ax.set_aspect('equal','box')

        ax.plot(self.center[0], self.center[1],'ro',label='Center')

        a=0

        for generator in self.generator:
            if a==0:
                ax.quiver(self.center[0], self.center[1], generator[0],generator[1],angles='xy', scale_units='xy', scale=1, color='b',label='Generator')
                a=1
            else:
                ax.quiver(self.center[0], self.center[1], generator[0],generator[1],angles='xy', scale_units='xy', scale=1, color='b')
        
        vertex = pc.extreme(self.p)
        vertex = np.vstack([vertex,vertex[0,:]])
        ax.plot(vertex[:,0], vertex[:,1], 'r-', label='Zonotope Boundary')
        # print(vertex)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True)
        plt.show()

    def __add__(self,other):
        """
        Returns a Zonotpe Z = Z1 +Z2. 
        Z.center  = Z1.center + Z2.center
        Z.Generator = [Z1.Generator,Z2.Generator]
        Overloaded the Plus '+' operator to add 2 Zonotopes
        @param other: Zonotope
        @rtype: Zonotope
        """
        if isinstance(other, Zonotope):
            center = self.center+other.center
            generator = np.concatenate((self.generator, other.generator), axis=0)
            return Zonotope(center,generator)
        else:
            raise TypeError("Unsupported operand type. Can add Zonotopes only.")
    
    def __mul__(self,other):
        """
        Returns a Zonotope Z = Z1*Z2
        Z.Center = [Z1.Center,Z2.Center]
        Z.Generator = Diag(Z1.Generator,Z2.Generator)
        Overloaded the Multiplication '*' operator to 
        @param other: Zonotope
        @rtype: Zonotope
        """
        if isinstance(other, Zonotope):
            center = np.concatenate([self.center,other.center])
            generator = block_diag(self.generator,other.generator)
            return Zonotope(center,generator)
        else:
            raise TypeError("Unsupported operand type. Can add Zonotopes only.")
    
    def __contains__(self, point):
        """
        Return 'True' if 'self' contains 'point'.
        Boundary points are included.
        @param point: column vector, e.g., as 'numpy.ndarray'
        @rtype: bool
        For multiple points, see the method 'self.contains'.
        """
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        test = self.A.dot(point.flatten()) - self.b < 10e-7
        return np.all(test)
    
    def contains(self, point):
        """
        Return 'True' if 'self' contains 'point'.
        Boundary points are included.
        @param point: row vector, e.g., as 'numpy.ndarray'
        @rtype: bool
        For multiple points, see the method 'self.contains'.
        """
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        test = self.A.dot(point.T) - self.b[:,np.newaxis] < 10e-7
        return np.all(test,axis=0)
    

    @property
    def A(self):
        """
        'A' Matrix of the H-representation
        @rtype: np.ndarray
        """
        m,n = self.generator.shape
        if n>1:
            if n==2:
                G = np.vstack([-self.generator[:,1],self.generator[:,0]])
                G /= np.linalg.norm(G,axis=0)
                return np.vstack([G.T,-G.T])
            else:
                raise ValueError("Dimensions not supported")
    @property
    def b(self):
        """
        'b' Matrix of the H-representation
        @rtype: np.ndarray
        """
        m,n = self.generator.shape
        if n>1:
            if n==2:
                G = np.vstack([-self.generator[:,1],self.generator[:,0]])
                G /= np.linalg.norm(G,axis=0)
        D = np.sum(np.abs(G.T@self.generator.T),axis=1)
        d = G.T@self.center
        
        return np.hstack([D+d,D-d])
    
    def lpcontains(self,point):
        """
        Linear programming version to check if a given point lies within the zonotope
        @param point: vector , e.g., as 'numpy.ndarray'
        @rtype: bool
        """
        if not isinstance(point,np.ndarray):
                point = np.asarray(point)
        def objective(x):
            return np.linalg.norm(x,np.inf)
        def constriant(x,y):
            return self.generator.T@x + self.center - y
        
        constraint_eq = {'type': 'eq','fun': lambda x: constriant(x,point)}
        x0  = np.zeros(self.generator.shape[0])
        result = minimize(objective, x0, method='SLSQP', constraints=constraint_eq)

        return result.fun<=1
