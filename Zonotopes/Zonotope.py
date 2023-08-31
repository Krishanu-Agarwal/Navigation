import torch
import matplotlib.pyplot as plt

class Zonotope():
    def __init__(self, Cen, Gen):
        
        self.center = torch.Tensor(Cen)
        self.generators = torch.Tensor(Gen)
        self.center.requires_grad_()
        self.generators.requires_grad_()
        self.center.retain_grad()
        self.generators.retain_grad()
    
    @property
    def A(self):
        """
        'A' Matrix of the H-representation
        @rtype: np.ndarray
        """
        m,n = self.generators.shape
        if n>1:
            if n==2:
                G = torch.vstack([-self.generators[:,1 ], self.generators[:, 0]])
                G /= torch.norm(G, dim=0)
                A = torch.vstack([G.T, -G.T])
                A.retain_grad()
                return A
            else:
                raise ValueError("Dimensions not supported")
    @property
    def b(self):
        """
        'b' Matrix of the H-representation
        @rtype: torch.TensorType
        """
        m,n = self.generators.shape
        if n>1:
            if n==2:
                G = torch.vstack([-self.generators[:, 1], self.generators[:, 0]])
                G /= torch.norm(G, dim=0)
        D = torch.sum(torch.abs(G.T @ self.generators.T), dim=1)
        d = torch.matmul(G.T,self.center)
        b = torch.hstack([D+d,D-d])
        b.retain_grad()
        return b

        
        return np.hstack([D+d,D-d])
    def __add__(self, other):
        """
        Returns a Zonotpe Z = Z1 +Z2. 
        Z.center  = Z1.center + Z2.center
        Z.Generator = [Z1.Generator,Z2.Generator]
        Overloaded the Plus '+' operator to add 2 Zonotopes
        @param other: Zonotope
        @rtype: Zonotope
        """        
        if isinstance(other, Zonotope):
            return Zonotope( self.center + other.center, torch.concatenate((self.generators, other.generators), axis=0))
        else: 
            raise TypeError( "Unsupported operand type. Can add Zonotopes only." )
        
    def __mul__(self, other):
        """
        Returns a Zonotope Z = Z1*Z2
        Z.Center = [Z1.Center,Z2.Center]
        Z.Generator = Diag(Z1.Generator,Z2.Generator)
        Overloaded the Multiplication '*' operator to 
        @param other: Zonotope
        @rtype: Zonotope
        """
        if isinstance(other, Zonotope):
            center = torch.concatenate([self.center, other.center])
            generator = torch.block_diag(self.generators, other.generators)
            return Zonotope(center, generator)
        else:
            raise TypeError("Unsupported operand type. Can multiply Zonotopes only.")
    def __contains__(self, point):
        """
        Return 'True' if 'self' contains 'point'.
        Boundary points are included.
        @param point: column vector, e.g., as 'numpy.ndarray'
        @rtype: bool
        For multiple points, see the method 'self.contains'.
        """
        if isinstance(point, torch.TensorType):
            test = self.A.dot(point.flatten()) - self.b < 10e-7
            return torch.all(test)
        elif isinstance(point, Zonotope):
            Zono = Zonotope(point.center,torch.concatenate((self.generators, point.generators), axis=0))
            # TODO: Handle linearly dependable generators. 
            test = Zono.A.dot(self.center.flatten()) - Zono.b < 10e-7
            return torch.all(test)
    def contains(self, point):
        """
        Return 'True' if 'self' contains 'point'.
        Boundary points are included.
        @param point: row vector, e.g., as 'numpy.ndarray'
        @rtype: bool
        For multiple points, see the method 'self.contains'.
        """
        if isinstance(point, torch.TensorType):
            test = self.A.dot(point.T) - self.b.unsqueeze(-1) < 10e-7
            return torch.all(test,axis=0)
    
   
