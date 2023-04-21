from sympy.matrices import Matrix, GramSchmidt
from sympy import shape
import functions
import sys, math

class Lattice:
    def __init__(self,B:Matrix):
        self.basisMatrix = B
        m, n = shape(self.basisMatrix)
        self.basisArray = []
        for i in range(n):
            self.basisArray.append(B.col(i))
        if self.basisMatrix.nullspace() != []:
            raise ValueError("Improper Basis, Column Vectors are linearly Dependent")
        if m == n:
            self.fullrank = True
        else:
            self.fullrank = False
    
    def det(self):
        if self.fullrank:
            return abs(self.basisMatrix.det())
        else:
            BtB:Matrix = self.basisMatrix.T * self.basisMatrix
            return math.sqrt(BtB.det())
    
    def LLL(self,delta=3/4):
        ortho:Matrix = GramSchmidt(self.basisArray)
        n = len(ortho)
        print(ortho)
        for i in range(1,n):
            for j in reversed(range(i-1)):
                pass
    
            
L = Lattice(Matrix([[1,2,3],[4,5,6],[7,8,10]]))
print(L.LLL())
