import numpy as np
import math

class RLWEPrivateKey:
    def __init__(self, S, q, t, l, finv):
        self.S = S
        self.q = q
        self.t = t
        self.l = l
        self.finv = finv

class RLWEPublicKey:
    def __init__(self, A, P, r, m, q, l, n, f):
        self.A = A
        self.P = P
        self.r = r
        self.m = m
        self.q = q
        self.l = l
        self.n = n
        self.f = f
    
    def generate_a(self):
        a = np.random.randint(-self.r,self.r + 1, size=self.m, dtype=np.int64)
        return a

def generateLWEKey(n,l,m,q,r,t,alpha):
        S = np.random.randint(0,q,n,dtype=np.int64)
        A = np.random.randint(0,q,m,dtype=np.int64)
        stddev = (alpha * q) / (math.sqrt(math.pi * 2))
        E = np.around(np.random.normal(0,stddev,m)).astype(np.int64) % q
        P = (multiplyPoly(A,S, m) + E) % q
        f = lambda v:  np.around(v * (q/t)).astype(np.int64)
        finv = lambda v: np.around(v * (t/q)).astype(np.int64)
        return RLWEPrivateKey(S, q, t, l, finv), RLWEPublicKey(A,P, r, m, q, l, n, f)

def encryptBlock(msg, PubKey: RLWEPublicKey):
    a = PubKey.generate_a()
    AMat = generateRotMatrix(PubKey.A, PubKey.n)
    PMat = generateRotMatrix(PubKey.P, PubKey.l)
    u = np.matmul(AMat,a) % PubKey.q
    c = (np.matmul(PMat,a) + PubKey.f(msg)) % PubKey.q
    return u,c

def decryptBlock(u, c, PrivKey: RLWEPrivateKey):
    SMat = generateRotMatrix(PrivKey.S, PrivKey.l)
    return PrivKey.finv((c - np.matmul(SMat,u)) % PrivKey.q) % PrivKey.t
 
def multiplyPoly(a, b, len):
    returnpoly = np.zeros(len)
    for i in range(a.size):
        for j in range(b.size):
            idx = (i + j) % len
            cycles = int((i + j) // len)
            entry = a[i] * b[j] * pow(-1, cycles)
            returnpoly[idx] += entry
    return returnpoly
 
def generateRotMatrix(v, l):
    m = v.size
    returnmat = np.zeros((l, m))
    currentvector = v
    for i in range(l):
        returnmat[i] = currentvector
        currentvector = np.roll(currentvector, -1)
        currentvector[m-1] = -currentvector[m-1]
    return returnmat
     
 
if __name__ == "__main__":
    N = 233
    L = 233
    M = 4536
    Q = 32749
    R = 1
    T = 40
    ALPHA = 0.000217
    totalbits = 0
    totalcorrect = 0
    Private, Public = generateLWEKey(N,L,M,Q,R,T,ALPHA)
    for i in range(0,10):
        msg = np.random.randint(0,T,size=L, dtype=np.int64)
        U,C = encryptBlock(msg, Public)
        decryptedmsg = decryptBlock(U,C,Private)
        totalcorrect += np.sum(msg == decryptedmsg)
        totalbits += L
    
    errorrate = 1 - (totalcorrect / totalbits)
    print(errorrate * 100)
    