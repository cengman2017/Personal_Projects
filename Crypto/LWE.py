import numpy as np
import math

results = open("LWEResults.txt", 'w')

class LWEPrivateKey:
    def __init__(self, S, q, t, finv):
        self.S = S
        self.q = q
        self.t = t
        self.finv = finv

class LWEPublicKey:
    def __init__(self, A, P, r, m, q, f):
        self.A = A
        self.P = P
        self.r = r
        self.m = m
        self.q = q
        self.f = f
    
    def generate_a(self):
        a = np.random.randint(-self.r,self.r + 1, size=self.m, dtype=np.int64)
        return a

def generateLWEKey(n,l,m,q,r,t,alpha):
        S = np.random.randint(0,q,(n,l),dtype=np.int64)
        A = np.random.randint(0,q,(m,n),dtype=np.int64)
        stddev = (alpha * q) / (math.sqrt(math.pi * 2))
        E = np.around(np.random.normal(0,stddev,(m,l))).astype(np.int64) % q
        P = (np.matmul(A,S) + E) % q
        f = lambda v:  np.around(v * (q/t)).astype(np.int64)
        finv = lambda v: np.around(v * (t/q)).astype(np.int64)
        return LWEPrivateKey(S, q, t, finv), LWEPublicKey(A,P, r, m, q, f)

def encryptBlock(msg, PubKey: LWEPublicKey):
    a = PubKey.generate_a()
    u = np.matmul(PubKey.A.T,a) % PubKey.q
    c = (np.matmul(PubKey.P.T,a) + PubKey.f(msg)) % PubKey.q
    return u,c

def decryptBlock(u, c, PrivKey: LWEPrivateKey):
    return PrivKey.finv((c - np.matmul(PrivKey.S.T,u)) % PrivKey.q) % PrivKey.t
 
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
    for i in range(0,1000):
        msg = np.random.randint(0,T,size=L, dtype=np.int64)
        U,C = encryptBlock(msg, Public)
        decryptedmsg = decryptBlock(U,C,Private)
        totalcorrect += np.sum(msg == decryptedmsg)
        totalbits += L
    
    errorrate = 1 - (totalcorrect / totalbits)
    print(errorrate * 100)
    results.write(str(errorrate*100) + "\n")
    