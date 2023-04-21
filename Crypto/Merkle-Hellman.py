import random
import functions
import math

class privateMHKCKey:
    def __init__(self,N,W,q,r,rprime):
        self.N = N
        self.W = W
        self.q = q
        self.r = r
        self.rprime = rprime

class publicMHKCKey:
    def __init__(self,N,B):
        self.N = N
        self.B = B
        

def generate_mhkc_key(block_size:int=64):
    N = block_size

    W = []
    w = random.randint(1,3)
    for i in range(N):
        W.append(w)
        w = random.randint(2*w,3*w)

    q = w

    r = random.randint(0,q)
    while functions.gcd(q,r) != 1:
        r = random.randint(0,q)

    rprime = functions.modular_inverse(r,q)

    B = []
    for w in W:
        b = (r * w) % q
        B.append(b)
    
    return(publicMHKCKey(N,B), privateMHKCKey(N,W,q,r,rprime))


def encrypt_message(message,publicKey:publicMHKCKey):
    B = publicKey.B
    N = publicKey.N
    
    def encrypt_block(m):
        c = 0
        for bit, b in zip(m,B):
            c += bit * b
        return c
    
    msgbitarray = functions.tobits(message)
    numbits = len(msgbitarray)
    goodnumbits = N * math.ceil(numbits / N)
    extrazeros = goodnumbits - numbits
    msgbitarray += [0]*extrazeros
    
    
    ciphertext = []
    for i in range(0,goodnumbits,N):
        block = msgbitarray[i:i+N]
        ciphertext.append(encrypt_block(block))
    
    return ciphertext
    
    

def decrypt_message(ciphertext,privateKey:privateMHKCKey):
    N = privateKey.N
    W = privateKey.W
    q = privateKey.q
    r = privateKey.r
    rprime = privateKey.rprime
    
    def decrypt_block(c):
        cprime = (c * rprime) % q
        m = []
        for w in reversed(W):
            if cprime >= w:
                m.append(1)
                cprime -= w
            else:
                m.append(0)
    
        m.reverse()
        return m
    
    
    plaintextbits = []
    for c in ciphertext:
        cbits = decrypt_block(c)
        plaintextbits += cbits
    plaintext = functions.frombits(plaintextbits)
    return plaintext

if __name__ == "__main__":
    public,private = generate_mhkc_key(64)
    M = "Hello World"
    C = encrypt_message(M,public)
    Mprime = decrypt_message(C,private)
    
    print(M)
    print(C)
    print(Mprime)
    
    

