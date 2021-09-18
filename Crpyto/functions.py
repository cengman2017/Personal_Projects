import math as m
import sympy
import numpy

def list_product(list):
    product = 1
    for n in list:
        product *= n
    return product

def gcd(a,b):
    if (b > a):
        return gcd(b,a)
    else:
        while (b != 0):
            r = a % b
            a = b
            b = r
        return a
    
def extended_euclid(a,b):
    if (b > a):
        return extended_euclid(b,a)
    else:
        r0 = a
        r1 = b
        s0 = 1
        s1 = 0
        t0 = 0
        t1 = 1
        while (r1 != 0):
            q = r0 // r1
            r2 = r0 % r1
            r = r1
            s = s1
            t = t1
            r1 = r2
            s1 = s0 - (s1 * q)
            t1 = t0 - (t1 * q)
            r0 = r
            s0 = s
            t0 = t
        d = r0
        return d, s0, t0

def extended_euclid_list(a,b):
    if (b > a):
        return extended_euclid_list(b,a)
    else:
        slist = []
        tlist = []
        rlist = []
        rlist.append(a)
        rlist.append(b)
        slist.append(1)
        slist.append(0)
        tlist.append(0)
        tlist.append(1)
        l = 1
        while (rlist[l] != 0):
            q = rlist[l-1] // rlist[l]
            r = rlist[l-1] % rlist[l]
            s = slist[l-1] - (slist[l] * q)
            t = tlist[l-1] - (tlist[l] * q)
            rlist.append(r)
            slist.append(s)
            tlist.append(t)
            l += 1
        return rlist, slist, tlist
             

def modular_inverse(b, n):
    d, s, t = extended_euclid(n, b)
    if (d != 1):
        return "no inverse"
    else:
        if (t < 0):
            return t + n
        else:
            return t

def solve_congruence(a, b, n):
    ## az = b mod n
    d = gcd(a, n)
    if ((b % d) != 0):
        return "no solution"
    else:
        a0 = a // d
        b0 = b // d
        n0 = n // d
        t = modular_inverse(a0, n0)
        z = (t * b0) % n0
        return z

def chinese_remainder(alist, nlist):
    n = list_product(nlist)
    if (len(nlist) != len(alist)):
        return "incorrect inputs"
    else:
        k = len(nlist)
        elist = []
        for i in range(k):
            nstar = n // nlist[i]
            b = nstar % nlist[i]
            t = modular_inverse(b, nlist[i])
            e = nstar * t
            elist.insert(i, e)
        a = 0
        for i in range(k):
            a += (alist[i] * elist[i])
        a = a % n
        return a
    
def factor(num):
    k = m.floor(m.sqrt(num))
    for i in range(2,k):
        if (num % i == 0):
            p = i
            q = num // i
            break
    return (p,q)

def primesInRange(x, y):
    prime_list = []
    for n in range(x, y):
        isPrime = True

        for num in prime_list:
            if n % num == 0:
                isPrime = False

        if isPrime:
            prime_list.append(n)
            
    return prime_list

def primesfrom2to(n):
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = numpy.ones(n//3 + (n%6==2), dtype=bool)
    for i in range(1,int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[       k*k//3     ::2*k] = False
            sieve[k*(k-2*(i&1)+4)//3::2*k] = False
    return numpy.r_[2,3,((3*numpy.nonzero(sieve)[0][1:]+1)|1)]

def pollard_rho_factor(n, start=2):
    x = start
    y = start
    d = 1
    while (d == 1):
        x = PR_g(x, n)
        y = PR_g(PR_g(y, n), n)
        xmy = abs(x-y)
        d = gcd(xmy, n)
    
    if d == n:
        return pollard_rho_factor(n, start+1)
    
    else:
        p = d
        q = n // d
        return (p, q)
    
def PR_g(x, n):
    output = (x**2 + 1) % n
    return output

def Pollard_rho_brent(n, start=2, m=1):
    y = start
    r = 1
    q = 1
    g = 1
    while g == 1:
        x = y
        for i in range(1, r+1):
            y = PR_g(y,n)
            k = 0
        while (k < r and g == 1):
            ys = y
            for i in range(1, min(m, r-k)+1):
                y = PR_g(y,n)
                q = q * abs(x-y) % n
            g = gcd(q,n)
            k += m
        r *= 2
    if g == n:
        while True:
            ys = PR_g(ys)
            g = gcd(abs(x-ys), n)
            if g > 1:
                break
    if g == n:
        Pollard_rho_brent(n, start+1)
    else:
        p = g
        q = n // g
        return (p,q)

