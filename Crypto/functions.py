import math as m
import sympy
import numpy
import random

def tobits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def frombits(bits):
    chars = []
    for b in range(len(bits) // 8):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)

def list_product(list):
    product = 1
    for n in list:
        product *= n
    return product


def factorial(n):
    fact = 1
    for i in range(1, n + 1):
        fact *= i
    return fact


def quicksort(l, start=0, end=None):
    length = len(l)
    if end is None:
        end = length - 1
    if start < end and length > 1:
        pivot = l[end]
        i = start - 1
        for j in range(start, end):
            if l[j] <= pivot:
                i += 1
                l[i], l[j] = l[j], l[i]
        l[i + 1], l[end] = l[end], l[i + 1]
        l = quicksort(l, start, i)
        l = quicksort(l, i + 2, end)
    return l


def search_list(A, key):
    for i in range(len(A)):
        if A[i] == key:
            return i
    return


def gcd(a, b):
    if b > a:
        return gcd(b, a)
    else:
        while b != 0:
            r = a % b
            a = b
            b = r
        return a


def extended_euclid(a, b):
    if b > a:
        return extended_euclid(b, a)
    else:
        r0 = a
        r1 = b
        s0 = 1
        s1 = 0
        t0 = 0
        t1 = 1
        while r1 != 0:
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


def extended_euclid_list(a, b):
    if b > a:
        return extended_euclid_list(b, a)
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
        while rlist[l] != 0:
            q = rlist[l - 1] // rlist[l]
            r = rlist[l - 1] % rlist[l]
            s = slist[l - 1] - (slist[l] * q)
            t = tlist[l - 1] - (tlist[l] * q)
            rlist.append(r)
            slist.append(s)
            tlist.append(t)
            l += 1
        return rlist, slist, tlist


def modular_inverse(b, n):
    d, s, t = extended_euclid(n, b)
    if d != 1:
        print("no inverse")
        return
    else:
        if t < 0:
            return t + n
        else:
            return t


def solve_congruence(a, b, n):
    # az = b mod n
    d = gcd(a, n)
    if (b % d) != 0:
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
    if len(nlist) != len(alist):
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


def modular_sqrt(a, p):
    # Simple cases
    #
    if legendre_symbol(a, p) != 1:
        return []
    elif a == 0:
        return [0]
    elif p == 2:
        return [0]
    elif p % 4 == 3:
        r = pow(int(a), int((p + 1) // 4), int(p))
        return [r, p - r]
    s = p - 1
    e = 0
    while s % 2 == 0:
        s /= 2
        e += 1
    n = 2
    while legendre_symbol(n, p) != -1:
        n += 1
    x = pow(int(a), int((s + 1) // 2), int(p))
    b = pow(int(a), int(s), int(p))
    g = pow(int(n), int(s), int(p))
    r = e

    while True:
        t = b
        m = 0
        for m in range(r):
            if t == 1:
                break
            t = pow(int(t), int(2), int(p))

        if m == 0:
            return [x, p - x]

        gs = pow(int(g), int(2 ** (r - m - 1)), int(p))
        g = (gs * gs) % p
        x = (x * gs) % p
        b = (b * g) % p
        r = m


def legendre_symbol(a, p):
    ls = pow(int(a), int((p - 1) // 2), int(p))
    if ls == p - 1:
        return -1
    else:
        return 1


def factor(num):
    k = m.floor(m.sqrt(num))
    for i in range(2, k):
        if num % i == 0:
            p = i
            q = num // i
            break
    return (p, q)


def primesInRange(x, y):
    prime_list = []
    for n in range(x, y + 1):
        isPrime = True

        for num in prime_list:
            if n % num == 0:
                isPrime = False

        if isPrime:
            prime_list.append(n)

    return prime_list


def primesfrom2to(n):
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = numpy.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return numpy.r_[2, 3, ((3 * numpy.nonzero(sieve)[0][1:] + 1) | 1)]


def pollard_rho_factor(n, start=2):
    x = start
    y = start
    d = 1
    while d == 1:
        x = PR_g(x, n)
        y = PR_g(PR_g(y, n), n)
        xmy = abs(x - y)
        d = gcd(xmy, n)

    if d == n:
        return pollard_rho_factor(n, start + 1)
    else:
        p = d
        q = n // d
        return (p, q)


def Pollard_rho_brent(n, start=2, m=1):
    if sympy.isprime(n):
        return [n]
    else:
        y = start
        r = 1
        q = 1
        g = 1
        while g == 1:
            x = y
            for i in range(1, r + 1):
                y = PR_g(y, n)
                k = 0
            while k < r and g == 1:
                ys = y
                for i in range(1, min(m, r - k) + 1):
                    y = PR_g(y, n)
                    q = q * abs(x - y) % n
                g = gcd(q, n)
                k += m

            r *= 2
        if g == n:
            while True:
                ys = PR_g(ys, n)
                g = gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g == n:
            return Pollard_rho_brent(n, start + 1)
        else:
            f1 = g
            f2 = n // g
            output = Pollard_rho_brent(f1) + Pollard_rho_brent(f2)
            return output


def PR_g(x, n):
    output = (x ** 2048 + 1) % n
    return output


def prime_factorization(n):
    factors = quicksort(Pollard_rho_brent(n))
    factorization = []
    p = factors[0]
    e = 0

    for i in factors:
        if i != p:
            factorization.append((p, e))
            p = i
            e = 1
        else:
            e += 1

    factorization.append((p, e))

    return factorization


def totient_function(n):
    factors = prime_factorization(n)

    phi = n

    for (p, e) in factors:
        phi *= (1 - (1 / p))

    return int(phi)


def Dixon_factor(n, b=1):
    if n < 2:
        return []
    if sympy.isprime(n):
        return [n]

    if b < 6:
        # b = m.floor(m.exp(m.sqrt(m.log(n) * m.log(m.log(n)))))
        b = 7

    base = primesfrom2to(b + 1)
    basesize = len(base)

    baseproduct = 1
    for i in base:
        baseproduct *= int(i)

    start = m.floor(m.sqrt(n)) + 1

    current = start
    x = 1
    y = 1

    while (x - y) % n == 0 or (x + y) % n == 0:
        k = 0
        Z = []
        A = sympy.Matrix()
        while k <= basesize:
            z2 = current ** 2 % n
            smooth = is_smooth_dixon(z2, baseproduct)
            if smooth and z2 > 0:
                factors = factor_over_base(z2, base)
                print(current, z2, k, factors)
                Z.append(current)
                A = A.col_insert(k, sympy.Matrix(factors))
                k += 1
            current += 1
        Aprime = A % 2
        Aprime = Aprime.col_insert(k + 1, sympy.Matrix([0 for x in range(basesize)]))
        [sols] = list(sympy.linsolve(Aprime))

        T = []
        for i in range(k):
            if sols[i] != 0:
                T.append(i)
        print(T)

        x = 1
        for i in T:
            x = (x * Z[i]) % n
        print(x)

        y = 1
        for j in range(basesize):
            power = 0
            for i in T:
                power += A[j, i]
            power = power // 2
            y = (y * (base[j] ** power)) % n
        print(y)

    g = gcd(x + y, n)
    print(g)

    if g == 1:
        Dixon_factor(n, b * 2)
    else:
        fact1 = g
        fact2 = n // g
        facts = Dixon_factor(fact1) + Dixon_factor(fact2)
        return facts


def factor_over_base(n, base):
    m = n
    factors = [0 for x in base]
    j = 0
    for i in base:
        mprime = m
        (m, rem) = divmod(m, i)
        while rem == 0:
            mprime = m
            (m, rem) = divmod(m, i)
            factors[j] += 1
        m = mprime
        if m == 1:
            break
        j += 1

    return factors


def is_smooth_dixon(n, baseproduct):
    if n == 1:
        return True

    m = n
    e = gcd(m, baseproduct)

    if e <= 1:
        return False

    rem = 0
    while rem == 0:
        mprime = m
        (m, rem) = divmod(m, e)
    m = mprime

    return is_smooth_dixon(m, e)


def quadratic_sieve(N, B=None):
    # initialization
    if N < 2:
        return []
    if sympy.isprime(N):
        return [N]

    if B is None:
        B = m.ceil(m.sqrt(m.exp(m.sqrt(m.log(N) * m.log(m.log(N))))))

    primes = primesfrom2to(B + 1)

    base = [2]
    for i in primes[1:]:
        if legendre_symbol(N, i) == 1:
            base.append(i)

    baseproduct = 1
    for i in base:
        baseproduct *= int(i)

    K = len(base)

    print(N, B, base, K)

    # sieving
    X = m.ceil(m.sqrt(N))
    I = 1000
    print(X)
    sieve_seq = [x ** 2 - N for x in range(X, X + I)]
    sieve_list = [m.log(x) for x in sieve_seq]

    for p in base:
        p1 = p
        while p1 < B:
            residues = modular_sqrt(N, p1)
            lp = m.log(p1)
            print(residues)
            for r in residues:  # r^2 - N = 0 mod p, r = x+X
                for i in range((r - X) % p1, len(sieve_list), p1):
                    sieve_list[i] -= lp
            p1 *= p

    # print(xlist)
    # print(sieve_seq)
    # print(sieve_list)
    xlist = []
    indicies = []
    smooths = []

    for i in range(len(sieve_list)):
        if len(smooths) >= K:
            break
        elif sieve_list[i] < 1:
            if is_smooth_dixon(sieve_seq[i], baseproduct):
                smooths.append(sieve_seq[i])
                xlist.append(i + X)
                indicies.append(i)

    print(indicies, xlist, smooths)

    # Matrix Construction

    A = sympy.Matrix()

    for i in smooths:
        factors = factor_over_base(i, base)
        A = A.col_insert(i, sympy.Matrix(factors))

    Aprime = A % 2
    Aprime = Aprime.col_insert(len(smooths) + 1, sympy.Matrix([0 for x in range(K)]))
    print(Aprime)
    [sols] = list(sympy.linsolve(Aprime))
    T = []
    for i in range(len(smooths)):
        if sols[i] != 0:
            T.append(i)
    print(T)

    x = 1
    for i in T:
        x = (x * xlist[i]) % N
    print(x)

    y = 1
    for j in range(K):
        power = 0
        for i in T:
            power += A[j, i]
        power = power // 2
        y = (y * (base[j] ** power)) % N
    print(y)

    g = gcd(x + y, N)
    print(g)
    return (g)


def pollard_rho_discrete_log(g, t, p):  # g ^ s = t mod p, returns s
    aj, ak = 0, 0
    bj, bk = 0, 0
    xj, xk = 1, 1

    # x = t^a * g^b
    def new_xab(x, a, b):
        if x < (1 / 3) * p:
            x, a, b = (t * x) % p, (a + 1) % (p - 1), b
        elif x < (2 / 3) * p:
            x, a, b = (x * x) % p, (2 * a) % (p - 1), (2 * b) % (p - 1)
        else:
            x, a, b = (g * x) % p, a, (b + 1) % (p - 1)
        return x, a, b

    xj, aj, bj = new_xab(xj, aj, bj)
    xk, ak, bk = new_xab(xk, ak, bk)
    xk, ak, bk = new_xab(xk, ak, bk)

    while (xj != xk):
        xj, aj, bj = new_xab(xj, aj, bj)
        xk, ak, bk = new_xab(xk, ak, bk)
        xk, ak, bk = new_xab(xk, ak, bk)

    a = (aj - ak) % (p - 1)
    b = (bk - bj) % (p - 1)
    l = solve_congruence(a, b, p - 1)
    return l


def baby_steps_giant_steps(g, t, n):
    b = m.ceil(m.sqrt(n - 1))
    g_inv = modular_inverse(g, n)
    h = pow(g_inv, b, n)

    A = [pow(g, i, n) for i in range(b)]
    B = [(t * pow(h, j, n)) % n for j in range(b)]

    Asorted = quicksort(A.copy())
    Bsorted = quicksort(B.copy())

    x, y = 0, 0

    while Asorted[x] != Bsorted[y]:
        if Asorted[x] < Bsorted[y]:
            x += 1
        else:
            y += 1
        if x >= b or y >= b:
            print("no discrete log")
            exit()

    q = Asorted[x]
    r = Bsorted[y]

    i = search_list(A, q)
    j = search_list(B, r)

    l = (i + j * b) % n

    return l


def Pohlig_Hellman_prime_power(g, h, p, e, n):  # p^e is the subgroup order, n is the modulus of the full group
    x = 0
    gamma = pow(g, p ** (e - 1), n)
    for k in range(e):
        hk = pow((h * pow(g, 0 - x, n)) % n, p ** (e - 1 - k), n)  # h = g ^ (xe xe-1 xe-2 ..xk)
        dk = baby_steps_giant_steps(gamma, hk, n)
        x = x + (p ** k) * dk
    return x


def Pohlig_Hellman_discrete_log(g, h, n):
    order = totient_function(n)

    order_factorization = prime_factorization(order)

    X = []
    N = []

    for (p, e) in order_factorization:
        gi = pow(g, order // (p ** e), n)
        hi = pow(h, order // (p ** e), n)
        xi = Pohlig_Hellman_prime_power(gi, hi, p, e, n)
        X.append(xi)
        N.append(p ** e)

    x = chinese_remainder(X, N)

    return x


def subgroup_orders(n):  # returns the order of each <x> from Z*_n
    group_order = totient_function(n)
    print("The order of the multiplicative group Z*_{} is".format(n, 's'), group_order, "\n")

    print("x", "subgroup order")

    for i in range(n):
        if gcd(i, n) == 1:
            x = i
            k = 1
            while x != 1:
                x = (x * i) % n
                k += 1
            print(i, k)


if __name__ == "__main__":
    print(pollard_rho_factor(400000001, 3))