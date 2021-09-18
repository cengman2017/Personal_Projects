import math as m
import functions as f
import time
import random

primesFile = open("list_of_primes.txt", 'r')

list_of_primes = []
for line in primesFile:
    primek = int(line)
    list_of_primes.append(primek)

p = random.choice(list_of_primes)
q = random.choice(list_of_primes)
e = 2**16 + 1

n = p * q

phi = (p - 1) * (q - 1)

d = f.modular_inverse(e, phi)

a = 483029108829

b = pow(a, e, n)

aprime = pow(b, d, n)

print(a)
print(b)
print(aprime)
print(n)



##start = time.time()

##factors = factor(n)

##end = time.time()
##duration = end - start

##print(factors)
##print(duration)
