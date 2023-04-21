import sympy
import numpy
import math as m
import time

primesFile = open("list_of_primes.txt", 'w')

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


#start_time_1 = time.time()

#primes1 = primesInRange(2, 2**16)

#end_time_1 = time.time()

#time1 = end_time_1 - start_time_1

#print(primes1)
#print(time1)

start_time_2 = time.time()

primes2 = primesfrom2to(2**24)

for n in primes2:
    nstr = str(n)
    primesFile.write(nstr + "\n")

end_time_2 = time.time()

time2 = end_time_2 - start_time_2

print(time2)

primesFile.close()