import time
import math as m
import random
import functions as f
import xlwt
from xlwt import Workbook

wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')

primesFile = open("list_of_primes.txt", 'r')

list_of_primes = []
for line in primesFile:
    primek = int(line)
    list_of_primes.append(primek)

def factor(num):
    k = m.floor(m.sqrt(num))
    for i in range(2,k):
        if (num % i == 0):
            p = i
            q = num // i
            break
    return (p,q)

def factor_with_primes(num):
    k = m.floor(m.sqrt(num))
    for i in list_of_primes:
        if (num % i == 0):
            p = i
            q = num // i
            break
    return (p,q)

def generate_rsa_mod():
    p = random.choice(list_of_primes)
    q = random.choice(list_of_primes)
    n = p * q
    return n

sheet1.write(0, 0, 'n')
sheet1.write(0, 1, 'regular trial division time')
sheet1.write(0, 2, 'prime trial division time')
sheet1.write(0, 3, 'pollard rho time')
sheet1.write(0, 4, 'factors')

for i in range(1000):
    n = generate_rsa_mod()
    sheet1.write(i+1, 0, n)

    start = time.time()
    factor1 = factor(n)
    end = time.time()
    time1 = end - start

    start = time.time()
    factor2 = factor_with_primes(n)
    end = time.time()
    time2 = end - start

    start = time.time()
    factor3 = f.pollard_rho_factor(n)
    end = time.time()
    time3 = end - start

    sheet1.write(i+1, 1, time1)
    sheet1.write(i+1, 2, time2)
    sheet1.write(i+1, 3, time3)
    sheet1.write(i+1, 4, str(factor2))

wb.save("factoringtimes.xls")

#2**24
#Regular Trial Division: 600 ms
#Prime Trial Division: 40 ms
#Pollard Rho: 16 ms
