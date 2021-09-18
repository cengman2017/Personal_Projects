import random
import math as m
import time

daysInYear = 10000

days = range(daysInYear)

collisionTimes = []

def average(list):
    length = len(list)
    sum = 0
    for i in list:
        sum += i
    average = sum / length
    return average

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.val = None
    
    def insert(self, key):
        if self.val is None:
            self.val = key
            self.left = Node()
            self.right = Node()
            return False
        else:
            if self.val == key:
                return True
            elif self.val < key:
                return self.right.insert(key)
            else:
                return self.left.insert(key)

# main loop

start = time.time()


for i in range(100):
    print(i)
    birthdays = Node()
    collision = False
    collisionTime = 0
    while not collision:
        newBirthday = random.choice(days)
        collisionTime += 1
        collision = birthdays.insert(newBirthday)
    collisionTimes.append(collisionTime)


'''

for i in range(100):
    print(i)
    birthdays = []
    collision = False
    collisionTime = 0
    while not collision:
        newBirthday = random.choice(days)
        collisionTime += 1
        for i in birthdays:
            if newBirthday == i:
                collision = True
                break
        birthdays.append(newBirthday)
    collisionTimes.append(collisionTime)

'''
    
end = time.time()
    
runtime = end - start

averageCollisionTime = average(collisionTimes)

theoreticalCollisionTime = 1/2 + m.sqrt(1/4 + 2.2 * m.log(2) * daysInYear)

print("Average Collision Time: ")
print(averageCollisionTime)
print("Approximate Theoretical Collision Time:")
print(theoreticalCollisionTime)
print("Calculation took this many seconds:")
print(runtime)

"""
Time to find 100 collisions
days | time     |  average n before collision
2^8  | 20 ms    |  20.74
365  | 27 ms    |  24.04
2^12 | 79 ms    |  84.55
2^16 | 410 ms   |  323
2^20 | 1.87 s   |  1233
2^24 | 7.92 s   |  4750
2^28 | 39.73 s  |  20232
2^32 | 185.61 s |  83975
"""