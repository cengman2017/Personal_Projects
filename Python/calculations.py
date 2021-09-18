people = 16
daysinyear = 365

prob = 1

for i in range(people):
    prob *= (daysinyear - i) / daysinyear
    
    
prob = 1 - prob

print(prob)