inputs = [1, -1, 3, 0.2, -0.6, -2, 4]

output = []

""""
for i in inputs:
    if i>0:
        output.append(i)
    else:
        output.append(0)
"""

for i in inputs:
    output.append(max(0,i))
print(output)