import matplotlib.pyplot as plt
x = []
y = []
with open("./performance.txt") as fin:
    for line in fin:
        arr = line.replace("\n","").split("  ")
        x.append(float(arr[0]))
        y.append(float(arr[1]))
plt.plot(x,y)
plt.show()
