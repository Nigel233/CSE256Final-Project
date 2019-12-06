from matplotlib import pyplot as plt
acc1_list = []
acc2_list = []
with open('log-test', 'r') as f:
    for line in f:
        if line != "\n":
            part1, part2 = line.strip('\n').split('%')[0], line.strip('\n').split('%')[1]
            acc1 = part1.split('acc1: ')[1]
            acc2 = part2.split('acc2: ')[1]
            acc1 = float(acc1)
            acc2 = float(acc2)
            acc1_list.append(acc1)
            acc2_list.append(acc2)
X = [i for i in range(len(acc1_list))]
plt.figure()
plt.plot(X, acc1_list, 'r')
plt.plot(X, acc2_list, 'b')
plt.show()


