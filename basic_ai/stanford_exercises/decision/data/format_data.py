# formating .dat files to csv for simplicity and because its prettier :)

with open("stanford_exercises\\decision\\data\\x.dat", "r") as file:
    contend = file.read()

with open("stanford_exercises\\decision\\data\\y.dat", "r") as file:
    contend_y = file.read() # a little bit of lazy coding :) 

contend = contend.split()
contend_y = contend_y.split()

x1 = []
x2 = []
for i in range(len(contend)):
    if(i%2):
        x2.append(contend[i])
    else:
        x1.append(contend[i])

with open("stanford_exercises\\decision\\data\\dataq2.csv", "w") as file:
    file.write("x1, x2, y\n")
    for i in range(len(x1)):
        file.write(x1[i] + ' , ' + x2[i] + ' , ' + contend_y[i] + "\n")  
