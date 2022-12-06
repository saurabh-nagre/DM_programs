
file = open("Apps\web-Google.txt","r")
data = file.readlines()
file.close()

file = open("Apps\web-Google.txt",'w')
count = 0
for i in data:
    str = ""
    flag = False
    for j in i:
        if not flag and j=='	':
            str = str+','
            flag = True
        else:
            str = str+j
    data[count] = str
    count+=1

file.writelines(data)

file.close()