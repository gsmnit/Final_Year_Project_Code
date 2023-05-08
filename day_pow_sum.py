import json

days = {'Mon','Tue','Wed','Thu','Fri','Sat','Sun'}
# Open the JSON file
for day in days:
    file_name = day+'_A.json'
    print(file_name)
    with open(file_name) as file:
        #Load the contents of the file into a Python object
        data = json.load(file)

    # Now you can work with the data object
    # print(data)
    pow_sum = 0
    for ses in data:
        pow_sum += ses[2]

    print(len(data))
    print(pow_sum)
