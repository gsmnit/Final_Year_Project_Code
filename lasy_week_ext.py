from datetime import datetime
import json
import matplotlib.pyplot as plt


def convert_time_format(time_str):
    time_str = time_str[-12:-4]
    time_obj = datetime.strptime(time_str, '%H:%M:%S')
    hour_val = time_obj.hour + time_obj.minute / 60.0
    return hour_val


def time_duration(time1, time2):
    if time2 >= time1:
        return time2 - time1
    else:
        return 24 - time1 + time2


with open('last_week.json', 'r') as f:
    data = json.load(f)

mapping = {
    'Mon': [],
    'Tue': [],
    'Wed': [],
    'Thu': [],
    'Fri': [],
    'Sat': [],
    'Sun': []
}

week_num = {
    'Mon': 0,
    'Tue': 1,
    'Wed': 2,
    'Thu': 3,
    'Fri': 4,
    'Sat': 5,
    'Sun': 6
}

arrivaltime_power = []

for item in data['_items']:
    weekday = item['connectionTime'][0:3];
    # print(weekday)
    arrival_hour = convert_time_format(item['connectionTime'])
    unplug_hour = convert_time_format(item['disconnectTime'])
    connection_time = time_duration(arrival_hour, unplug_hour)
    kwh = item['kWhDelivered']
    feature = [arrival_hour, connection_time, kwh]
    mapping[weekday].append(feature)
    arrivaltime_power.append((week_num[weekday]+(arrival_hour/24),kwh))

# for weekday,sessions in mapping.items():
#     print(weekday)
#     if(weekday == 'Mon'):print(sessions)
# print(mapping)
# print(data)

with open('Mon_A.json', 'w') as M:
    json.dump(mapping['Mon'], M)

with open('Tue_A.json', 'w') as T:
    data = json.dump(mapping['Tue'], T)

with open('Wed_A.json', 'w') as W:
    data = json.dump(mapping['Wed'], W)

with open('Thu_A.json', 'w') as Th:
    data = json.dump(mapping['Thu'], Th)

with open('Fri_A.json', 'w') as f:
    data = json.dump(mapping['Fri'], f)

with open('Sat_A.json', 'w') as S:
    data = json.dump(mapping['Sat'], S)

with open('Sun_A.json', 'w') as Su:
    data = json.dump(mapping['Sun'], Su)


# Plotting
arrivaltime_power = sorted(arrivaltime_power, key=lambda x: x[0])
x = [i[0] for i in arrivaltime_power]
y = [i[1] for i in arrivaltime_power]
plt.plot(x,y)

# Customize the plot
plt.title("Actual Power consumption for week 7 sept 2021 to 14 sept 2021" )
plt.xlabel("time ")
plt.ylabel("power(kWh)")
plt.grid(True)

# Show the plot
plt.show()





