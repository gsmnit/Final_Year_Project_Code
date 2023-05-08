from datetime import datetime
import json
from math import sqrt


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


with open('acndata_sessions_jpl_dec2020_may2023.json', 'r') as f:
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


for item in data['_items']:
    weekday = item['connectionTime'][0:3];
    # print(weekday)
    arrival_hour = convert_time_format(item['connectionTime'])
    unplug_hour = convert_time_format(item['disconnectTime'])
    connection_time = time_duration(arrival_hour, unplug_hour)
    kwh = item['kWhDelivered']
    feature = [arrival_hour, connection_time, kwh]
    mapping[weekday].append(feature)

# for weekday,sessions in mapping.items():
#     print(weekday)
#     if(weekday == 'Mon'):print(sessions)
# print(mapping)
# print(data)

with open('Mon.json', 'w') as M:
    json.dump(mapping['Mon'], M)

with open('Tue.json', 'w') as T:
    data = json.dump(mapping['Tue'], T)

with open('Wed.json', 'w') as W:
    data = json.dump(mapping['Wed'], W)

with open('Thu.json', 'w') as Th:
    data = json.dump(mapping['Thu'], Th)

with open('Fri.json', 'w') as f:
    data = json.dump(mapping['Fri'], f)

with open('Sat.json', 'w') as S:
    data = json.dump(mapping['Sat'], S)

with open('Sun.json', 'w') as Su:
    data = json.dump(mapping['Sun'], Su)



