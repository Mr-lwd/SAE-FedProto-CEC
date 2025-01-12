import json

# 读取原始JSON文件
with open('frequency&power.json', 'r') as file:
    data = json.load(file)

# 提取frequency和average_power字段
extracted_data = [{"frequency": item["frequency"], "average_power": item["average_power"]} for item in data]

# 将提取的数据保存到新文件
with open('extracted_data.json', 'w') as file:
    json.dump(extracted_data, file, indent=4)