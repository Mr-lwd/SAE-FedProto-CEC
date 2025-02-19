import json
import copy


alpha = 0.75
# alpha = 1.0
class CustomObject:
    def __init__(self, frequency, average_power, infer_time=0):
        self.frequency = frequency
        self.average_power = average_power
        self.infer_time = infer_time
        self.energy = self.average_power * self.infer_time
    

def knapsack_problem(objects, max_frequency_time, leave_time, num_items=1):
    tempdata = copy.deepcopy(objects)
    # print("objects",objects[0])
    # exit(0)
    time_energy_tuple = []
    intscale = 10
    leave_time *= intscale
    max_frequency_time *= intscale
    max_frequency = max(item["frequency"] for item in tempdata)
    for item in tempdata:
        if item["frequency"] == max_frequency:
            obj = CustomObject(
                item["frequency"], item["average_power"], int(round(max_frequency_time))
            )
        else:
            obj = CustomObject(
                item["frequency"],
                item["average_power"],
                int(round(max_frequency_time * (max_frequency / item["frequency"]) ** alpha)),
            )
        # print("obj.infer_time",obj.infer_time)
        time_energy_tuple.append(obj)
    leave_time=int(leave_time)
    # print("leave_time",leave_time)
    # dp[t] represents the minimum energy to achieve total time t
    dp = [[float("inf")] * (num_items + 1) for _ in range(leave_time + 1)]
    dp[0][0] = 0

    item_count = [[[] for _ in range(num_items + 1)] for _ in range(leave_time + 1)]

    # Dynamic Programming solution
    for obj in time_energy_tuple:
        infer_time_int = int(obj.infer_time)
        for t in range(infer_time_int, leave_time + 1):
            for k in range(1, num_items + 1):
                if dp[t - infer_time_int][k - 1] + obj.energy < dp[t][k]:
                    dp[t][k] = dp[t - infer_time_int][k - 1] + obj.energy
                    item_count[t][k] = item_count[t - infer_time_int][k - 1] + [
                        (obj, 1)
                    ]  # Update item count

    # Now we need to find the minimum energy for exactly 6 items and any time <= leave_time
    min_energy = float("inf")
    best_time = -1
    for t in range(leave_time + 1):
        if dp[t][num_items] < min_energy:
            min_energy = dp[t][num_items]
            best_time = t

    # print(f"最小能量: {min_energy}")
    # print(f"消耗时间: {best_time}")
    # print("选择的功率及其次数:")
    selected_freqs = []
    for item in item_count[best_time][num_items]:
        selected_freqs.append(item[0].frequency)
    # print("selected_freqs",selected_freqs)
    return min_energy, selected_freqs
        # print(f"Frequency: {item[0].frequency}，选择次数: {item[1]}")


def get_dvfs_set(objects, max_frequency_time, leave_time, num_items=1):
    energy, selected_frequency_set = knapsack_problem(objects, max_frequency_time, leave_time, num_items)
    # print("theory_min_energy",energy)
    # print("selected_frequency_set",selected_frequency_set)
    return selected_frequency_set


# data_dvfs = [
#     {
#         "frequency": 345600,
#         "average_power": 482.8531468531468
#     },
#     {
#         "frequency": 499200,
#         "average_power": 560.6060606060606
#     },
#     {
#         "frequency": 652800,
#         "average_power": 637.7672955974842
#     },
#     {
#         "frequency": 806400,
#         "average_power": 720.2574626865671
#     },
#     {
#         "frequency": 960000,
#         "average_power": 807.2231759656653
#     },
#     {
#         "frequency": 1113600,
#         "average_power": 893.0714285714286
#     },
#     {
#         "frequency": 1267200,
#         "average_power": 1030.2746113989638
#     },
#     {
#         "frequency": 1420800,
#         "average_power": 1221.142857142857
#     },
#     {
#         "frequency": 1574400,
#         "average_power": 1431.7575757575758
#     },
#     {
#         "frequency": 1728000,
#         "average_power": 1685.1483870967743
#     },
#     {
#         "frequency": 1881600,
#         "average_power": 1991.4931506849316
#     },
#     {
#         "frequency": 2035200,
#         "average_power": 2376.536231884058
#     }
# ]

# get_dvfs_set(data_dvfs, 6.7, 13.9, 2)
