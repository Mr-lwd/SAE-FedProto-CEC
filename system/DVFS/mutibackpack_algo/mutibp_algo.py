import json
import time


max_frequency_time = 7000


class CustomObject:
    def __init__(self, frequency, average_power, infer_time=0):
        self.frequency = frequency
        self.average_power = average_power
        self.infer_time = infer_time
        self.energy = self.average_power * self.infer_time


def create_objects_from_json(file_path):
    with open(file_path, "r") as file:
        extracted_data = json.load(file)

    objects = []

    max_frequency = max(item["frequency"] for item in extracted_data)

    for item in extracted_data:
        if item["frequency"] == max_frequency:
            obj = CustomObject(
                item["frequency"],
                int(round(item["average_power"])),
                int(round(max_frequency_time)),
            )
        else:
            obj = CustomObject(
                item["frequency"],
                int(round(item["average_power"])),
                int(round(max_frequency_time * max_frequency / item["frequency"])),
            )
        objects.append(obj)

    return objects


def knapsack_problem(objects, max_time, num_items=6):

    dp = [[float("inf")] * (num_items + 1) for _ in range(max_time + 1)]
    dp[0][0] = 0

    item_count = [[[] for _ in range(num_items + 1)] for _ in range(max_time + 1)]

    # Dynamic Programming solution
    for obj in objects:
        infer_time_int = int(round(obj.infer_time))
        for t in range(infer_time_int, max_time + 1):
            for k in range(1, num_items + 1):
                if dp[t - obj.infer_time][k - 1] + obj.energy < dp[t][k]:
                    dp[t][k] = dp[t - obj.infer_time][k - 1] + obj.energy
                    item_count[t][k] = item_count[t - obj.infer_time][k - 1] + [
                        (obj, 1)
                    ]  # Update item count

    # Now we need to find the minimum energy for exactly 6 items and any time <= max_time
    min_energy = float("inf")
    best_time = -1
    for t in range(max_time + 1):
        if dp[t][num_items] < min_energy:
            min_energy = dp[t][num_items]
            best_time = t

    # print(f"最小能量: {min_energy}")
    # print(f"消耗时间: {best_time}")
    # print("选择的功率及其次数:")
    selected_freqs = []
    for item in item_count[best_time][num_items]:
        selected_freqs.append(item[0].frequency)
        # print(f"Frequency: {item[0].frequency}，选择次数: {item[1]}")

    return selected_freqs


def main():
    extracted_data_file = "extracted_data.json"
    objects = create_objects_from_json(extracted_data_file)

    # test
    max_time = (
        max_frequency_time * 7 - max_frequency_time + 200
    )  # Backpacks' total time capacity
    selected_freqs = knapsack_problem(objects, max_time, num_items=6)
    print(selected_freqs)

if __name__ == "__main__":
    main()
