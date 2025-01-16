import time
import datetime

def monitor_time_with_sleep():
    # Record start time
    start_time = time.perf_counter()
    start_datetime = datetime.datetime.now()
    
    print(f"Starting at: {start_datetime}")
    
    # Do some work
    print("Doing task 1...")
    time.sleep(2)  # Sleep for 2 seconds
    
    # Check intermediate time
    mid_time = time.perf_counter()
    print(f"After sleep 2s - Time elapsed: {mid_time - start_time:.2f} seconds")
    
    print("Doing task 2...")
    time.sleep(3)  # Sleep for 3 seconds
    
    # Record end time
    end_time = time.perf_counter()
    end_datetime = datetime.datetime.now()
    
    print(f"Ending at: {end_datetime}")
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")
    print(f"Expected total sleep time: 5 seconds")

if __name__ == "__main__":
    monitor_time_with_sleep()
