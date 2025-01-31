import multiprocessing
import subprocess
import time


# Check the endpoint
def run_monitor_health():

    subprocess.run(["python", "health_check.py"])


# Call the endpoint
def run_endpoint_calling():
    time.sleep(70)
    subprocess.run(["python", "endpoint_calling.py"])


if __name__ == "__main__":
    # Create processes for each script
    monitor_process = multiprocessing.Process(target=run_monitor_health)
    endpoint_process = multiprocessing.Process(target=run_endpoint_calling)

    # Start both processes
    monitor_process.start()
    endpoint_process.start()

    # Wait for both processes to finish
    monitor_process.join()
    endpoint_process.join()
