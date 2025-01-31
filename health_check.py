import requests
import subprocess
import time

# Configurations
ENDPOINT = "http://127.0.0.1:8000/score"
UVICORN_COMMAND = ["uvicorn", "main:app", "--reload"]


def is_endpoint_healthy(endpoint):

    try:
        response = requests.get(endpoint, timeout=5)
        return response.status_code == 200
    except requests.RequestException as e:
        print(f"Health check failed: {e}")
        return False


def restart_uvicorn():

    print("Restarting Uvicorn server...")
    subprocess.Popen(UVICORN_COMMAND)


def monitor_health():

    while True:
        if is_endpoint_healthy(ENDPOINT):
            print(f"Endpoint {ENDPOINT} is healthy.")
        else:
            print(f"Endpoint {ENDPOINT} is down. Restarting Uvicorn...")
            restart_uvicorn()

        # Check every 30 seconds
        time.sleep(60)


if __name__ == "__main__":
    monitor_health()
