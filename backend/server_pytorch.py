import os

print("### ENV in docker_main_baseline.py")
print(os.environ)


# run PyTorch model server
uvicorn_proc = subprocess.Popen(["uvicorn main_pytorch:app --reload"], shell=True)

print(" ### Start baseline-model endpoint :)")