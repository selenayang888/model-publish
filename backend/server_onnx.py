import os

# run ONNX model server
uvicorn_proc = subprocess.Popen(["uvicorn main_onnx:app --reload"], shell=True)

print(" ### Start ONNX-model endpoint :)")