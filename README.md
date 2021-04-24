# Overview

This provides a proof-of-concept ONNX runtime able to serve requests over gRPC.


# Usage

All code blocks assume you are running from the repository root directory.

- Install any Python requirements (Python >=3.6)
  ```bash
  python3 -m pip install -r requirements.txt
  ```
- Train and export the model (using the iris dataset)
  ```bash
  python3 src/train.py
  ```
- Generate the gRPC code from our service definition:
  ```bash
  python3 -m grpc_tools.protoc \
    -I./src/protos \
    --python_out=./src \
    --grpc_python_out=./src \
    ./src/protos/onnxmodel.proto
  ```
- Start the gRPC server
  ```bash
  python3 src/model_server.py
  ```
- Serve a request. In a separate terminal window, run the client application
  ```bash
  python3 src/model_client.py
  ```
