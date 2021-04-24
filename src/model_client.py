import logging
import grpc

import onnxmodel_pb2  # generated
import onnxmodel_pb2_grpc  # generated


def run():
    with grpc.insecure_channel('localhost:8501') as channel:
        stub = onnxmodel_pb2_grpc.InferencerStub(channel)
        response = stub.MakeInference(
            onnxmodel_pb2.InferenceRequest(
                sepal_length=5.1,
                sepal_width=4.9,
                petal_length=4.7,
                petal_width=4.6
            )
        )
    print(response)


if __name__ == '__main__':
    logging.basicConfig()
    run()
