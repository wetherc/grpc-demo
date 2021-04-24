import logging
import grpc
import onnxruntime as rt
import numpy as np

import onnxmodel_pb2  # generated
import onnxmodel_pb2_grpc  # generated

from concurrent import futures


class Inferencer(onnxmodel_pb2_grpc.InferencerServicer):

    def MakeInference(self, request, context):
        input = np.array([
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]).astype(np.float32).reshape([1, 4])

        session = rt.InferenceSession('src/model.onnx')
        pred_onx = session.run(
            None,
            {session.get_inputs()[0].name: input})[0]

        return onnxmodel_pb2.Prediction(pred_prob=pred_onx)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    onnxmodel_pb2_grpc.add_InferencerServicer_to_server(Inferencer(), server)
    server.add_insecure_port('[::]:8501')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    print('starting server')
    serve()
    print('stopping server')
