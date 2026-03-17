import os
from concurrent import futures
import grpc
from google.protobuf.json_format import MessageToDict
from pprint import pprint

from pisa_api import av_server_pb2, av_server_pb2_grpc
from pisa_api.pong_pb2 import Pong
from pisa_api.empty_pb2 import Empty

from autoware import AutowarePureAV

import logging

from exception.av import (
    RouteError,
    LocalizationTimeoutError,
    PlanningTimeoutError,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)


class AVServer(av_server_pb2_grpc.AvServerServicer):
    def __init__(self):
        super().__init__()
        self._av = None

    def Ping(self, request, context):
        logger.info(f"Received ping from client: {context.peer()}")
        return Pong(msg="Autoware alive")

    def Init(self, request, context):
        output_dir = request.output_dir.path
        config = MessageToDict(request.config.config)
        scenario_pack = request.scenario_pack
        pprint(config)

        self._av = AutowarePureAV(output_dir, config)
        self._av.init(scenario_pack)

        return av_server_pb2.AvServerMessages.InitResponse(
            success=True, msg="Autoware initialized"
        )

    def Reset(self, request, context):
        output_dir = request.output_dir.path
        scenario_pack = request.scenario_pack
        initial_observation = request.initial_observation
        try:
            ret = self._av.reset(output_dir, scenario_pack, initial_observation)
        except LocalizationTimeoutError as e:
            logger.error(f"LocalizationTimeoutError during Reset: {str(e)}")
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(str(e))
            return av_server_pb2.AvServerMessages.ResetResponse(ctrl_cmd={})
        except PlanningTimeoutError as e:
            logger.error(f"PlanningTimeoutError during Reset: {str(e)}")
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(str(e))
            return av_server_pb2.AvServerMessages.ResetResponse(ctrl_cmd={})
        except RouteError as e:
            logger.error(f"RouteError during Reset: {str(e)}")
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(str(e))
            return av_server_pb2.AvServerMessages.ResetResponse(ctrl_cmd={})
        except Exception as e:
            logger.error(f"Unexpected error during Reset: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Unexpected error: {str(e)}")
            return av_server_pb2.AvServerMessages.ResetResponse(ctrl_cmd={})
        else:
            return av_server_pb2.AvServerMessages.ResetResponse(ctrl_cmd=ret)

    def Step(self, request, context):
        observation = request.observation
        timestamp_ns = request.timestamp_ns
        return av_server_pb2.AvServerMessages.StepResponse(
            ctrl_cmd=self._av.step(observation, timestamp_ns)
        )

    def Stop(self, request, context):
        self._av.stop()
        return Empty()

    def ShouldQuit(self, request, context):
        should_quit = self._av.should_quit()
        return av_server_pb2.AvServerMessages.ShouldQuitResponse(
            should_quit=should_quit
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    av_server_pb2_grpc.add_AvServerServicer_to_server(AVServer(), server)

    PORT = os.environ.get("PORT", "50051")

    server.add_insecure_port(f"[::]:{PORT}")
    server.start()

    print(f"gRPC server is running on port {PORT}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("Shutting down gRPC server")
        server.stop(0)


if __name__ == "__main__":
    serve()
