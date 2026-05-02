import logging
from pprint import pprint

import grpc
from autoware import AutowarePureAV
from exception.av import (
    LocalizationTimeoutError,
    PlanningTimeoutError,
    RouteError,
)
from google.protobuf.json_format import MessageToDict
from pisa_api import av_server_pb2
from pisa_api.empty_pb2 import Empty
from pisa_api.wrapper import BaseAvServer, serve_av, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class AVServer(BaseAvServer):
    _name = "Autoware"

    def __init__(self):
        super().__init__()
        self._av = None

    def Init(self, request, context):
        logger.debug(f"Received Init request from client: {context.peer()}")
        config = MessageToDict(request.config.config)
        output_dir = request.output_dir.path
        map_name = request.map_name
        pprint(config)

        self._av = AutowarePureAV(output_dir, config)
        self._av.init(map_name)

        return av_server_pb2.AvServerMessages.InitResponse(success=True, msg="Autoware initialized")

    def Reset(self, request, context):
        logger.debug(f"Received Reset request from client: {context.peer()}")
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
            logger.exception(f"Unexpected error during Reset: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Unexpected error: {str(e)}")
            return av_server_pb2.AvServerMessages.ResetResponse(ctrl_cmd={})
        else:
            return av_server_pb2.AvServerMessages.ResetResponse(ctrl_cmd=ret)

    def Step(self, request, context):
        logger.debug(f"Received Step request with timestamp_ns={request.timestamp_ns}")
        observation = request.observation
        timestamp_ns = request.timestamp_ns
        return av_server_pb2.AvServerMessages.StepResponse(
            ctrl_cmd=self._av.step(observation, timestamp_ns)
        )

    def Stop(self, request, context):
        logger.debug(f"Received Stop request from client: {context.peer()}")
        self._av.stop()
        return Empty()

    def ShouldQuit(self, request, context):
        should_quit = self._av.should_quit()
        return av_server_pb2.AvServerMessages.ShouldQuitResponse(should_quit=should_quit)


if __name__ == "__main__":
    serve_av(AVServer(), name="Autoware")
