from autoware import AutowarePureAV
from pisa_api.av import serve_av_system
from pisa_api.wrapper import setup_logging
from version import wrapper_version

setup_logging()


if __name__ == "__main__":
    serve_av_system(
        AutowarePureAV(),
        name="autoware-wrapper",
        version=wrapper_version(),
    )
