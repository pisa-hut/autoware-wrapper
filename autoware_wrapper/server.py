from autoware import AutowarePureAV
from pisa_api.av import serve_av_system
from pisa_api.wrapper import setup_logging

setup_logging()


if __name__ == "__main__":
    serve_av_system(
        AutowarePureAV(),
        name="Autoware",
    )
