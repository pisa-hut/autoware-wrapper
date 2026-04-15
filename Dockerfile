FROM ghcr.io/autowarefoundation/autoware:universe-devel-cuda-20260209

ENV DEBIAN_FRONTEND=noninteractive

RUN <<EOF
    apt update
    apt install -y git
    rm -rf /var/lib/apt/lists/*
EOF

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN useradd -ms /bin/bash autoware
RUN chown -R autoware:autoware /opt/autoware/share/autoware_launch/launch /opt/autoware/share/autoware_launch/config
USER autoware

WORKDIR /app
COPY ./pyproject.toml .
COPY ./uv.lock .
RUN uv sync --locked

COPY misc/pisa.launch.xml /opt/autoware/share/autoware_launch/launch/

COPY . .
RUN python3 misc/config.py --apply

ENV PORT=50051
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ENTRYPOINT [ "/bin/bash" ]
CMD [ "/app/entrypoint.sh" ]
