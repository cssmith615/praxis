# Praxis Agent — production Docker image
#
# Build:  docker build -f praxis/agent/Dockerfile -t praxis-agent .
# Run:    docker run --env-file .env praxis-agent
#
# All configuration is via environment variables (see docker-compose.yml).

FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install praxis with agent dependencies
# Copy pyproject + source before installing so layer cache is efficient
COPY pyproject.toml README.md ./
COPY praxis/ praxis/

# Install with the 'agent' optional group (anthropic SDK)
RUN pip install --no-cache-dir -e ".[agent]"

# Persist databases and logs in a volume
VOLUME ["/root/.praxis"]

# Non-root user for security
RUN useradd -m -u 1000 praxis
RUN mkdir -p /root/.praxis && chown -R praxis:praxis /root/.praxis
USER praxis

# Health check — verify praxis CLI loads
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from praxis.agent import PraxisAgent; print('ok')"

ENTRYPOINT ["praxis", "agent"]
