# Conduit gateway image.
#
# Two stages, for two different reasons. The builder needs a compiler toolchain
# and pip's whole cache to resolve a dependency tree; the runtime needs neither,
# and shipping them would put a build environment on a public-facing service.
# What crosses the boundary is one directory: a fully-populated virtualenv.
#
# The image installs the *core* distribution only — no `[anthropic]`, no
# `[openai]`, no `[l2c]`. That is the containerised form of AC-16 and AC-17: a
# gateway that physically cannot import a vendor SDK, and that therefore starts
# with zero API keys on the mock + Ollama path. Adding a key at runtime is a
# config change (the registry instantiates an adapter when it sees the env var),
# not an image rebuild — the adapters are pure httpx and need no vendor package.

# --------------------------------------------------------------------------- #
# Stage 1 — build the virtualenv
# --------------------------------------------------------------------------- #
FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

RUN python -m venv "$VIRTUAL_ENV"

# The sources are copied before the install because hatchling needs the package
# tree present to build the wheel it then installs, and `readme = "README.md"`
# in pyproject makes the README a build input rather than documentation.
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY apps/ ./apps/

RUN pip install --no-cache-dir .

# --------------------------------------------------------------------------- #
# Stage 2 — runtime
# --------------------------------------------------------------------------- #
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# `conduit.config.default_config_dir` walks up from the *source* tree, which no
# longer exists once the package is installed into site-packages. Pointing the
# loader at the copied config directory is what makes the installed
# (non-editable) layout work at all.
ENV CONDUIT_CONFIG_DIR=/app/config
ENV CONDUIT_TELEMETRY__DIR=/app/var/telemetry
ENV PORT=8000

# uid/gid are pinned rather than left to useradd's next-free counter, so a
# volume created by one build is still writable by the next one.
RUN groupadd --gid 10001 conduit \
    && useradd --uid 10001 --gid 10001 --no-log-init --create-home --shell /usr/sbin/nologin conduit

WORKDIR /app

COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv
COPY --chown=10001:10001 config/ /app/config/

# Docker seeds a fresh named volume from the image's directory *including its
# ownership*, so pre-creating this as conduit-owned is what lets the non-root
# process write telemetry into a mounted volume without a chown sidecar.
RUN install -d -o 10001 -g 10001 /app/var /app/var/telemetry

USER 10001:10001

EXPOSE 8000

# No curl in the runtime image, and no reason to add one: the interpreter is
# already here. `urlopen` raises on a non-2xx, so any failure — connection
# refused, 503 from a half-started app — exits non-zero and the check fails.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/healthz',timeout=4).read()"]

# Shell form so Cloud Run's injected $PORT is honoured; `exec` so uvicorn is
# pid 1 and receives SIGTERM directly rather than through a shell that drops it.
# `--proxy-headers` because both Cloud Run and compose put a proxy in front.
CMD ["sh", "-c", "exec uvicorn conduit.gateway.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*'"]
