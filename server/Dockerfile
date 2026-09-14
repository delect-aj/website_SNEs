# The only container the site needs. Static files are served by nginx from the
# host, not from here.
#
# vsearch comes from bioconda because it is not packaged on most distributions
# and a source build needs its full dependency set. Pinning it means a rebuild
# produces the same aligner the results were checked against.

FROM mambaorg/micromamba:1.5.10-jammy

COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /tmp/requirements.txt

RUN micromamba install --yes --name base \
        --channel conda-forge --channel bioconda \
        "vsearch=2.28.1" "python=3.11" && \
    micromamba clean --all --yes

# The image activates the environment only for CMD. RUN lines and the compose
# healthcheck (which bypasses the entrypoint) need it on PATH, or `python` is
# not found and the build stops here.
ENV PATH=/opt/conda/bin:$PATH

RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /srv
COPY --chown=$MAMBA_USER:$MAMBA_USER app /srv/app

# Mounted read-only at run time; see docker-compose.yml.
ENV OTU_REFSEQS=/srv/data/otu_refseqs.fasta

EXPOSE 8000

# One worker on purpose: vsearch runs single-threaded and a second worker would
# only contend with the first on a 1-2 core host.
#
# nginx reaches the container through Docker's bridge, not from 127.0.0.1, so
# uvicorn would ignore X-Forwarded-For by default and every visitor would share
# one rate-limit budget. Trusting it is safe: the port is published on the
# host's loopback only, and nginx overwrites the header rather than appending.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", \
     "--port", "8000", "--workers", "1", "--no-access-log", \
     "--forwarded-allow-ips", "*"]
