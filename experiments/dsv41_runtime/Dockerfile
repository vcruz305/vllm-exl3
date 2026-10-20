# The caller must supply a retained compatible V4.1 image. No public default
# is implied: the historical parent image was local, not a registry release.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY . /opt/sage330-source
RUN python3 -S /opt/sage330-source/install_overlay.py
RUN python3 /opt/sage330-source/build_extensions.py --output /opt/sage330-built
RUN python3 -S /opt/sage330-source/install_overlay.py \
    --apply-in-container --extensions /opt/sage330-built
