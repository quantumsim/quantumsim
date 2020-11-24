FROM nvidia/cuda:11.0-devel-ubuntu20.04

# make our environment sane
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git curl pandoc python3.9-dev python3.9-venv python3.9-tk \
        # Additional tools for running CI
        file rsync openssh-client && \
    echo UTC > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# virtualenv preparation
# vritualenv is needed because CI fails because of SyntaxError in pytools,
# which is fixed in Python 3.9. Current Ubuntu does not allow to get
# Python-3.9 with pip otherwise.
RUN /usr/bin/python3.9 -m venv /root/venv && \
    /root/venv/bin/pip install --upgrade --no-cache wheel && \
    /root/venv/bin/pip install --upgrade --no-cache \
        codecov \
        pytest \
        pylint \
        pytest-cov \
        twine

# actual requirements
COPY requirements*.txt /root/
# We need to install mpmath from git because in Python 3.8 `smth is 0` gives `SyntaxError`
# Please, @fredrik-johansson, release :)
# https://github.com/fredrik-johansson/mpmath/issues/547
RUN /root/venv/bin/pip install --upgrade --no-cache -r /root/requirements.txt \
                                     -r /root/requirements-docs.txt && \
    /root/venv/bin/pip install --upgrade --no-cache -r /root/requirements-cuda.txt

RUN ssh-keyscan -t rsa gitlab.com >> /etc/ssh/ssh_known_hosts && \
    ssh-keyscan -t rsa github.com >> /etc/ssh/ssh_known_hosts
