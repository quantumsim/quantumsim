FROM nvidia/cuda:11.0-devel-ubuntu20.04

# make our environment sane
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git curl pandoc python3-all-dev python3-pip python3-setuptools \
        python3-wheel python3-tk \
        # Additional tools for running CI
        file rsync openssh-client && \
    echo UTC > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# auxillary dependencies
RUN pip3 install --upgrade --no-cache pip && \
    pip install --upgrade --no-cache \
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
RUN pip install --upgrade --no-cache git+https://github.com/fredrik-johansson/mpmath.git && \
    pip install --upgrade --no-cache -r /root/requirements.txt \
                                     -r /root/requirements-docs.txt && \
    pip install --upgrade --no-cache -r /root/requirements-cuda.txt

RUN ssh-keyscan -t rsa gitlab.com >> /etc/ssh/ssh_known_hosts && \
    ssh-keyscan -t rsa github.com >> /etc/ssh/ssh_known_hosts
