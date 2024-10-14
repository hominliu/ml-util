FROM ccir.prci.com/base_registry.redhat.io/ubi8/python-311 as cicd-image

# build time CLI Arguments
# for build arguments that does not have default value (ex. GIT_USERNAME)
# use --build-arg when calling docker build command in CLI
ARG PIP_INDEX
ARG HTTP_PROXY
ARG NO_PROXY

# set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    HTTP_PROXY=${HTTP_PROXY} \
    http_proxy=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTP_PROXY} \
    https_proxy=${HTTP_PROXY} \
    NO_PROXY=${NO_PROXY} \
    no_proxy=${NO_PROXY} \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_INDEX_URL=${PIP_INDEX}/simple \
    PIP_TRUSTED_HOSTS=yourorg.jfrog.io \
    REQUESTS_CA_BUNDLE=/location/to/your/ssl/certs/ca-bundle.crt

USER root

RUN yum -y install krb5-workstation krb5-libs bind-utils

# install custom elements
RUN yum -y install unixODBC unixODBC-devel

WORKDIR /source
COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install -U pip
RUN python -m pip install .[dev]
# create fake credential file
#RUN touch ~/.credrc
#RUN echo "USERNAME=tester" >> ~/.credrc
#RUN echo "PASSWORD=123" >> ~/.credrc
