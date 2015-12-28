# PDTB-style discourse parser (CoNLL15st format).
#
# Example:
#   NAME=conll15st-s4-all
#   docker build -t $NAME .
#   docker run -d -v /srv/storage/conll15st-ex:/srv/ex --name $NAME-train $NAME ex/$NAME-train conll15st-train conll15st-dev x x
#
# Author: GW [http://gw.tnode.com/] <gw.2015@tnode.com>

FROM debian:jessie
MAINTAINER gw0 [http://gw.tnode.com/] <gw.2015@tnode.com>

ENV DEBIAN_FRONTEND noninteractive
WORKDIR /srv/

# packages
RUN apt-get update -qq \
 && apt-get install -y \
    python \
    python-pip \
    python-virtualenv

# setup virtualenv
ADD requirements.sh ./

RUN ./requirements.sh

# setup parser
ADD conll15st-dev/ ./conll15st-dev/
ADD conll15st-train/ ./conll15st-train/
ADD conll15st-trial/ ./conll15st-trial/
ADD conll15st_scorer/ ./conll15st_scorer/
ADD parser12/ ./parser12/

RUN useradd -r -d /srv parser \
 && mkdir -p /srv/ex \
 && chown -R parser:parser /srv

# expose interface
VOLUME /srv/ex

USER parser
ENTRYPOINT ["/srv/venv/bin/python", "/srv/parser12/run.py"]
CMD ["ex/ex12", "conll15st-train", "conll15st-dev", "conll15st-dev", "ex/ex12"]
