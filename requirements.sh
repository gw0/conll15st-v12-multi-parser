#!/bin/bash
# Requirements installation for conll15st-multi-lstm-keras
#
# Author: gw0 [http://gw.tnode.com/] <gw.2015@tnode.com>
# License: All rights reserved

NAME="(`basename $(realpath ${0%/*})`)"
SRC="venv/src"
SITE_PACKAGES='venv/lib/python*/site-packages'
DIST_PACKAGES='/usr/lib/python*/dist-packages'
PYTHON_EXE="python2"

sudo() { [ -x "/usr/bin/sudo" ] && /usr/bin/sudo "$@" || "$@"; }
[ "$PYTHON_EXE" -eq "pypy" -a ! -x "/usr/bin/$PYTHON_EXE" ] && sudo aptitude install pypy

cd "${0%/*}"
virtualenv --prompt="$NAME" --python="$PYTHON_EXE"  venv || exit 1
source venv/bin/activate
[ ! -e "$SRC" ] && mkdir "$SRC"

# Prerequisites for Theano
sudo aptitude install python-numpy python-scipy
[ ! -d $SITE_PACKAGES/numpy ] && cp -a $DIST_PACKAGES/numpy* $SITE_PACKAGES
[ ! -d $SITE_PACKAGES/scipy ] && cp -a $DIST_PACKAGES/scipy* $SITE_PACKAGES

# Prerequisites for h5py
sudo aptitude install cython libhdf5-dev
[ ! -d $SITE_PACKAGES/Cython ] && cp -a $DIST_PACKAGES/[Cc]ython* $SITE_PACKAGES

# Prerequisites for Keras
sudo aptitude install libyaml-dev

# Prerequisites for matplotlib
#sudo aptitude install libfreetype6-dev

# Requirements
pip install git+https://github.com/Theano/Theano.git
pip install h5py
pip install pydot-ng
pip install -e git+https://github.com/fchollet/keras.git#egg=Keras

# Workarounds for keras
sed -i 's/^import pydot$/import pydot_ng as pydot\nfrom keras.models import Sequential, Graph/' $SITE_PACKAGES/keras/utils/visualize_util.py

echo
echo "Use: . venv/bin/activate"
echo
