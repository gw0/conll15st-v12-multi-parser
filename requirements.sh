#!/bin/bash
# Requirements installation for conll15st-multi-lstm-keras
#
# Author: gw0 [http://gw.tnode.com/] <gw.2015@tnode.com>
# License: All rights reserved

NAME="(`basename $(realpath ${0%/*})`)"
SRC="venv/src"
PYTHON_EXE="python2"
SITE_PACKAGES='venv/lib/python*/site-packages'
DIST_PACKAGES='/usr/lib/python*/dist-packages'

# Initialization
sudo() { [ -x "/usr/bin/sudo" ] && /usr/bin/sudo "$@" || "$@"; }
[ ! -x "/usr/bin/$PYTHON_EXE" ] && sudo apt-get install -y "$PYTHON_EXE"

cd "${0%/*}"
virtualenv --prompt="$NAME" --python="$PYTHON_EXE"  venv || exit 1
source venv/bin/activate
[ ! -e "$SRC" ] && mkdir "$SRC"

# Prerequisites for pip
sudo apt-get install -y git

# Prerequisites for Theano
sudo apt-get install -y python-numpy python-scipy
[ ! -d $SITE_PACKAGES/numpy ] && cp -a $DIST_PACKAGES/numpy* $SITE_PACKAGES
[ ! -d $SITE_PACKAGES/scipy ] && cp -a $DIST_PACKAGES/scipy* $SITE_PACKAGES

# Prerequisites for h5py
#sudo apt-get install -y cython libhdf5-8 libhdf5-dev
#[ ! -d $SITE_PACKAGES/Cython ] && cp -a $DIST_PACKAGES/[Cc]ython* $SITE_PACKAGES

# Prerequisites for Keras
sudo apt-get install -y python-h5py libyaml-dev graphviz
[ ! -d $SITE_PACKAGES/h5py ] && cp -a $DIST_PACKAGES/h5py* $SITE_PACKAGES

# Prerequisites for matplotlib
#sudo apt-get install -y libfreetype6-dev

# Requirements
pip install git+https://github.com/Theano/Theano.git
pip install pydot-ng
pip install git+https://github.com/fchollet/keras.git

echo
echo "Use: . venv/bin/activate"
echo
