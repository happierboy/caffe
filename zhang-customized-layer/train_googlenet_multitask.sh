#!/usr/bin/env bash
export PYTHONPATH=$PWD:/home/mozat/git/caffe-revision/caffe/python
echo $PYTHONPATH
/home/mozat/git/caffe-revision/caffe/build/tools/caffe train \
-solver  "models/bvlc_googlenet_multask/solver.prototxt" \
-weights "models/bvlc_googlenet_multask/snapshot/mult__iter_100000.caffemodel" \
-gpu 0 2>&1 | tee googlenet-multask3.txt
