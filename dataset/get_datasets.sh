#!/bin/sh

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

echo "Downloading..."


if [ ! -f "epsilon_normalized" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
bunzip2 epsilon_normalized.bz2
fi

if [ ! -f "url_combined_normalized" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/url_combined_normalized.bz2
bunzip2 url_combined_normalized.bz2
fi

if [ ! -f "real-sim" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2
bunzip2 real-sim.bz2
fi

if [ ! -f "new20.binary" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2 
bunzip2 news20.binary.bz2
fi

if [ ! -f "kdda" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.bz2 
bunzip2 kdda.bz2
fi

if [ ! -f "kddb" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kddb.bz2 
bunzip2 kddb.bz2
fi

if [ ! -f "kdd12" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdd12.xz
xz -d kdd12.xz
fi

if [ ! -f "kddb-raw-libsvm" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kddb-raw-libsvm.bz2 
bunzip2 kddb-raw-libsvm.bz2
fi

if [ ! -f "webspam_wc_normalized_trigram.svm" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.xz 
xz -d webspam_wc_normalized_trigram.svm.xz 
fi

if [ ! -f "E2006.train" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.train.bz2
bunzip2 E2006.train.bz2
fi

if [ ! -f "log1p.E2006.train" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/log1p.E2006.train.bz2 
bunzip2 log1p.E2006.train.bz2
fi

if [ ! -f "rcv1_test.binary" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2 
bunzip2 rcv1_test.binary.bz2
fi


