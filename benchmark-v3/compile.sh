#!/bin/bash

cd ../build_temp &&
# cmake
cmake -DCMAKE_INSTALL_PREFIX=../ .. &&
make &&
make install &&
rm -rf ./* &&



# 4. insatll Magicsphere pytorch 
cd ../ &&
rm -rf build &&
python setup.py install