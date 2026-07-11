#!/usr/bin/bash
if [[ "${1,,}" == "clean" ]]; then
    rm -rf build
    mkdir build
fi

cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) && ./raytracer
