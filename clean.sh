#!/bin/bash

rm 9*.loop

rm -rf cpp/build/*

unzip ../googletest-release-1.8.0.zip -d cpp/build/
mv cpp/build/googletest-release-1.8.0 cpp/build/googletest
