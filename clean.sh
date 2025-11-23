#!/bin/bash

# Clean up log files
rm *.log

# Clean up build directory
rm -rf ./cpp/build

# Clean up distribution directory
rm -rf ./dist

# Recreate build directory
mkdir -p ./cpp/build/googletest
unzip ../googletest-release-1.8.0.zip -d ./cpp/build/googletest
