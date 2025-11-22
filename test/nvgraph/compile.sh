#!/bin/bash

# compiling test scripts

echo "=> Compiling Page Rank"
nvcc page_rank.cpp -o page_rank.out -lnvgraph -lcudart -lstdc++
echo "Finished Compiling Page Rank"

echo "=> Compiling Triangle Count"
nvcc triangle_count.cpp -o triangle_count.out -lnvgraph -lcudart -lstdc++
echo "Finished Compiling Triangle Count"

# executing test scripts

echo "=> Executing Page Rank"
./page_rank.out
echo "Finished Page Rank"

echo "=> Executing Triangle Count"
./triangle_count.out
echo "Finished Triangle Count"