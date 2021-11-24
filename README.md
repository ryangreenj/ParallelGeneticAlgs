## Parallel Computing Final Project - Ryan Green and Jacob Carlson

Build the project using the following command:

``nvcc -std c++17 -o out -I Include -g $(find Source -type f \( -iname \*.cpp -o -iname \*.cu \))``

Then run the resulting output using:

`./out`