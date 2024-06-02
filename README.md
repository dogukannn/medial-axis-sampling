## Compile

Compile this project using the standard cmake routine:

    mkdir build
    cmake -S . -B build

## Dependencies

There is a problem in the build script that requires to manually copy the 'libgmp-10.dll' to build folder from build\_deps\gmp-src\lib
