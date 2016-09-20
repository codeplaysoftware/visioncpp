set -e
# create build folder
mkdir build
cd build
cmake ..

# create documentation
make doc
