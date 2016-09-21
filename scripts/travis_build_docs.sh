set -e
# create build folder
mkdir build
cd build
cmake -DVISIONCPP_DOC_ONLY=TRUE ..

# create documentation
make doc
