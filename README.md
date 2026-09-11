Toy Llama

Build llama inference stack from scratch.

Build the debug version:

cmake -B build           -DCMAKE_BUILD_TYPE=Debug

or

cmake -B build

Build the release version:

cmake -B build-release   -DCMAKE_BUILD_TYPE=RelWithDebInfo

Run REPL

$build/toy_llama

Print logs to terminal

GLOG_logtostderr=1 $build/toy_llama

