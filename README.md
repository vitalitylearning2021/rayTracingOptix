# rayTracingOptix

Compiling without CMake is the same as compiling normal CUDA code except that this must be compiled with the "-ptx" flag and then loaded using the OptiX API in order for it to link properly.

https://github.com/torchling/optix

https://owl-project.github.io/

https://developer.nvidia.com/blog/how-to-get-started-with-optix-7/

https://github.com/ingowald

https://www.cs.ucdavis.edu/~ma/ECS275/OptiX_lecture.pdf

https://github.com/nvpro-samples/optix_advanced_samples

Directory di inclusione: aggiungere C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.2.0\include;C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.2.0\SDK

Aggiungere cuda.lib al linker

Compilare in modalità release

INTEROPERABILITY

## Getting started with OptiX

The core OptiX 7 API is header only. The include directory contains everything needed to access the OptiX API core functions. The OptiX 7 headers along with the CUDA toolkit is everything needed to develop GPU accelerated ray tracing algorithms with OptiX 7. To account for the OptiX include directory under Visual Studio, add `C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.2.0\include` to the VC++ include directories under `Configuration Properties`.

## Optix initializaion

In this very simple example, how initializing OptiX is shown. The core of the example is the `initOptix()` function reported below:

``` c++
void initOptix() {

    try {
        std::cout << "Initializing Optix..." << std::endl;

        gpuErrchk(cudaFree(0));
        int numDevices;
        gpuErrchk(cudaGetDeviceCount(&numDevices));
        if (numDevices == 0) throw std::runtime_error("No CUDA capable devices found!");
        std::cout << "Found " << numDevices << " CUDA devices" << std::endl;

        optixAssert(optixInit());

        std::cout << "Optix successfully initialized." << std::endl;
    }
    catch (std::runtime_error& e) {
        std::cout << "FATAL ERROR: " << e.what() << std::endl;
        exit(1);
    }
}```
