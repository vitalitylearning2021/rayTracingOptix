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
