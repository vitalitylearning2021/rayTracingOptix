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

Gli esempi illustrati sono modellati a partire dal corso tenuto da Ingo Wald al SIGGRAPH Couse 2019/2020 con qualche variante:

  - il codice è organizzato in modo da minimizzare le dipendenze esterne;
  - la `thrust` library è utilizzata come container al posto del `CUDABuffer`;
  - the CUDA-OpenGL interoperability è utilizzata per la visualizzazione delle rendered images;

## Getting started with OptiX

The core OptiX 7 API is header only. The include directory contains everything needed to access the OptiX API core functions. The OptiX 7 headers along with the CUDA toolkit is everything needed to develop GPU accelerated ray tracing algorithms with OptiX 7. To account for the OptiX include directory under Visual Studio, add `C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.2.0\include` to the VC++ include directories under `Configuration Properties`.

OptiX consente il tracciamento dei raggi in maniera particolarmente efficiente ed efficace. Il motore di tracciamento di OptiX è basato sull'uso della Boundary Volume Hierarchy (BVH) data structure [RIF] accelerata su GPU e relieves the User da una complicata implementazione. Lo User deve semplicemnte specificare, attraverso la scrittura di appositi CUDA kernel, quali sono le operazioni da eseguire per la generazione dei raggi, nel caso di intersezione dei raggi con primitive, nel caso di miss, cioè nel caso in cui i raggi non intersechino primitive etc.

## OptiX initializaion

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
}
```

Before initializing OptiX, the presence of a GPU is checked by the `cudaGetDeviceCount()` function. If no GPU is found, an error message is emitted and the program stops.
The actual function initializing OptiX is `optixInit()`. The decorating functions `gpuErrchk()` and `optixAssert()` are error checking functions reported below, respectively:

``` c++
void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) { exit(code); }
	}
}

extern "C" void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }
```

``` c++
extern "C" void optixAssert(OptixResult res)
{
    if (res != OPTIX_SUCCESS)                                          
    {                                                                 
        fprintf(stderr, "Optix call failed with code %d (line %d)\n", res, __LINE__); 
        exit(2);                                                      
    }                                                                 
}
```

They emit an error and the program stops if the execution of the decorated function is unsuccessful.

## Creating the OptiX pipeline and generating the rays

Lo scopo di questo esempio è illustrare esclusivamente la generazione dei raggi. Non ci sono oggetti da intersecare e l'immagine è fictitiously costruita direttamente durante la fase di ray launching. 

La OptiX pipeline è gestita attraverso la `renderer` class che espone tre metodi pubblici:

``` c++
public:
    // --- Constructor
    renderer();

    // --- Render one frame
    void render();

    // --- Resize buffer
    void resize(const int2& newSize);

    thrust::device_vector<uint32_t>  d_colorBuffer;
```

ossia il costruttore `renderer()`, il metodo `render()` che materialmente costruisce l'immagine e il metodo `resize()` che effettua il resizing del `d_colorBuffer` gestito, come detto, dalla `thrust` library.

Il costruttore, di seguito riportato, effettua la creazione della OptiX pipeline e della *shader binding table* (SBT).

``` c++
renderer::renderer() {

   h_launchParams.resize(1);
   d_launchParams.reserve(1);
	
   initOptix();

   createContext();

   createModule();

   createRaygenPrograms();

   createMissPrograms();
	 
   createHitgroupPrograms();

   createPipeline();

   buildSBT(); }
```

`h_launchParams` and `d_launchParams` sono un `thrust` `host_vector` and `device_vector` usati per passare alla GPU informazioni utili al rendering, come sarà chiaro tra breve.

Il metodo `initOptix()` effettua l'OptiX initialization, come già fatto nell'esempio precedente, e non verrà ulteriormente commentato.

On the other side, il metodo `createContext()` crea l'OptiX context nel seguente modo:

``` c++
void renderer::createContext()
{
	const int deviceID = 0;
	gpuErrchk(cudaSetDevice(deviceID));
	gpuErrchk(cudaStreamCreate(&stream));

	gpuErrchk(cudaGetDeviceProperties(&deviceProps, deviceID));

	CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
	if (cuRes != CUDA_SUCCESS) fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

	optixAssert(optixDeviceContextCreate(cudaContext, 0, &optixContext));
	optixAssert(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4)); }
```

In altre parole, in questo esempio si considera a single-GPU running e l'esecuzione di OptiX è agganciata alla GPU number `0`. Viene dunque selezionata la GPU number `0` e creato uno stream all'interno del quale dovrà avvenire l'esecuzione delle primitive di OptiX. 

Successivamente, tramite la primitiva `cuCtxGetCurrent` del CUDA driver, il CUDA context viene immagazzinato all'interno della variabile `cudaContext`. The CUDA context, indeed, holds all the management data to control and use the device. For instance, it holds the list of allocated memory, the loaded modules that contain device code, the mapping between CPU and GPU memory for zero copy, etc. Una volta fatto questo, è possibile agganciare l'OptiX context `optixContext` al CUDA context `cudaContext` tramite la primitiva `optixDeviceContextCreate`. Infine, l'OptiX context è agganciata ad una callback function to communicate various messages tramite la primitiva `optixDeviceContextSetLogCallback`.



## CUDA interoperability
