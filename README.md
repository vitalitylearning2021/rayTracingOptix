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

Al contrario di coloro che sono esperti di computer graphics, utilizzeremo un linguaggio meno tecnico e più fisico, in modo che il materiale sia accessibile ad una platea quanto più ampia possibile.

## Getting started with OptiX

[AGGIUNGERE OPTIX PROGRAMMING GUIDE]

The core OptiX 7 API is header only. The include directory contains everything needed to access the OptiX API core functions. The OptiX 7 headers along with the CUDA toolkit is everything needed to develop GPU accelerated ray tracing algorithms with OptiX 7. To account for the OptiX include directory under Visual Studio, add `C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.2.0\include` to the VC++ include directories under `Configuration Properties`.

OptiX consente il tracciamento dei raggi in maniera particolarmente efficiente ed efficace. Il motore di tracciamento di OptiX è basato sull'uso della Boundary Volume Hierarchy (BVH) data structure [RIF] accelerata su GPU e relieves the User da una complicata implementazione thereof. Lo User deve semplicemnte specificare, attraverso la scrittura di appositi CUDA kernel, quali sono le operazioni da eseguire per la generazione dei raggi, nel caso di intersezione dei raggi con primitive, nel caso di miss, cioè nel caso in cui i raggi non intersechino primitive etc.

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

### Initialization

Il metodo `initOptix()` effettua l'OptiX initialization, come già fatto nell'esempio precedente, e non verrà ulteriormente commentato.

### Context creation

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

### Module creation

Il metodo `createModule()`, riportato di seguito, specifica le informazione necessarie alla compilazione dei kernel che rappresentano le operazioni da effettuare su GPU per la generazione dei raggi, la loro riflessione, rifrazione etc., effettua la compilazione di tali kernel e li associa all'OptiX context. Negli altri metodi descritti di sotto, al modulo verranno associati anche i kernel compilati che specificano le operazioni menzionate assegnando loro uno specifico significato tra: generazione dei raggi, ray miss, closest hit, any hit. E' da notare che, in questo semplice esempio, viene gestito un solo modulo, ma OptiX può gestire anche più moduli perché nella scena possono essere presenti materiali differenti, da associare a moduli differenti, per i quali le leggi di riflessione e rifrazione possono essere differenti.

``` c++
void renderer::createModule() {

    moduleCompileOptions.maxRegisterCount					= 50;
    moduleCompileOptions.optLevel						= OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel						= OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags				= OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur					= false;
    pipelineCompileOptions.numPayloadValues					= 2;
    pipelineCompileOptions.numAttributeValues					= 2;
    pipelineCompileOptions.exceptionFlags					= OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName 			= "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth						= 2;

    	const std::string nvccCommand = "nvcc C:/Users/angel/source/repos/optixPipelineAndRayGen/optixPipelineAndRayGen/devicePrograms.cu -ptx -allow-unsupported-compiler -I \"C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.2.0/include\"";
    system(nvccCommand.c_str());
    std::ifstream input("C:/Users/angel/source/repos/optixPipelineAndRayGen/optixPipelineAndRayGen/devicePrograms.ptx");
    std::stringstream ptxCode;
    while (input >> ptxCode.rdbuf());
    const std::string ptxCode2 = ptxCode.str();

    char log[2048];
    size_t sizeof_log = sizeof(log);
    optixAssert(optixModuleCreateFromPTX(optixContext, &moduleCompileOptions, &pipelineCompileOptions,
		ptxCode2.c_str(), ptxCode2.size(), log, &sizeof_log, &module));
    if (sizeof_log > 1) PRINT(log);
}
```

`moduleCompileOptions` consente di set informazioni sulla compilazione dei kernel, mentre `pipelineCompileOptions` consente di set informazioni sulla compilazione del motore di tracciamento dei raggi interno ad OptiX. The link option `maxTraceDepth` specifies the maximum recursion depth setting for recursive ray tracing, not used here.

E' da notare che, tra i campi da settare riguardanti le informazioni di compilazione del motore di ray tracing, vi è anche `pipelineLaunchParamsVariableName` che, nel caso in esame, prende il nome di `"optixLaunchParams"`. `pipelineLaunchParamsVariableName` consente infatti di specificare il nome di una variabile utilizzata per lo scambio dati con i già menzionati User-defined kernels per la gestione dei raggi. Nella fattispecie, `optixLaunchParams` è una variabile di tipo `LaunchParams`, ossia del seguente `struct`:

``` c++
struct LaunchParams {
    int       frameID{ 0 };
    unsigned int *colorBuffer;
    int2      fbSize; };
```

Il campo `frameID` di questo `struct` verrà rigidamente fissato a `0` in questo esempio e, di fatto, rimarrà inutilizzato. Al contrario, `colorBuffer` sarà un puntatore all'area dati in cui verrà memorizzata l'immagine generata. Infine, `fbSize` conterrà le dimensioni dell'immagine.

I kernel per la generazione dei raggi e la loro riflessione/rifrazione vengono associati al modulo OptiX una volta compilati in linguaggio PTX. I kernel CUDA vengono compilati to PTX creando una stringa di comando e utilizzando una chiamata di sistema tramite `system`. Il file PTX così generato viene caricato in una `std::string` `ptxCode2`. L'associazione dei programmi PTX con le modalità di compilazione dei kernel e del motore di tracciamento dei raggi avviene grazie alla primitiva `optixModuleCreateFromPTX`.

### Raygen program creation

E' ora giunto il momento di associare, ai vari kernel compilati in linguaggio PTX, un significato. Il metodo `createRaygenPrograms()` associa al kernel scritto per la generazione dei raggi il suo significato di generazione dei raggi:

``` c++
void renderer::createRaygenPrograms() {

	raygenPGs.resize(1);

	OptixProgramGroupOptions pgOptions		= {};
	OptixProgramGroupDesc pgDesc			= {};
	pgDesc.kind					= OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module				= module;
	pgDesc.raygen.entryFunctionName			= "__raygen__renderFrame";

	OptixProgramGroup				raypg;
	char log[2048];
	size_t sizeof_log				= sizeof(log);
	optixAssert(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &raygenPGs[0]));
	if (sizeof_log > 1) PRINT(log);
}
```

In particolare, il nome del kernel è `__raygen__renderFrame` e questo viene identificato come `OPTIX_PROGRAM_GROUP_KIND_RAYGEN`. 

La `__global__` function `__raygen__renderFrame`, contenuta nel file `devicePrograms.cu`, è di seguito riportata

``` c++
extern "C" __global__ void __raygen__renderFrame() {
    
    if (optixLaunchParams.frameID == 0 && optixGetLaunchIndex().x == 0 && optixGetLaunchIndex().y == 0) {
            printf("OptiX 7 ray generation program. The image is %ix%i-sized\n",
                optixLaunchParams.fbSize.x,
                optixLaunchParams.fbSize.y);
        }

        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;

        const int r = (ix % 256);
        const int g = (iy % 256);
        const int b = ((ix + iy) % 256);

        const unsigned int rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

        const unsigned int fbIndex = ix + iy * optixLaunchParams.fbSize.x;
        optixLaunchParams.colorBuffer[fbIndex] = rgba;
    }
```

`optixGetLaunchIndex().x` and `optixGetLaunchIndex().y` sono sostanzialmente i thread IDs lungo x ed y. Dunque, in accordo al kernel, il solo thread con coordinate `(0,0)` invia un messaggio. Tutti i kernel riempiono poi il pixel ad essi assegnato dell'immagine con un colore dipendente dal loro thread ID.

### Miss and hitgroup programs creation

I metodi `createMissPrograms()` e `createHitgroupPrograms()` operano in maniera del tutto simile a `createRaygenPrograms()` in quanto associano i kernel `__miss__radiance`, `__closesthit__radiance` e `__anyhit__radiance` ai loro significati di operazioni da effettuare in caso di mancata intersezione, di intersezione più vicina o di intersezione generica. Le istruzioni che differenziano `createMissPrograms()` e `createHitgroupPrograms()` e che operano a questo scopo sono le seguenti:


``` c++
pgDesc.kind					= OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
pgDesc.hitgroup.moduleCH			= module;
pgDesc.hitgroup.entryFunctionNameCH		= "__closesthit__radiance";
pgDesc.hitgroup.moduleAH			= module;
pgDesc.hitgroup.entryFunctionNameAH		= "__anyhit__radiance";
```

Riportiamo di seguito anche i kernel contenuti nel file `devicePrograms.cu`:

``` c++
extern "C" __global__ void __closesthit__radiance() { }

extern "C" __global__ void __anyhit__radiance() { }

extern "C" __global__ void __miss__radiance() { }
```

Essi sono vuoti perché nessuna operazione è prevista nel caso di miss o hit.

### Pipeline creation

Dopo aver associato ciascun kernel al suo modulo corrispondente, tutti i kernel vengono inseriti in una *pipeline* `pipeline` dal metodo `createPipeline()`:

``` c++
void renderer::createPipeline() {

	thrust::host_vector<OptixProgramGroup> programGroups;
	for (auto pg : raygenPGs)		programGroups.push_back(pg);
	for (auto pg : missPGs)			programGroups.push_back(pg);
	for (auto pg : hitgroupPGs)		programGroups.push_back(pg);

	char log[2048];
	size_t sizeof_log				= sizeof(log);
	optixAssert(optixPipelineCreate(optixContext, &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(),
									(int)programGroups.size(), log, &sizeof_log, &pipeline));
	if (sizeof_log > 1) PRINT(log);

	optixAssert(optixPipelineSetStackSize(pipeline, 2 * 1024, 2 * 1024, 2 * 1024, 1));
	if (sizeof_log > 1) PRINT(log);
}
```

L'obiettivo viene raggiunto tramite la primitiva `optixPipelineCreate`. Ancora una volta, per evitare l'uso del `CUDABuffer`, viene sfruttato un `thrust::host_vector`. La primitiva `optixPipelineSetStackSize` fissa infine the stack size for the considered pipeline.

### Shader binding table creation

When a ray hits an object in the scene, the ray tracer needs some way to determine which kernel to call to perform intersection tests, reflections or refractions. The Shader Binding Table (SBT) is a lookup table providing this information. It associates each geometry in the scene with a set of kernel function handles and parameters for these
functions. Each set of function handles and parameters is referred to as a shader record.

In questo esempio specifico, non vi sono oggetti in quanto ci limitiamo al solo lancio dei raggi. Riportiamo di seguito il metodo `buildSBT()` senza fornire ulteriori dettagli:

``` c++
void renderer::buildSBT() {

	// --- Raygen records
	thrust::host_vector<RaygenRecord> raygenRecords;
	for (int i = 0; i < raygenPGs.size(); i++) {
		RaygenRecord rec;
		optixAssert(optixSbtRecordPackHeader(raygenPGs[i], &rec));
		rec.data = nullptr; /* for now ... */
		raygenRecords.push_back(rec);
	}
	// --- Copy-assignment can also be used, as can thrust::copy
	raygenRecordsBuffer		= raygenRecords;
	sbt.raygenRecord		= (CUdeviceptr)thrust::raw_pointer_cast(raygenRecordsBuffer.data());

	// --- Miss records
	thrust::host_vector<MissRecord> missRecords;
	for (int i = 0; i < missPGs.size(); i++) {
		MissRecord rec;
		optixAssert(optixSbtRecordPackHeader(missPGs[i], &rec));
		rec.data = nullptr; /* for now ... */
		missRecords.push_back(rec);
	}
	missRecordsBuffer				= missRecords;
	sbt.missRecordBase				= (CUdeviceptr)thrust::raw_pointer_cast(missRecordsBuffer.data());
	sbt.missRecordStrideInBytes		= sizeof(MissRecord);
	sbt.missRecordCount				= (int)missRecords.size();

	// --- Hitgroup records
	int numObjects = 1;
	thrust::host_vector<HitgroupRecord> hitgroupRecords;
	for (int i = 0; i < numObjects; i++) {
		int objectType = 0;
		HitgroupRecord rec;
		optixAssert(optixSbtRecordPackHeader(hitgroupPGs[objectType], &rec));
		rec.objectID = i;
		hitgroupRecords.push_back(rec);
	}
	hitgroupRecordsBuffer			= hitgroupRecords;
	sbt.hitgroupRecordBase			= (CUdeviceptr)thrust::raw_pointer_cast(hitgroupRecordsBuffer.data());
	sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sbt.hitgroupRecordCount			= (int)hitgroupRecords.size();
}
```

### Rendering

Il metodo che effettua materialmente la creazione dell'immagine (rendering) è il metodo `render()` di seguito riportato.

``` c++
void renderer::render()
{
	if (h_launchParams[0].fbSize.x == 0) return;

	d_launchParams = h_launchParams;
	h_launchParams[0].frameID++;

	optixAssert(optixLaunch(pipeline, stream, (CUdeviceptr)thrust::raw_pointer_cast(d_launchParams.data()),	d_launchParams.size() * sizeof(LaunchParams),
				&sbt, h_launchParams[0].fbSize.x, h_launchParams[0].fbSize.y, 1));

	gpuErrchk(cudaDeviceSynchronize());
}
```

Esenzialmente, esso si limita a lanciare la primitiva `optixLaunch()`. E' da notare che, tramite tale primitiva, è possibile passare i dati che i kernel (in questo caso, il solo kernel di generazione dei raggi) dovranno utilizzare tramite `d_launchParams`. La primitiva di lancio specifica anche il grigliato di lancio che, in questo caso, è `(h_launchParams[0].fbSize.x, h_launchParams[0].fbSize.y)`.

### Main function

La `main` function di seguito riportata definisce un'istanza della classe `renderer`, fissa le dimensioni dell'immagine e quindi del grigliato di lancio dei kernel coinvolti `(1200, 1024)`, dimensiona in accordo il buffer realizzato con la libreria `thrust`, lancia il metodo di rendering e salva l'immagine in formato `BMP` tramite la funzione `writeBmp()` non riportata qui.

``` c++
int main() {
	
	try {
		renderer sample;
		const int2 fbSize = make_int2(1200, 1024);
		sample.resize(fbSize);
		std::cout << GREEN << "Execution starts..." << std::endl << WHITE;
		sample.render();
		std::cout << GREEN << "Execution ends..." << std::endl << WHITE;
		thrust::host_vector<uint32_t> pixels = sample.d_colorBuffer;
		std::string path = IMAGE_PATH;
		writeBmp(path, fbSize.x, fbSize.y, pixels.data());
	}
	catch (std::runtime_error& e) {
		std::cout << RED << "FATAL ERROR: " << e.what() << std::endl << WHITE;
		exit(1);
	}

	return 0;
}
```

## CUDA interoperability
