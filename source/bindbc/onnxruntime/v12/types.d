module bindbc.onnxruntime.v12.types;

import core.stdc.stddef;

extern (C):

// This value is used in structures passed to ORT so that a newer version of ORT will still work with them
enum ORT_API_VERSION = 2;

// SAL2 Definitions

// Define ORT_DLL_IMPORT if your program is dynamically linked to Ort.
// dllexport is not used, we use a .def file.

alias ORTCHAR_T = wchar_t;

// Any pointer marked with _In_ or _Out_, cannot be NULL.

// Windows users should use unicode paths when possible to bypass the MAX_PATH limitation
// Every pointer marked with _In_ or _Out_, cannot be NULL. Caller should ensure that.
// for ReleaseXXX(...) functions, they can accept NULL pointer.

// Copied from TensorProto::DataType
// Currently, Ort doesn't support complex64, complex128, bfloat16 types
enum ONNXTensorElementDataType
{
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED = 0,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1, // maps to c type float
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 = 2, // maps to c type uint8_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 = 3, // maps to c type int8_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 = 4, // maps to c type uint16_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 = 5, // maps to c type int16_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 = 6, // maps to c type int32_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7, // maps to c type int64_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING = 8, // maps to c++ type std::string
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL = 9,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE = 11, // maps to c type double
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 = 12, // maps to c type uint32_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 = 13, // maps to c type uint64_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 = 14, // complex with float32 real and imaginary components
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 = 15, // complex with float64 real and imaginary components
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16 // Non-IEEE floating-point format based on IEEE754 single-precision
}

// Synced with onnx TypeProto oneof
enum ONNXType
{
    ONNX_TYPE_UNKNOWN = 0,
    ONNX_TYPE_TENSOR = 1,
    ONNX_TYPE_SEQUENCE = 2,
    ONNX_TYPE_MAP = 3,
    ONNX_TYPE_OPAQUE = 4,
    ONNX_TYPE_SPARSETENSOR = 5
}

enum OrtLoggingLevel
{
    ORT_LOGGING_LEVEL_VERBOSE = 0,
    ORT_LOGGING_LEVEL_INFO = 1,
    ORT_LOGGING_LEVEL_WARNING = 2,
    ORT_LOGGING_LEVEL_ERROR = 3,
    ORT_LOGGING_LEVEL_FATAL = 4
}

enum OrtErrorCode
{
    ORT_OK = 0,
    ORT_FAIL = 1,
    ORT_INVALID_ARGUMENT = 2,
    ORT_NO_SUCHFILE = 3,
    ORT_NO_MODEL = 4,
    ORT_ENGINE_ERROR = 5,
    ORT_RUNTIME_EXCEPTION = 6,
    ORT_INVALID_PROTOBUF = 7,
    ORT_MODEL_LOADED = 8,
    ORT_NOT_IMPLEMENTED = 9,
    ORT_INVALID_GRAPH = 10,
    ORT_EP_FAIL = 11
}

// __VA_ARGS__ on Windows and Linux are different

// Used in *.cc files. Almost as same as ORT_API_STATUS, except without ORT_MUST_USE_RESULT

//  ORT_API(void, OrtRelease##X, _Frees_ptr_opt_ Ort##X* input);

// The actual types defined have an Ort prefix
struct OrtEnv;
struct OrtStatus; // nullptr for Status* indicates success
struct OrtMemoryInfo;
struct OrtSession; //Don't call OrtReleaseSession from Dllmain (because session owns a thread pool)
struct OrtValue;
struct OrtRunOptions;
struct OrtTypeInfo;
struct OrtTensorTypeAndShapeInfo;
struct OrtSessionOptions;
struct OrtCustomOpDomain;
struct OrtMapTypeInfo;
struct OrtSequenceTypeInfo;
struct OrtModelMetadata;

// When passing in an allocator to any ORT function, be sure that the allocator object
// is not destroyed until the last allocated object using it is freed.
struct OrtAllocator
{
    uint version_; // Initialize to ORT_API_VERSION
    void* function(OrtAllocator* this_, size_t size) Alloc;
    void function(OrtAllocator* this_, void* p) Free;
    const(OrtMemoryInfo)* function(const(OrtAllocator)* this_) Info;
}

alias OrtLoggingFunction = void function(void* param, OrtLoggingLevel severity,
        const(char)* category, const(char)* logid,
        const(char)* code_location, const(char)* message);

// Set Graph optimization level.
// Refer https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Graph_Optimizations.md
// for in-depth undersrtanding of Graph Optimizations in ORT
enum GraphOptimizationLevel
{
    ORT_DISABLE_ALL = 0,
    ORT_ENABLE_BASIC = 1,
    ORT_ENABLE_EXTENDED = 2,
    ORT_ENABLE_ALL = 99
}

enum ExecutionMode
{
    ORT_SEQUENTIAL = 0,
    ORT_PARALLEL = 1
}

struct OrtKernelInfo;
struct OrtKernelContext;

enum OrtAllocatorType
{
    Invalid = -1,
    OrtDeviceAllocator = 0,
    OrtArenaAllocator = 1
}

/**
 * memory types for allocator, exec provider specific types should be extended in each provider
 * Whenever this struct is updated, please also update the MakeKey function in onnxruntime/core/framework/execution_provider.cc
*/
enum OrtMemType
{
    OrtMemTypeCPUInput = -2, // Any CPU memory used by non-CPU execution provider
    OrtMemTypeCPUOutput = -1, // CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
    OrtMemTypeCPU = OrtMemTypeCPUOutput, // temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
    OrtMemTypeDefault = 0 // the default allocator for execution provider
}

struct OrtApiBase
{
    const(OrtApi)* function(uint version_) GetApi; // Pass in ORT_API_VERSION
    // nullptr will be returned if the version is unsupported, for example when using a runtime older than this header file

    const(char)* GetVersionString;
}

struct OrtApi
{
    /**
    * \param msg A null-terminated string. Its content will be copied into the newly created OrtStatus
    */
    OrtStatus* function(OrtErrorCode code, const(char)* msg) CreateStatus;

    OrtErrorCode function(const(OrtStatus)* status) GetErrorCode;

    /**
    * \param status must not be NULL
    * \return The error message inside the `status`. Do not free the returned value.
    */
    const(char)* function(const(OrtStatus)* status) GetErrorMessage;

    /**
       * \param out Should be freed by `OrtReleaseEnv` after use
       */
    OrtStatus* function(OrtLoggingLevel default_logging_level, const(char)* logid, OrtEnv** out_) CreateEnv;

    /**
     * \param out Should be freed by `OrtReleaseEnv` after use
     */
    OrtStatus* function(OrtLoggingFunction logging_function, void* logger_param,
            OrtLoggingLevel default_warning_level, const(char)* logid, OrtEnv** out_) CreateEnvWithCustomLogger;

    // Platform telemetry events are on by default since they are lightweight.  You can manually turn them off.
    OrtStatus* function(const(OrtEnv)* env) EnableTelemetryEvents;
    OrtStatus* function(const(OrtEnv)* env) DisableTelemetryEvents;

    // TODO: document the path separator convention? '/' vs '\'
    // TODO: should specify the access characteristics of model_path. Is this read only during the
    // execution of CreateSession, or does the OrtSession retain a handle to the file/directory
    // and continue to access throughout the OrtSession lifetime?
    //  What sort of access is needed to model_path : read or read/write?
    OrtStatus* function(const(OrtEnv)* env, const(wchar_t)* model_path,
            const(OrtSessionOptions)* options, OrtSession** out_) CreateSession;

    OrtStatus* function(const(OrtEnv)* env, const(void)* model_data,
            size_t model_data_length, const(OrtSessionOptions)* options, OrtSession** out_) CreateSessionFromArray;

    OrtStatus* function(OrtSession* sess, const(OrtRunOptions)* run_options,
            const(char*)* input_names, const(OrtValue*)* input, size_t input_len,
            const(char*)* output_names, size_t output_names_len, OrtValue** output) Run;

    /**
      * \return A pointer of the newly created object. The pointer should be freed by OrtReleaseSessionOptions after use
      */
    OrtStatus* function(OrtSessionOptions** options) CreateSessionOptions;

    // Set filepath to save optimized model after graph level transformations.
    OrtStatus* function(OrtSessionOptions* options, const(wchar_t)* optimized_model_filepath) SetOptimizedModelFilePath;

    // create a copy of an existing OrtSessionOptions
    OrtStatus* function(const(OrtSessionOptions)* in_options, OrtSessionOptions** out_options) CloneSessionOptions;

    // Controls whether you want to execute operators in your graph sequentially or in parallel. Usually when the model
    // has many branches, setting this option to ExecutionMode.ORT_PARALLEL will give you better performance.
    // See [docs/ONNX_Runtime_Perf_Tuning.md] for more details.
    OrtStatus* function(OrtSessionOptions* options, ExecutionMode execution_mode) SetSessionExecutionMode;

    // Enable profiling for this session.
    OrtStatus* function(OrtSessionOptions* options, const(wchar_t)* profile_file_prefix) EnableProfiling;
    OrtStatus* function(OrtSessionOptions* options) DisableProfiling;

    // Enable the memory pattern optimization.
    // The idea is if the input shapes are the same, we could trace the internal memory allocation
    // and generate a memory pattern for future request. So next time we could just do one allocation
    // with a big chunk for all the internal memory allocation.
    // Note: memory pattern optimization is only available when SequentialExecution enabled.
    OrtStatus* function(OrtSessionOptions* options) EnableMemPattern;
    OrtStatus* function(OrtSessionOptions* options) DisableMemPattern;

    // Enable the memory arena on CPU
    // Arena may pre-allocate memory for future usage.
    // set this option to false if you don't want it.
    OrtStatus* function(OrtSessionOptions* options) EnableCpuMemArena;
    OrtStatus* function(OrtSessionOptions* options) DisableCpuMemArena;

    // < logger id to use for session output
    OrtStatus* function(OrtSessionOptions* options, const(char)* logid) SetSessionLogId;

    // < applies to session load, initialization, etc
    OrtStatus* function(OrtSessionOptions* options, int session_log_verbosity_level) SetSessionLogVerbosityLevel;
    OrtStatus* function(OrtSessionOptions* options, int session_log_severity_level) SetSessionLogSeverityLevel;

    OrtStatus* function(OrtSessionOptions* options, GraphOptimizationLevel graph_optimization_level) SetSessionGraphOptimizationLevel;

    // Sets the number of threads used to parallelize the execution within nodes
    // A value of 0 means ORT will pick a default
    OrtStatus* function(OrtSessionOptions* options, int intra_op_num_threads) SetIntraOpNumThreads;

    // Sets the number of threads used to parallelize the execution of the graph (across nodes)
    // If sequential execution is enabled this value is ignored
    // A value of 0 means ORT will pick a default
    OrtStatus* function(OrtSessionOptions* options, int inter_op_num_threads) SetInterOpNumThreads;

    /*
    Create a custom op domain. After all sessions using it are released, call OrtReleaseCustomOpDomain
    */
    OrtStatus* function(const(char)* domain, OrtCustomOpDomain** out_) CreateCustomOpDomain;

    /*
       * Add custom ops to the OrtCustomOpDomain
       *  Note: The OrtCustomOp* pointer must remain valid until the OrtCustomOpDomain using it is released
      */
    OrtStatus* function(OrtCustomOpDomain* custom_op_domain, OrtCustomOp* op) CustomOpDomain_Add;

    /*
       * Add a custom op domain to the OrtSessionOptions
       *  Note: The OrtCustomOpDomain* must not be deleted until the sessions using it are released
      */
    OrtStatus* function(OrtSessionOptions* options, OrtCustomOpDomain* custom_op_domain) AddCustomOpDomain;

    /*
       * Loads a DLL named 'library_path' and looks for this entry point:
       *		OrtStatus* RegisterCustomOps(OrtSessionOptions * options, const OrtApiBase* api);
       * It then passes in the provided session options to this function along with the api base.
       * The handle to the loaded library is returned in library_handle. It can be freed by the caller after all sessions using the passed in
       * session options are destroyed, or if an error occurs and it is non null.
    */
    OrtStatus* function(OrtSessionOptions* options, const(char)* library_path,
            void** library_handle) RegisterCustomOpsLibrary;

    /**
      * To use additional providers, you must build ORT with the extra providers enabled. Then call one of these
      * functions to enable them in the session:
      *   OrtSessionOptionsAppendExecutionProvider_CPU
      *   OrtSessionOptionsAppendExecutionProvider_CUDA
      *   OrtSessionOptionsAppendExecutionProvider_<remaining providers...>
      * The order they care called indicates the preference order as well. In other words call this method
      * on your most preferred execution provider first followed by the less preferred ones.
      * If none are called Ort will use its internal CPU execution provider.
      */

    OrtStatus* function(const(OrtSession)* sess, size_t* out_) SessionGetInputCount;
    OrtStatus* function(const(OrtSession)* sess, size_t* out_) SessionGetOutputCount;
    OrtStatus* function(const(OrtSession)* sess, size_t* out_) SessionGetOverridableInitializerCount;

    /**
     * \param out  should be freed by OrtReleaseTypeInfo after use
     */
    OrtStatus* function(const(OrtSession)* sess, size_t index, OrtTypeInfo** type_info) SessionGetInputTypeInfo;

    /**
     * \param out  should be freed by OrtReleaseTypeInfo after use
     */
    OrtStatus* function(const(OrtSession)* sess, size_t index, OrtTypeInfo** type_info) SessionGetOutputTypeInfo;

    /**
    * \param out  should be freed by OrtReleaseTypeInfo after use
    */
    OrtStatus* function(const(OrtSession)* sess, size_t index, OrtTypeInfo** type_info) SessionGetOverridableInitializerTypeInfo;

    /**
     * \param value  is set to a null terminated string allocated using 'allocator'. The caller is responsible for freeing it.
     */
    OrtStatus* function(const(OrtSession)* sess, size_t index,
            OrtAllocator* allocator, char** value) SessionGetInputName;

    OrtStatus* function(const(OrtSession)* sess, size_t index,
            OrtAllocator* allocator, char** value) SessionGetOutputName;

    OrtStatus* function(const(OrtSession)* sess, size_t index,
            OrtAllocator* allocator, char** value) SessionGetOverridableInitializerName;

    /**
     * \return A pointer to the newly created object. The pointer should be freed by OrtReleaseRunOptions after use
     */
    OrtStatus* function(OrtRunOptions** out_) CreateRunOptions;

    OrtStatus* function(OrtRunOptions* options, int value) RunOptionsSetRunLogVerbosityLevel;
    OrtStatus* function(OrtRunOptions* options, int value) RunOptionsSetRunLogSeverityLevel;
    OrtStatus* function(OrtRunOptions*, const(char)* run_tag) RunOptionsSetRunTag;

    OrtStatus* function(const(OrtRunOptions)* options, int* out_) RunOptionsGetRunLogVerbosityLevel;
    OrtStatus* function(const(OrtRunOptions)* options, int* out_) RunOptionsGetRunLogSeverityLevel;
    OrtStatus* function(const(OrtRunOptions)*, const(char*)* out_) RunOptionsGetRunTag;

    // Set a flag so that ALL incomplete OrtRun calls that are using this instance of OrtRunOptions
    // will exit as soon as possible.
    OrtStatus* function(OrtRunOptions* options) RunOptionsSetTerminate;
    // Unset the terminate flag to enable this OrtRunOptions instance being used in new OrtRun calls.
    OrtStatus* function(OrtRunOptions* options) RunOptionsUnsetTerminate;

    /**
     * Create a tensor from an allocator. OrtReleaseValue will also release the buffer inside the output value
     * \param out Should be freed by calling OrtReleaseValue
     * \param type must be one of TENSOR_ELEMENT_DATA_TYPE_xxxx
     */
    OrtStatus* function(OrtAllocator* allocator, const(long)* shape,
            size_t shape_len, ONNXTensorElementDataType type, OrtValue** out_) CreateTensorAsOrtValue;

    /**
     * Create a tensor with user's buffer. You can fill the buffer either before calling this function or after.
     * p_data is owned by caller. OrtReleaseValue won't release p_data.
     * \param out Should be freed by calling OrtReleaseValue
     */
    OrtStatus* function(const(OrtMemoryInfo)* info, void* p_data, size_t p_data_len,
            const(long)* shape, size_t shape_len, ONNXTensorElementDataType type, OrtValue** out_) CreateTensorWithDataAsOrtValue;

    /**
     * \Sets *out to 1 iff an OrtValue is a tensor, 0 otherwise
     */
    OrtStatus* function(const(OrtValue)* value, int* out_) IsTensor;

    // This function doesn't work with string tensor
    // this is a no-copy method whose pointer is only valid until the backing OrtValue is free'd.
    OrtStatus* function(OrtValue* value, void** out_) GetTensorMutableData;

    /**
       * \param value A tensor created from OrtCreateTensor... function.
       * \param s each A string array. Each string in this array must be null terminated.
       * \param s_len length of s
       */
    OrtStatus* function(OrtValue* value, const(char*)* s, size_t s_len) FillStringTensor;

    /**
       * \param value A tensor created from OrtCreateTensor... function.
       * \param len total data length, not including the trailing '\0' chars.
       */
    OrtStatus* function(const(OrtValue)* value, size_t* len) GetStringTensorDataLength;

    /**
       * \param s string contents. Each string is NOT null-terminated.
       * \param value A tensor created from OrtCreateTensor... function.
       * \param s_len total data length, get it from OrtGetStringTensorDataLength
       */
    OrtStatus* function(const(OrtValue)* value, void* s, size_t s_len,
            size_t* offsets, size_t offsets_len) GetStringTensorContent;

    /**
       * Don't free the 'out' value
       */
    OrtStatus* function(const(OrtTypeInfo)*, const(OrtTensorTypeAndShapeInfo*)* out_) CastTypeInfoToTensorInfo;

    /**
       * Return OnnxType from OrtTypeInfo
       */
    OrtStatus* function(const(OrtTypeInfo)*, ONNXType* out_) GetOnnxTypeFromTypeInfo;

    /**
       * The 'out' value should be released by calling OrtReleaseTensorTypeAndShapeInfo
       */
    OrtStatus* function(OrtTensorTypeAndShapeInfo** out_) CreateTensorTypeAndShapeInfo;

    OrtStatus* function(OrtTensorTypeAndShapeInfo*, ONNXTensorElementDataType type) SetTensorElementType;

    /**
    * \param info Created from CreateTensorTypeAndShapeInfo() function
    * \param dim_values An array with length of `dim_count`. Its elements can contain negative values.
    * \param dim_count length of dim_values
    */
    OrtStatus* function(OrtTensorTypeAndShapeInfo* info, const(long)* dim_values, size_t dim_count) SetDimensions;

    OrtStatus* function(const(OrtTensorTypeAndShapeInfo)*, ONNXTensorElementDataType* out_) GetTensorElementType;
    OrtStatus* function(const(OrtTensorTypeAndShapeInfo)* info, size_t* out_) GetDimensionsCount;
    OrtStatus* function(const(OrtTensorTypeAndShapeInfo)* info,
            long* dim_values, size_t dim_values_length) GetDimensions;
    OrtStatus* function(const(OrtTensorTypeAndShapeInfo)* info,
            const(char*)* dim_params, size_t dim_params_length) GetSymbolicDimensions;

    /**
    * Return the number of elements specified by the tensor shape.
    * Return a negative value if unknown (i.e., any dimension is negative.)
    * e.g.
    * [] -> 1
    * [1,3,4] -> 12
    * [2,0,4] -> 0
    * [-1,3,4] -> -1
    */
    OrtStatus* function(const(OrtTensorTypeAndShapeInfo)* info, size_t* out_) GetTensorShapeElementCount;

    /**
    * \param out Should be freed by OrtReleaseTensorTypeAndShapeInfo after use
    */
    OrtStatus* function(const(OrtValue)* value, OrtTensorTypeAndShapeInfo** out_) GetTensorTypeAndShape;

    /**
    * Get the type information of an OrtValue
    * \param value
    * \param out The returned value should be freed by OrtReleaseTypeInfo after use
    */
    OrtStatus* function(const(OrtValue)* value, OrtTypeInfo** out_) GetTypeInfo;

    OrtStatus* function(const(OrtValue)* value, ONNXType* out_) GetValueType;

    OrtStatus* function(const(char)* name1, OrtAllocatorType type, int id1,
            OrtMemType mem_type1, OrtMemoryInfo** out_) CreateMemoryInfo;

    /**
    * Convenience function for special case of CreateMemoryInfo, for the CPU allocator. Uses name = "Cpu" and id = 0.
    */
    OrtStatus* function(OrtAllocatorType type, OrtMemType mem_type1, OrtMemoryInfo** out_) CreateCpuMemoryInfo;

    /**
    * Test if two memory info are equal
    * \Sets 'out' to 0 if equal, -1 if not equal
    */
    OrtStatus* function(const(OrtMemoryInfo)* info1, const(OrtMemoryInfo)* info2, int* out_) CompareMemoryInfo;

    /**
    * Do not free the returned value
    */
    OrtStatus* function(const(OrtMemoryInfo)* ptr, const(char*)* out_) MemoryInfoGetName;
    OrtStatus* function(const(OrtMemoryInfo)* ptr, int* out_) MemoryInfoGetId;
    OrtStatus* function(const(OrtMemoryInfo)* ptr, OrtMemType* out_) MemoryInfoGetMemType;
    OrtStatus* function(const(OrtMemoryInfo)* ptr, OrtAllocatorType* out_) MemoryInfoGetType;

    OrtStatus* function(OrtAllocator* ptr, size_t size, void** out_) AllocatorAlloc;
    OrtStatus* function(OrtAllocator* ptr, void* p) AllocatorFree;
    OrtStatus* function(const(OrtAllocator)* ptr, const(OrtMemoryInfo*)* out_) AllocatorGetInfo;

    // The returned pointer doesn't have to be freed.
    // Always returns the same instance on every invocation.
    OrtStatus* function(OrtAllocator** out_) GetAllocatorWithDefaultOptions;

    // Override symbolic dimensions with actual values if known at session initialization time to enable
    // optimizations that can take advantage of fixed values (such as memory planning, etc)
    OrtStatus* function(OrtSessionOptions* options, const(char)* symbolic_dim, long dim_override) AddFreeDimensionOverride;

    /**
     * APIs to support non-tensor types - map and sequence.
     * Currently only the following types are supported
     * Note: the following types should be kept in sync with data_types.h
     * Map types
     * =========
     * std::map<std::string, std::string>
     * std::map<std::string, int64_t>
     * std::map<std::string, float>
     * std::map<std::string, double>
     * std::map<int64_t, std::string>
     * std::map<int64_t, int64_t>
     * std::map<int64_t, float>
     * std::map<int64_t, double>
     *
     * Sequence types
     * ==============
     * std::vector<std::string>
     * std::vector<int64_t>
     * std::vector<float>
     * std::vector<double>
     * std::vector<std::map<std::string, float>>
     * std::vector<std::map<int64_t, float>
     */

    /**
     * If input OrtValue represents a map, you need to retrieve the keys and values
     * separately. Use index=0 to retrieve keys and index=1 to retrieve values.
     * If input OrtValue represents a sequence, use index to retrieve the index'th element
     * of the sequence.
     */
    OrtStatus* function(const(OrtValue)* value, int index,
            OrtAllocator* allocator, OrtValue** out_) GetValue;

    /**
     * Returns 2 for type map and N for sequence where N is the number of elements
     * in the sequence.
     */
    OrtStatus* function(const(OrtValue)* value, size_t* out_) GetValueCount;

    /**
     * To construct a map, use num_values = 2 and 'in' should be an arrary of 2 OrtValues
     * representing keys and values.
     * To construct a sequence, use num_values = N where N is the number of the elements in the
     * sequence. 'in' should be an arrary of N OrtValues.
     * \value_type should be either map or sequence.
     */
    OrtStatus* function(const(OrtValue*)* in_, size_t num_values,
            ONNXType value_type, OrtValue** out_) CreateValue;

    /**
       * Construct OrtValue that contains a value of non-standard type created for
       * experiments or while awaiting standardization. OrtValue in this case would contain
       * an internal representation of the Opaque type. Opaque types are distinguished between
       * each other by two strings 1) domain and 2) type name. The combination of the two
       * must be unique, so the type representation is properly identified internally. The combination
       * must be properly registered from within ORT at both compile/run time or by another API.
       *
       * To construct the OrtValue pass domain and type names, also a pointer to a data container
       * the type of which must be know to both ORT and the client program. That data container may or may
       * not match the internal representation of the Opaque type. The sizeof(data_container) is passed for
       * verification purposes.
       *
       * \domain_name - domain name for the Opaque type, null terminated.
       * \type_name   - type name for the Opaque type, null terminated.
       * \data_contianer - data to populate OrtValue
       * \data_container_size - sizeof() of the data container. Must match the sizeof() of the expected
       *                    data_container size internally.
       */
    OrtStatus* function(const(char)* domain_name, const(char)* type_name,
            const(void)* data_container, size_t data_container_size, OrtValue** out_) CreateOpaqueValue;

    /**
       * Fetch data from an OrtValue that contains a value of non-standard type created for
       * experiments or while awaiting standardization.
       * \domain_name - domain name for the Opaque type, null terminated.
       * \type_name   - type name for the Opaque type, null terminated.
       * \data_contianer - data to populate OrtValue
       * \data_container_size - sizeof() of the data container. Must match the sizeof() of the expected
       *                    data_container size internally.
       */

    OrtStatus* function(const(char)* domain_name, const(char)* type_name,
            const(OrtValue)* in_, void* data_container, size_t data_container_size) GetOpaqueValue;

    OrtStatus* function(const(OrtKernelInfo)* info, const(char)* name, float* out_) KernelInfoGetAttribute_float;
    OrtStatus* function(const(OrtKernelInfo)* info, const(char)* name, long* out_) KernelInfoGetAttribute_int64;
    OrtStatus* function(const(OrtKernelInfo)* info, const(char)* name, char* out_, size_t* size) KernelInfoGetAttribute_string;

    OrtStatus* function(const(OrtKernelContext)* context, size_t* out_) KernelContext_GetInputCount;
    OrtStatus* function(const(OrtKernelContext)* context, size_t* out_) KernelContext_GetOutputCount;
    OrtStatus* function(const(OrtKernelContext)* context, size_t index, const(OrtValue*)* out_) KernelContext_GetInput;
    OrtStatus* function(OrtKernelContext* context, size_t index,
            const(long)* dim_values, size_t dim_count, OrtValue** out_) KernelContext_GetOutput;

    void function(OrtEnv* input) ReleaseEnv;
    void function(OrtStatus* input) ReleaseStatus; // nullptr for Status* indicates success
    void function(OrtMemoryInfo* input) ReleaseMemoryInfo;
    void function(OrtSession* input) ReleaseSession; //Don't call OrtReleaseSession from Dllmain (because session owns a thread pool)
    void function(OrtValue* input) ReleaseValue;
    void function(OrtRunOptions* input) ReleaseRunOptions;
    void function(OrtTypeInfo* input) ReleaseTypeInfo;
    void function(OrtTensorTypeAndShapeInfo* input) ReleaseTensorTypeAndShapeInfo;
    void function(OrtSessionOptions* input) ReleaseSessionOptions;
    void function(OrtCustomOpDomain* input) ReleaseCustomOpDomain;

    // End of Version 1 - DO NOT MODIFY ABOVE (see above text for more information)

    // Version 2 - In development, feel free to add/remove/rearrange here

    /**
        * GetDenotationFromTypeInfo
    	 * This api augments OrtTypeInfo to return denotations on the type.
    	 * This is used by WinML to determine if an input/output is intended to be an Image or a Tensor.
        */
    OrtStatus* function(const(OrtTypeInfo)*, const char** denotation, size_t* len) GetDenotationFromTypeInfo;

    // OrtTypeInfo Casting methods

    /**
        * CastTypeInfoToMapTypeInfo
    	 * This api augments OrtTypeInfo to return an OrtMapTypeInfo when the type is a map.
    	 * The OrtMapTypeInfo has additional information about the map's key type and value type.
    	 * This is used by WinML to support model reflection APIs.
    	 *
    	 * Don't free the 'out' value
        */
    OrtStatus* function(const(OrtTypeInfo)* type_info, const(OrtMapTypeInfo*)* out_) CastTypeInfoToMapTypeInfo;

    /**
        * CastTypeInfoToSequenceTypeInfo
    	 * This api augments OrtTypeInfo to return an OrtSequenceTypeInfo when the type is a sequence.
    	 * The OrtSequenceTypeInfo has additional information about the sequence's element type.
        * This is used by WinML to support model reflection APIs.
    	 *
    	 * Don't free the 'out' value
        */
    OrtStatus* function(const(OrtTypeInfo)* type_info, const(OrtSequenceTypeInfo*)* out_) CastTypeInfoToSequenceTypeInfo;

    // OrtMapTypeInfo Accessors

    /**
        * GetMapKeyType
    	 * This api augments get the key type of a map. Key types are restricted to being scalar types and use ONNXTensorElementDataType.
    	 * This is used by WinML to support model reflection APIs.
        */
    OrtStatus* function(const(OrtMapTypeInfo)* map_type_info, ONNXTensorElementDataType* out_) GetMapKeyType;

    /**
        * GetMapValueType
    	 * This api augments get the value type of a map.
        */
    OrtStatus* function(const(OrtMapTypeInfo)* map_type_info, OrtTypeInfo** type_info) GetMapValueType;

    // OrtSequenceTypeInfo Accessors

    /**
        * GetSequenceElementType
    	 * This api augments get the element type of a sequence.
    	 * This is used by WinML to support model reflection APIs.
        */
    OrtStatus* function(const(OrtSequenceTypeInfo)* sequence_type_info, OrtTypeInfo** type_info) GetSequenceElementType;

    void function(OrtMapTypeInfo* input) ReleaseMapTypeInfo;
    void function(OrtSequenceTypeInfo* input) ReleaseSequenceTypeInfo;

    /**
     * \param out is set to a null terminated string allocated using 'allocator'. The caller is responsible for freeing it.
     * Profiling is turned ON automatically if enabled for the particular session by invoking EnableProfiling()
     * on the SessionOptions instance used to create the session.
     */
    OrtStatus* function(OrtSession* sess, OrtAllocator* allocator, char** out_) SessionEndProfiling;

    /**
     * \param out is a pointer to the newly created object. The pointer should be freed by calling ReleaseModelMetadata after use.
     */
    OrtStatus* function(const(OrtSession)* sess, OrtModelMetadata** out_) SessionGetModelMetadata;

    /**
     * \param value  is set to a null terminated string allocated using 'allocator'. The caller is responsible for freeing it.
     */
    OrtStatus* function(const(OrtModelMetadata)* model_metadata,
            OrtAllocator* allocator, char** value) ModelMetadataGetProducerName;

    OrtStatus* function(const(OrtModelMetadata)* model_metadata,
            OrtAllocator* allocator, char** value) ModelMetadataGetGraphName;

    OrtStatus* function(const(OrtModelMetadata)* model_metadata,
            OrtAllocator* allocator, char** value) ModelMetadataGetDomain;

    OrtStatus* function(const(OrtModelMetadata)* model_metadata,
            OrtAllocator* allocator, char** value) ModelMetadataGetDescription;

    /**
     * \param value  is set to a null terminated string allocated using 'allocator'. The caller is responsible for freeing it.
     * 'value' will be a nullptr if the given key is not found in the custom metadata map.
     */
    OrtStatus* function(const(OrtModelMetadata)* model_metadata,
            OrtAllocator* allocator, const(char)* key, char** value) ModelMetadataLookupCustomMetadataMap;

    OrtStatus* function(const(OrtModelMetadata)* model_metadata, long* value) ModelMetadataGetVersion;

    void function(OrtModelMetadata* input) ReleaseModelMetadata;
}

/*
 * Steps to use a custom op:
 *   1 Create an OrtCustomOpDomain with the domain name used by the custom ops
 *   2 Create an OrtCustomOp structure for each op and add them to the domain
 *   3 Call OrtAddCustomOpDomain to add the custom domain of ops to the session options
*/
alias OrtCustomOpApi = OrtApi;

/*
 * The OrtCustomOp structure defines a custom op's schema and its kernel callbacks. The callbacks are filled in by
 * the implementor of the custom op.
*/
struct OrtCustomOp
{
    uint version_; // Initialize to ORT_API_VERSION

    // This callback creates the kernel, which is a user defined parameter that is passed to the Kernel* callbacks below.
    void* function(OrtCustomOp* op, const(OrtApi)* api, const(OrtKernelInfo)* info) CreateKernel;

    // Returns the name of the op
    const(char)* function(OrtCustomOp* op) GetName;

    // Returns the type of the execution provider, return nullptr to use CPU execution provider
    const(char)* function(OrtCustomOp* op) GetExecutionProviderType;

    // Returns the count and types of the input & output tensors
    ONNXTensorElementDataType function(OrtCustomOp* op, size_t index) GetInputType;
    size_t function(OrtCustomOp* op) GetInputTypeCount;
    ONNXTensorElementDataType function(OrtCustomOp* op, size_t index) GetOutputType;
    size_t function(OrtCustomOp* op) GetOutputTypeCount;

    // Op kernel callbacks
    void function(void* op_kernel, OrtKernelContext* context) KernelCompute;
    void function(void* op_kernel) KernelDestroy;
}

/*
 * END EXPERIMENTAL
*/
