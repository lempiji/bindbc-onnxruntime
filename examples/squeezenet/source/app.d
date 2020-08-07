import std;

import bindbc.onnxruntime;
import imageformats;

void main()
{
	if (!exists("squeezenet.onnx"))
	{
		writeln("Please download the model file of 'SqueezeNet' from https://github.com/onnx/models/tree/master/vision/classification/squeezenet");
		return;
	}

	const support = loadONNXRuntime();
	if (support == ONNXRuntimeSupport.noLibrary || support == ONNXRuntimeSupport.badLibrary)
	{
		writeln("Please download library from https://github.com/microsoft/onnxruntime/releases");
		return;
	}

	const(OrtApi)* ort = OrtGetApiBase().GetApi(ORT_API_VERSION);
	assert(ort);

	void checkStatus(OrtStatus* status)
	{
		if (status)
		{
			auto msg = ort.GetErrorMessage(status).to!string();
			stderr.writeln(msg);
			ort.ReleaseStatus(status);
			throw new Error(msg);
		}
	}

	OrtEnv* env;
	checkStatus(ort.CreateEnv(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE, "test", &env));
	scope (exit)
		ort.ReleaseEnv(env);

	OrtSessionOptions* session_options;
	checkStatus(ort.CreateSessionOptions(&session_options));
	scope (exit)
		ort.ReleaseSessionOptions(session_options);
	ort.SetIntraOpNumThreads(session_options, 1);

	ort.SetSessionGraphOptimizationLevel(session_options, GraphOptimizationLevel.ORT_ENABLE_BASIC);

	OrtSession* session;
	checkStatus(ort.CreateSession(env, "squeezenet.onnx", session_options, &session));
	scope (exit)
		ort.ReleaseSession(session);

	OrtAllocator* allocator;
	checkStatus(ort.GetAllocatorWithDefaultOptions(&allocator));

	size_t num_input_nodes;
	// print number of model input nodes
	checkStatus(ort.SessionGetInputCount(session, &num_input_nodes));

	char*[] input_node_names;
	long[] input_node_dims;
	// iterate over all input nodes
	foreach (i; 0 .. num_input_nodes)
	{
		// print input node names
		char* input_name;
		checkStatus(ort.SessionGetInputName(session, i, allocator, &input_name));
		input_node_names ~= input_name;
		writeln("Input ", i, " : name=", input_name.to!string());

		// print input node types
		OrtTypeInfo* typeinfo;
		checkStatus(ort.SessionGetInputTypeInfo(session, i, &typeinfo));
		scope (exit)
			ort.ReleaseTypeInfo(typeinfo);

		const OrtTensorTypeAndShapeInfo* tensor_info;
		checkStatus(ort.CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

		ONNXTensorElementDataType type;
		checkStatus(ort.GetTensorElementType(tensor_info, &type));
		writeln("Input ", i, " : type=", type);

		// print input shapes/dims
		size_t num_dims;
		checkStatus(ort.GetDimensionsCount(tensor_info, &num_dims));
		writeln("Input ", i, " : num_dims=", num_dims);

		input_node_dims.length = num_dims;
		ort.GetDimensions(tensor_info, input_node_dims.ptr, num_dims);
		foreach (j, dim; input_node_dims)
		{
			writeln("Input ", i, " : dim ", j, "=", dim);
		}
	}

	// Score the model using sample data, and inspect values

	size_t input_tensor_size = 224 * 224 * 3; // simplify ... using known dim values to calculate size
	// use OrtGetTensorShapeElementCount() to get official size!

	auto image = read_image("image.png", ColFmt.RGB);
	assert(image.pixels.length == input_tensor_size);
	auto input_tensor_values = preprocess(image);


	// create input tensor object from data values
	OrtMemoryInfo* memory_info;
	checkStatus(ort.CreateCpuMemoryInfo(OrtAllocatorType.OrtArenaAllocator,
			OrtMemType.OrtMemTypeDefault, &memory_info));
	scope (exit)
		ort.ReleaseMemoryInfo(memory_info);

	OrtValue* input_tensor;
	checkStatus(ort.CreateTensorWithDataAsOrtValue(memory_info,
			input_tensor_values.ptr, input_tensor_size * float.sizeof, input_node_dims.ptr,
			input_node_dims.length,
			ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
	scope (exit)
		ort.ReleaseValue(input_tensor);

	int is_tensor;
	checkStatus(ort.IsTensor(input_tensor, &is_tensor));
	assert(is_tensor);

	// score model & input tensor, get back output tensor
	const(char)*[] output_node_names = ["softmaxout_1".toStringz()];
	OrtValue* output_tensor;
	checkStatus(ort.Run(session, null, input_node_names.ptr, &input_tensor, 1,
			output_node_names.ptr, 1, &output_tensor));
	scope (exit)
		ort.ReleaseValue(output_tensor);
	checkStatus(ort.IsTensor(output_tensor, &is_tensor));
	assert(is_tensor);

	float* floatarr;
	checkStatus(ort.GetTensorMutableData(output_tensor, cast(void**)&floatarr));

	// check output
	auto result = new Tuple!(size_t, float)[](1000);
	foreach (i, v; floatarr[0 .. 1000])
	{
		result[i] = tuple(i, v);
	}
	result.sort!"a[1] > b[1]"();

	auto labels = loadLabels();
	foreach (x; result[0 .. 10])
	{
		writeln(labels[x[0]], " : ", x[1]);
	}
}

float[] preprocess(IFImage image)
{
	import mir.ndslice : sliced, flattened, transposed;

	auto result = new float[](image.pixels.length);
	auto data = image.pixels.sliced(image.w, image.h, image.c).transposed(2, 0, 1);
	result.sliced[] = data.flattened[];

	return result;
}

string[] loadLabels()
{
	return File("synset.txt").byLineCopy().array();
}
