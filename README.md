# bindbc-onnxruntime
This project provides dynamic bindings to the C API of [ONNX Runtime](https://github.com/microsoft/onnxruntime). The bindings are `@nogc` and `nothrow` compatible and can be compiled for compatibility with `-betterC`.

## Usage
By default, `bindbc-onnxruntime` is configured to compile as a dynamic binding that is not `-betterC` compatible. The dynamic binding has no link-time dependency on the library, so the shared library must be manually loaded at runtime.

To use ONNX Runtime, add `bindbc-onnxruntime` as a dependency to your project's package config file. For example, the following is configured to use ONNX Runtime as a dynamic binding that is not `-betterC` compatible:

__dub.json__
```
dependencies {
    "bindbc-onnxruntime": "~>1.2.0",
}
```

__dub.sdl__
```
dependency "bindbc-onnxruntime" version="~>1.2.0"
```

### Enable support versions

Support for ONNX Runtime versions can be configured at compile time by adding the appropriate version to a versions directive in your package configuration file (or on the command line if you are building with a tool other than dub).

`bindbc-onnxruntime` defines a D version identifier for each ONNX Runtime version. The following table lists each identifier and the ONNX Runtime versions they enable.

| Version | Version ID     | `ONNXRuntimeSupport` Member |
|---------|----------------|-----------------------------|
| 1.2     | ONNXRuntime_12 | `ONNXRuntimeSupport.v12`    |

## TODO

- Support GPU binding
- more test