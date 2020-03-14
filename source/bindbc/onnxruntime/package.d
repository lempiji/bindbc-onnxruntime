module bindbc.onnxruntime;

public import bindbc.onnxruntime.config;

import bindbc.loader;

version (BindBC_Static)
{
    version = BindONNXRuntime_Static;
}
else
{
    version = BindONNXRuntime_Dynamic;
}

version (ONNXRuntime_12)
{
    public import bindbc.onnxruntime.v12;
}
else
{
    public import bindbc.onnxruntime.v12;
}
