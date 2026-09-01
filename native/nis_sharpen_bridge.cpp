#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>

#include "NIS_Config.h"

using Microsoft::WRL::ComPtr;

namespace {

std::string Hex(unsigned long value) {
    std::ostringstream stream;
    stream << "0x" << std::hex << std::uppercase << value;
    return stream.str();
}

void WriteError(char* output, uint32_t capacity, const std::string& message) {
    if (!output || capacity == 0) return;
    const size_t count = std::min<size_t>(message.size(), capacity - 1);
    std::memcpy(output, message.data(), count);
    output[count] = '\0';
}

bool Check(HRESULT result, const char* action, std::string& error) {
    if (SUCCEEDED(result)) return true;
    error = std::string(action) + " failed: " + Hex(result);
    return false;
}

bool Sharpen(
    const uint8_t* rgba,
    uint32_t rgbaStride,
    uint32_t width,
    uint32_t height,
    float sharpness,
    const wchar_t* shaderPath,
    uint32_t gpuIndex,
    uint8_t* destination,
    uint32_t destinationStride,
    std::string& error) {
    if (!rgba || !destination || !shaderPath || width == 0 || height == 0 ||
        rgbaStride < width * 4 || destinationStride < width * 4) {
        error = "Invalid NIS image buffer, dimensions, stride, or shader path";
        return false;
    }
    if (!std::filesystem::is_regular_file(shaderPath)) {
        error = "NIS_Main.hlsl was not found";
        return false;
    }

    ComPtr<IDXGIFactory1> factory;
    if (!Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1", error)) return false;
    ComPtr<IDXGIAdapter1> adapter;
    if (factory->EnumAdapters1(gpuIndex, &adapter) == DXGI_ERROR_NOT_FOUND) {
        error = "Requested NIS GPU adapter was not found";
        return false;
    }

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL featureLevel{};
    if (!Check(
            D3D11CreateDevice(
                adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0,
                D3D11_SDK_VERSION, &device, &featureLevel, &context),
            "D3D11CreateDevice", error)) {
        return false;
    }

    NISOptimizer optimizer(false, NISGPUArchitecture::NVIDIA_Generic);
    const uint32_t blockWidth = optimizer.GetOptimalBlockWidth();
    const uint32_t blockHeight = optimizer.GetOptimalBlockHeight();
    const uint32_t threadGroupSize = optimizer.GetOptimalThreadGroupSize();
    const std::string blockWidthText = std::to_string(blockWidth);
    const std::string blockHeightText = std::to_string(blockHeight);
    const std::string threadGroupText = std::to_string(threadGroupSize);
    const D3D_SHADER_MACRO defines[] = {
        {"NIS_SCALER", "0"},
        {"NIS_HDR_MODE", "0"},
        {"NIS_BLOCK_WIDTH", blockWidthText.c_str()},
        {"NIS_BLOCK_HEIGHT", blockHeightText.c_str()},
        {"NIS_THREAD_GROUP_SIZE", threadGroupText.c_str()},
        {"NIS_CLAMP_OUTPUT", "1"},
        {nullptr, nullptr},
    };

    ComPtr<ID3DBlob> shaderBlob;
    ComPtr<ID3DBlob> compilerErrors;
    const HRESULT compileResult = D3DCompileFromFile(
        shaderPath, defines, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &shaderBlob, &compilerErrors);
    if (FAILED(compileResult)) {
        if (compilerErrors) {
            error.assign(
                static_cast<const char*>(compilerErrors->GetBufferPointer()),
                compilerErrors->GetBufferSize());
        } else {
            error = "D3DCompileFromFile failed: " + Hex(compileResult);
        }
        return false;
    }

    ComPtr<ID3D11ComputeShader> shader;
    if (!Check(
            device->CreateComputeShader(
                shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), nullptr, &shader),
            "CreateComputeShader", error)) {
        return false;
    }

    D3D11_TEXTURE2D_DESC textureDescription{};
    textureDescription.Width = width;
    textureDescription.Height = height;
    textureDescription.MipLevels = 1;
    textureDescription.ArraySize = 1;
    textureDescription.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    textureDescription.SampleDesc.Count = 1;
    textureDescription.Usage = D3D11_USAGE_DEFAULT;
    textureDescription.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA inputData{};
    inputData.pSysMem = rgba;
    inputData.SysMemPitch = rgbaStride;
    ComPtr<ID3D11Texture2D> inputTexture;
    if (!Check(
            device->CreateTexture2D(&textureDescription, &inputData, &inputTexture),
            "Create input texture", error)) {
        return false;
    }

    ComPtr<ID3D11ShaderResourceView> inputView;
    if (!Check(
            device->CreateShaderResourceView(inputTexture.Get(), nullptr, &inputView),
            "Create input shader-resource view", error)) {
        return false;
    }

    textureDescription.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    ComPtr<ID3D11Texture2D> outputTexture;
    if (!Check(
            device->CreateTexture2D(&textureDescription, nullptr, &outputTexture),
            "Create output texture", error)) {
        return false;
    }
    ComPtr<ID3D11UnorderedAccessView> outputView;
    if (!Check(
            device->CreateUnorderedAccessView(outputTexture.Get(), nullptr, &outputView),
            "Create output unordered-access view", error)) {
        return false;
    }

    NISConfig config{};
    sharpness = std::clamp(sharpness, 0.0f, 1.0f);
    if (!NVSharpenUpdateConfig(
            config, sharpness,
            0, 0, width, height, width, height,
            0, 0, NISHDRMode::None)) {
        error = "NVSharpenUpdateConfig rejected the image dimensions";
        return false;
    }
    D3D11_BUFFER_DESC bufferDescription{};
    bufferDescription.ByteWidth = sizeof(NISConfig);
    bufferDescription.Usage = D3D11_USAGE_DEFAULT;
    bufferDescription.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA bufferData{};
    bufferData.pSysMem = &config;
    ComPtr<ID3D11Buffer> constantBuffer;
    if (!Check(
            device->CreateBuffer(&bufferDescription, &bufferData, &constantBuffer),
            "Create NIS constant buffer", error)) {
        return false;
    }

    D3D11_SAMPLER_DESC samplerDescription{};
    samplerDescription.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    samplerDescription.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDescription.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDescription.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDescription.MaxLOD = D3D11_FLOAT32_MAX;
    ComPtr<ID3D11SamplerState> sampler;
    if (!Check(
            device->CreateSamplerState(&samplerDescription, &sampler),
            "Create NIS sampler", error)) {
        return false;
    }

    ID3D11ShaderResourceView* inputViews[] = {inputView.Get()};
    ID3D11UnorderedAccessView* outputViews[] = {outputView.Get()};
    ID3D11SamplerState* samplers[] = {sampler.Get()};
    ID3D11Buffer* buffers[] = {constantBuffer.Get()};
    context->CSSetShaderResources(0, 1, inputViews);
    context->CSSetUnorderedAccessViews(0, 1, outputViews, nullptr);
    context->CSSetSamplers(0, 1, samplers);
    context->CSSetConstantBuffers(0, 1, buffers);
    context->CSSetShader(shader.Get(), nullptr, 0);
    context->Dispatch(
        static_cast<uint32_t>(std::ceil(width / static_cast<float>(blockWidth))),
        static_cast<uint32_t>(std::ceil(height / static_cast<float>(blockHeight))), 1);

    ID3D11ShaderResourceView* nullSrv[] = {nullptr};
    ID3D11UnorderedAccessView* nullUav[] = {nullptr};
    context->CSSetShaderResources(0, 1, nullSrv);
    context->CSSetUnorderedAccessViews(0, 1, nullUav, nullptr);

    textureDescription.BindFlags = 0;
    textureDescription.Usage = D3D11_USAGE_STAGING;
    textureDescription.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> stagingTexture;
    if (!Check(
            device->CreateTexture2D(&textureDescription, nullptr, &stagingTexture),
            "Create staging texture", error)) {
        return false;
    }
    context->CopyResource(stagingTexture.Get(), outputTexture.Get());

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (!Check(
            context->Map(stagingTexture.Get(), 0, D3D11_MAP_READ, 0, &mapped),
            "Map NIS output", error)) {
        return false;
    }
    for (uint32_t row = 0; row < height; ++row) {
        std::memcpy(
            destination + row * destinationStride,
            static_cast<const uint8_t*>(mapped.pData) + row * mapped.RowPitch,
            width * 4);
    }
    context->Unmap(stagingTexture.Get(), 0);
    return true;
}

}  // namespace

extern "C" __declspec(dllexport) int bokujuu_nis_sharpen(
    const uint8_t* rgba,
    uint32_t rgbaStride,
    uint32_t width,
    uint32_t height,
    float sharpness,
    const wchar_t* shaderPath,
    uint32_t gpuIndex,
    uint8_t* output,
    uint32_t outputStride,
    char* error,
    uint32_t errorCapacity) {
    try {
        std::string message;
        if (!Sharpen(
                rgba, rgbaStride, width, height, sharpness, shaderPath, gpuIndex,
                output, outputStride, message)) {
            WriteError(error, errorCapacity, message);
            return 0;
        }
        return 1;
    } catch (const std::exception& exception) {
        WriteError(error, errorCapacity, exception.what());
        return 0;
    }
}
