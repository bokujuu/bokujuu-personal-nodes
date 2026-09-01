#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_params.h"

using Microsoft::WRL::ComPtr;

namespace {

using InitProjectFn = NVSDK_NGX_Result(NVSDK_CONV*)(
    const char*, NVSDK_NGX_EngineType, const char*, const wchar_t*, ID3D11Device*,
    NVSDK_NGX_Version, const NVSDK_NGX_FeatureCommonInfo*);
using AllocateParametersFn = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter**);
using DestroyParametersFn = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter*);
using CreateFeatureFn = NVSDK_NGX_Result(NVSDK_CONV*)(
    ID3D11DeviceContext*, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
using EvaluateFeatureFn = NVSDK_NGX_Result(NVSDK_CONV*)(
    ID3D11DeviceContext*, const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*,
    PFN_NVSDK_NGX_ProgressCallback);
using ReleaseFeatureFn = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);
// nvngx.dll exports the driver ABI. Its shutdown export has an implementation
// result out-parameter; the public one-argument function is a static-wrapper ABI.
using ShutdownFn = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D11Device*, int*);

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

template <typename T>
T Export(HMODULE module, const char* name) {
    return reinterpret_cast<T>(GetProcAddress(module, name));
}

class Session {
public:
    uint32_t inputWidth = 0;
    uint32_t inputHeight = 0;
    uint32_t outputWidth = 0;
    uint32_t outputHeight = 0;
    HMODULE ngxCore = nullptr;
    InitProjectFn initialize = nullptr;
    AllocateParametersFn allocateParameters = nullptr;
    DestroyParametersFn destroyParameters = nullptr;
    CreateFeatureFn createFeature = nullptr;
    EvaluateFeatureFn evaluateFeature = nullptr;
    ReleaseFeatureFn releaseFeature = nullptr;
    ShutdownFn shutdown = nullptr;
    NVSDK_NGX_Parameter* parameters = nullptr;
    NVSDK_NGX_Handle* feature = nullptr;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11Texture2D> color;
    ComPtr<ID3D11Texture2D> depth;
    ComPtr<ID3D11Texture2D> motion;
    ComPtr<ID3D11Texture2D> output;
    ComPtr<ID3D11Texture2D> staging;

    ~Session() { Destroy(); }

    bool Initialize(
        const wchar_t* runtimePath,
        const wchar_t* corePath,
        const char* projectId,
        uint32_t inWidth,
        uint32_t inHeight,
        uint32_t outWidth,
        uint32_t outHeight,
        int quality,
        uint32_t gpuIndex,
        std::string& error) {
        if (!runtimePath || !corePath || !projectId || inWidth < 64 || inHeight < 32 ||
            outWidth < 64 || outHeight < 32) {
            error = "Invalid DLSS runtime path, project ID, or image dimensions";
            return false;
        }
        const std::filesystem::path runtime(runtimePath);
        if (!std::filesystem::is_regular_file(runtime)) {
            error = "DLSS runtime does not exist";
            return false;
        }

        inputWidth = inWidth;
        inputHeight = inHeight;
        outputWidth = outWidth;
        outputHeight = outHeight;

        ComPtr<IDXGIFactory1> factory;
        HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
        if (FAILED(hr)) {
            error = "CreateDXGIFactory1 failed: " + Hex(hr);
            return false;
        }
        ComPtr<IDXGIAdapter1> adapter;
        uint32_t nvidiaIndex = 0;
        for (uint32_t index = 0;; ++index) {
            ComPtr<IDXGIAdapter1> candidate;
            if (factory->EnumAdapters1(index, &candidate) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 adapterDescription{};
            candidate->GetDesc1(&adapterDescription);
            if (adapterDescription.VendorId != 0x10DE ||
                (adapterDescription.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                continue;
            }
            if (nvidiaIndex++ == gpuIndex) {
                adapter = candidate;
                break;
            }
        }
        if (!adapter) {
            error = "Requested NVIDIA GPU index is not available: " + std::to_string(gpuIndex);
            return false;
        }
        const D3D_FEATURE_LEVEL levels[] = {
            D3D_FEATURE_LEVEL_12_1,
            D3D_FEATURE_LEVEL_12_0,
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
        };
        D3D_FEATURE_LEVEL selected{};
        hr = D3D11CreateDevice(
            adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, levels, ARRAYSIZE(levels),
            D3D11_SDK_VERSION, &device, &selected, &context);
        if (FAILED(hr)) {
            error = "D3D11CreateDevice failed: " + Hex(hr);
            return false;
        }

        ngxCore = LoadLibraryW(corePath);
        if (!ngxCore) {
            error = "Could not load NVIDIA NGX core: " + Hex(GetLastError());
            return false;
        }
        initialize = Export<InitProjectFn>(ngxCore, "NVSDK_NGX_D3D11_Init_ProjectID");
        allocateParameters = Export<AllocateParametersFn>(ngxCore, "NVSDK_NGX_D3D11_AllocateParameters");
        destroyParameters = Export<DestroyParametersFn>(ngxCore, "NVSDK_NGX_D3D11_DestroyParameters");
        createFeature = Export<CreateFeatureFn>(ngxCore, "NVSDK_NGX_D3D11_CreateFeature");
        evaluateFeature = Export<EvaluateFeatureFn>(ngxCore, "NVSDK_NGX_D3D11_EvaluateFeature");
        releaseFeature = Export<ReleaseFeatureFn>(ngxCore, "NVSDK_NGX_D3D11_ReleaseFeature");
        shutdown = Export<ShutdownFn>(ngxCore, "NVSDK_NGX_D3D11_Shutdown1");
        if (!initialize || !allocateParameters || !destroyParameters || !createFeature ||
            !evaluateFeature || !releaseFeature || !shutdown) {
            error = "Installed NVIDIA NGX core is missing required D3D11 exports";
            return false;
        }

        const std::wstring runtimeDirectory = runtime.parent_path().wstring();
        const wchar_t* paths[] = {runtimeDirectory.c_str()};
        NVSDK_NGX_FeatureCommonInfo common{};
        common.PathListInfo.Path = paths;
        common.PathListInfo.Length = 1;
        const std::filesystem::path cache = runtime.parent_path() / L"ngx_cache";
        std::error_code filesystemError;
        std::filesystem::create_directories(cache, filesystemError);
        const auto initResult = initialize(
            projectId,
            NVSDK_NGX_ENGINE_TYPE_CUSTOM,
            "Bokujuu-ComfyUI-1.0",
            cache.c_str(),
            device.Get(),
            NVSDK_NGX_Version_API,
            &common);
        if (initResult != NVSDK_NGX_Result_Success) {
            error = "NVSDK_NGX_D3D11_Init_ProjectID failed: " + Hex(initResult);
            return false;
        }
        const auto parameterResult = allocateParameters(&parameters);
        if (parameterResult != NVSDK_NGX_Result_Success || !parameters) {
            error = "NVSDK_NGX_D3D11_AllocateParameters failed: " + Hex(parameterResult);
            return false;
        }

        if (!CreateTextures(error)) return false;

        parameters->Set(NVSDK_NGX_Parameter_Width, inputWidth);
        parameters->Set(NVSDK_NGX_Parameter_Height, inputHeight);
        parameters->Set(NVSDK_NGX_Parameter_OutWidth, outputWidth);
        parameters->Set(NVSDK_NGX_Parameter_OutHeight, outputHeight);
        parameters->Set(NVSDK_NGX_Parameter_PerfQualityValue, quality);
        const int flags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                          NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
        parameters->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, flags);
        parameters->Set(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0u);
        const auto createResult = createFeature(
            context.Get(), NVSDK_NGX_Feature_SuperSampling, parameters, &feature);
        if (createResult != NVSDK_NGX_Result_Success || !feature) {
            error = "DLSS Super Resolution feature creation failed: " + Hex(createResult) +
                    ". Check NGX logs, scale/quality, driver, and runtime compatibility.";
            return false;
        }
        return true;
    }

    bool CreateTextures(std::string& error) {
        D3D11_TEXTURE2D_DESC description{};
        description.Width = inputWidth;
        description.Height = inputHeight;
        description.MipLevels = 1;
        description.ArraySize = 1;
        description.SampleDesc.Count = 1;
        description.Usage = D3D11_USAGE_DYNAMIC;
        description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        description.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        HRESULT hr = device->CreateTexture2D(&description, nullptr, &color);
        description.Format = DXGI_FORMAT_R32_FLOAT;
        if (SUCCEEDED(hr)) hr = device->CreateTexture2D(&description, nullptr, &depth);
        description.Format = DXGI_FORMAT_R16G16_FLOAT;
        if (SUCCEEDED(hr)) hr = device->CreateTexture2D(&description, nullptr, &motion);
        if (FAILED(hr)) {
            error = "Could not allocate DLSS input textures: " + Hex(hr);
            return false;
        }

        description = {};
        description.Width = outputWidth;
        description.Height = outputHeight;
        description.MipLevels = 1;
        description.ArraySize = 1;
        description.SampleDesc.Count = 1;
        description.Usage = D3D11_USAGE_DEFAULT;
        description.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        hr = device->CreateTexture2D(&description, nullptr, &output);
        description.Usage = D3D11_USAGE_STAGING;
        description.BindFlags = 0;
        description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (SUCCEEDED(hr)) hr = device->CreateTexture2D(&description, nullptr, &staging);
        if (FAILED(hr)) {
            error = "Could not allocate DLSS output textures: " + Hex(hr);
            return false;
        }

        D3D11_MAPPED_SUBRESOURCE mapped{};
        hr = context->Map(motion.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
        if (FAILED(hr)) {
            error = "Could not initialize the motion-vector texture: " + Hex(hr);
            return false;
        }
        for (uint32_t row = 0; row < inputHeight; ++row) {
            std::memset(static_cast<uint8_t*>(mapped.pData) + row * mapped.RowPitch, 0,
                        inputWidth * sizeof(uint16_t) * 2);
        }
        context->Unmap(motion.Get(), 0);
        return true;
    }

    bool Process(
        const uint8_t* rgba,
        uint32_t rgbaStride,
        const float* depthValues,
        uint32_t depthStride,
        bool reset,
        uint8_t* destination,
        uint32_t destinationStride,
        std::string& error) {
        if (!rgba || !depthValues || !destination) {
            error = "DLSS input, depth, and output buffers are required";
            return false;
        }
        D3D11_MAPPED_SUBRESOURCE mapped{};
        HRESULT hr = context->Map(color.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
        if (FAILED(hr)) {
            error = "Could not map the DLSS color texture: " + Hex(hr);
            return false;
        }
        for (uint32_t row = 0; row < inputHeight; ++row) {
            std::memcpy(static_cast<uint8_t*>(mapped.pData) + row * mapped.RowPitch,
                        rgba + row * rgbaStride, inputWidth * 4);
        }
        context->Unmap(color.Get(), 0);

        hr = context->Map(depth.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
        if (FAILED(hr)) {
            error = "Could not map the DLSS depth texture: " + Hex(hr);
            return false;
        }
        for (uint32_t row = 0; row < inputHeight; ++row) {
            std::memcpy(static_cast<uint8_t*>(mapped.pData) + row * mapped.RowPitch,
                        reinterpret_cast<const uint8_t*>(depthValues) + row * depthStride,
                        inputWidth * sizeof(float));
        }
        context->Unmap(depth.Get(), 0);

        parameters->Set(NVSDK_NGX_Parameter_Color, color.Get());
        parameters->Set(NVSDK_NGX_Parameter_Output, output.Get());
        parameters->Set(NVSDK_NGX_Parameter_Depth, depth.Get());
        parameters->Set(NVSDK_NGX_Parameter_MotionVectors, motion.Get());
        parameters->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, 0.0f);
        parameters->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, 0.0f);
        parameters->Set(NVSDK_NGX_Parameter_Sharpness, 0.0f);
        parameters->Set(NVSDK_NGX_Parameter_Reset, reset ? 1 : 0);
        parameters->Set(NVSDK_NGX_Parameter_MV_Scale_X, static_cast<float>(inputWidth));
        parameters->Set(NVSDK_NGX_Parameter_MV_Scale_Y, static_cast<float>(inputHeight));
        parameters->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, inputWidth);
        parameters->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, inputHeight);
        parameters->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
        parameters->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
        const auto evaluateResult = evaluateFeature(context.Get(), feature, parameters, nullptr);
        if (evaluateResult != NVSDK_NGX_Result_Success) {
            error = "DLSS Super Resolution evaluation failed: " + Hex(evaluateResult);
            return false;
        }

        context->CopyResource(staging.Get(), output.Get());
        hr = context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) {
            error = "Could not read the DLSS output texture: " + Hex(hr);
            return false;
        }
        for (uint32_t row = 0; row < outputHeight; ++row) {
            std::memcpy(destination + row * destinationStride,
                        static_cast<const uint8_t*>(mapped.pData) + row * mapped.RowPitch,
                        outputWidth * 4);
        }
        context->Unmap(staging.Get(), 0);
        return true;
    }

    void Destroy() {
        if (feature && releaseFeature) {
            releaseFeature(feature);
            feature = nullptr;
        }
        if (parameters && destroyParameters) {
            destroyParameters(parameters);
            parameters = nullptr;
        }
        if (device && shutdown) {
            int implementationResult = 0;
            shutdown(device.Get(), &implementationResult);
        }
        staging.Reset();
        output.Reset();
        motion.Reset();
        depth.Reset();
        color.Reset();
        context.Reset();
        device.Reset();
        if (ngxCore) {
            FreeLibrary(ngxCore);
            ngxCore = nullptr;
        }
    }
};

}  // namespace

extern "C" __declspec(dllexport) void* bokujuu_dlss_create(
    const wchar_t* runtimePath,
    const wchar_t* corePath,
    const char* projectId,
    uint32_t inputWidth,
    uint32_t inputHeight,
    uint32_t outputWidth,
    uint32_t outputHeight,
    int quality,
    uint32_t gpuIndex,
    char* error,
    uint32_t errorCapacity) {
    try {
        auto session = std::make_unique<Session>();
        std::string message;
        if (!session->Initialize(runtimePath, corePath, projectId, inputWidth, inputHeight,
                                 outputWidth, outputHeight, quality, gpuIndex, message)) {
            WriteError(error, errorCapacity, message);
            return nullptr;
        }
        return session.release();
    } catch (const std::exception& exception) {
        WriteError(error, errorCapacity, exception.what());
        return nullptr;
    }
}

extern "C" __declspec(dllexport) int bokujuu_dlss_process(
    void* handle,
    const uint8_t* rgba,
    uint32_t rgbaStride,
    const float* depth,
    uint32_t depthStride,
    int reset,
    uint8_t* output,
    uint32_t outputStride,
    char* error,
    uint32_t errorCapacity) {
    if (!handle) {
        WriteError(error, errorCapacity, "DLSS session is null");
        return 0;
    }
    try {
        std::string message;
        if (!static_cast<Session*>(handle)->Process(
                rgba, rgbaStride, depth, depthStride, reset != 0, output, outputStride,
                message)) {
            WriteError(error, errorCapacity, message);
            return 0;
        }
        return 1;
    } catch (const std::exception& exception) {
        WriteError(error, errorCapacity, exception.what());
        return 0;
    }
}

extern "C" __declspec(dllexport) void bokujuu_dlss_destroy(void* handle) {
    delete static_cast<Session*>(handle);
}
