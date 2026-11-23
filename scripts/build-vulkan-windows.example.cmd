@echo off
setlocal

REM Configure these paths for your machine before running:
if "%LLAMA_SRC%"=="" set "LLAMA_SRC=%CD%\llama.cpp"
if "%BUILD_DIR%"=="" set "BUILD_DIR=%CD%\build"
if "%VULKAN_SDK%"=="" set "VULKAN_SDK=C:\VulkanSDK\1.4.304.0"

set "PATH=%VULKAN_SDK%\Bin;%PATH%"

REM Initialize Visual Studio x64 Build Tools (adjust path for VS 2022/2026)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
) else (
    echo Could not locate vcvarsall.bat. Run this script from a Visual Studio x64 Developer Command Prompt.
)

cmake -S "%LLAMA_SRC%" -B "%BUILD_DIR%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_VULKAN=ON ^
  -DVulkan_GLSLC_EXECUTABLE="%VULKAN_SDK%\Bin\glslc.exe" ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF || exit /b 1

cmake --build "%BUILD_DIR%" --target llama-server --parallel || exit /b 1

echo Built: %BUILD_DIR%\bin\llama-server.exe
endlocal
