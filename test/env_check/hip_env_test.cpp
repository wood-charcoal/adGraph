#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>

/**
 * 最小化 HIP 测试程序。
 * 目标：仅测试 hipGetDeviceCount 函数是否能正确初始化和查询设备。
 * 如果该程序崩溃，则表明 ROCm/HIP 基础环境或驱动存在问题。
 */
int main() {
    int device_count = 0;
    
    // 1. 获取设备数量
    hipError_t err = hipGetDeviceCount(&device_count);

    if (err != hipSuccess) {
        std::cerr << "--- 错误信息 ---" << std::endl;
        std::cerr << "hipGetDeviceCount 失败: " 
                  << hipGetErrorString(err) 
                  << " (" << err << ")" << std::endl;
        std::cerr << "请检查您的 ROCm/HIP 驱动和环境变量配置。" << std::endl;
        return 1;
    }

    std::cout << "--- 设备查询结果 ---" << std::endl;
    std::cout << "成功找到 DCU/GPU 设备数量: " << device_count << std::endl;

    // 2. 如果找到设备，尝试设置并同步设备，进一步确认运行时环境稳定
    if (device_count > 0) {
        std::cout << "尝试设置设备 0 并同步..." << std::endl;
        
        // 设置设备 0
        err = hipSetDevice(0);
        if (err != hipSuccess) {
            std::cerr << "hipSetDevice(0) 失败: " << hipGetErrorString(err) << std::endl;
            return 1;
        }

        // 尝试设备同步，这是一个轻量级的测试
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::cerr << "hipDeviceSynchronize 失败: " << hipGetErrorString(err) << std::endl;
            // 注意: 在您的原始问题中，错误发生在 hipGetDeviceCount 阶段。
            // 如果能到这里，说明大部分初始化已完成。
        } else {
            std::cout << "设备同步成功。HIP 运行时似乎是稳定的。" << std::endl;
        }
    } else {
        std::cout << "未找到任何 DCU/GPU 设备。" << std::endl;
        std::cout << "请确保您的 Slurm 脚本正确请求了 GPU 资源。" << std::endl;
    }

    return 0;
}
