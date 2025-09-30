#!/bin/bash

SOURCE_DIR="adGraph"
TARGET_ROOT="adGraph_hip"
# 设置最大并发进程数 (推荐 CPU 核心数的 1~2 倍)
MAX_THREADS=32

# -------------------------- 函数定义 --------------------------

# 定义核心的转换/复制函数，将被 xargs 并行调用
process_file() {
    local source_file_path="$1"
    
    # 忽略目录，只处理文件
    if [ ! -f "$source_file_path" ]; then
        return 0
    fi

    # 1. 获取文件的原始相对路径
    relative_path="${source_file_path#$SOURCE_DIR/}"
    
    # 2. 确定目标文件的新后缀名和操作类型 (转换或复制)
    extension="${source_file_path##*.}"
    target_ext=""
    operation="COPY"
    
    case "$extension" in
        # 需要 Hipify 转换的文件
        cpp|cu)
            target_ext="cpp"
            operation="HIP"
            ;;
        h|hxx|cuh)
            target_ext="h" # 头文件统一转为 .h
            operation="HIP"
            ;;
        # 其他文件（包括 Makefile, txt, log 等）直接复制，保留原始后缀
        *)
            target_ext="$extension"
            operation="COPY"
            ;;
    esac
    
    # 3. 确定目标文件路径
    if [ "$operation" == "HIP" ]; then
        # 移除原始后缀并添加新的目标后缀（用于转换）
        base_path_no_ext="${relative_path%.*}"
        target_relative_path="$base_path_no_ext.$target_ext"
    else
        # 保持原始相对路径（用于复制）
        target_relative_path="$relative_path"
    fi
    target_file_path="$TARGET_ROOT/$target_relative_path"
    
    # 4. 确保目标子目录存在
    target_dir=$(dirname "$target_file_path")
    mkdir -p "$target_dir"
    
    # 5. 执行操作
    if [ "$operation" == "HIP" ]; then
        echo "HIPIFY: $relative_path -> $target_relative_path (PID: $$)"
        hipify-perl "$source_file_path" > "$target_file_path"
        
        if [ $? -ne 0 ]; then
            echo "ERROR: hipify-perl failed for $relative_path"
            return 1
        else
            echo "DONE: $relative_path (Hipify)"
        fi
    else
        echo "COPY: $relative_path (PID: $$)"
        # 直接复制文件，保留原名
        cp -f "$source_file_path" "$target_file_path"
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to copy $relative_path"
            return 1
        else
            echo "DONE: $relative_path (Copy)"
        fi
    fi
    
    return 0
}

# 导出函数和变量，以便 xargs 调用的子 shell 可以访问
export -f process_file
export SOURCE_DIR TARGET_ROOT 

# -------------------------- 主逻辑 --------------------------

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' not found."
    exit 1
fi

echo "Starting HIPIFY/COPY process (Max Threads: $MAX_THREADS)..."
echo "--------------------------------------------------------"

# 查找 src_bak 中的所有文件 (-type f) 和目录 (-type d)
# find "$SOURCE_DIR" -print0 查找所有文件和目录
find "$SOURCE_DIR" -print0 | 
xargs -0 -P "$MAX_THREADS" -I {} bash -c 'process_file "$1"' _ {}

echo "--------------------------------------------------------"
echo "Conversion and Copy process finished."
