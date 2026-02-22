#!/bin/bash

# 设置文件名
input="note.md"
output="note.pdf"

echo "开始转换: $input → $output"

# 检查文件是否存在
if [ ! -f "$input" ]; then
    echo "错误: 文件 $input 不存在"
    exit 1
fi

# 检查图片路径
echo "检查文档中的图片引用..."
grep -E '!\[.*\]\(.*\)' "$input" || echo "未找到图片引用"

# 创建临时目录存放图片（如果需要）
mkdir -p images

# 转换命令 - 简化版
echo "正在生成PDF..."
pandoc "$input" -o "$output" \
  --pdf-engine=xelatex \
  -V mainfont="Noto Sans CJK SC" \
  -V geometry:margin=1in \
  --resource-path=.  # 重要：在当前目录查找图片

# 检查结果
if [ -f "$output" ]; then
    echo "✅ 转换成功!"
    echo "文件: $output"
    echo "大小: $(du -h "$output" | cut -f1)"
else
    echo "❌ 转换失败"
    echo "尝试备用方案..."
    
    # 备用方案：先转HTML再转PDF
    pandoc "$input" -o temp.html --self-contained
    if command -v wkhtmltopdf > /dev/null 2>&1; then
        wkhtmltopdf temp.html "$output"
        rm temp.html
        echo "使用wkhtmltopdf转换成功"
    fi
fi
