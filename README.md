# LongContext-ICL-Annotation

Large Language Models Automatic Data Annotation under Long-Context Scenarios.

---

## Environment

```sh
conda create -n licl python=3.11 -y
conda activate licl
# 搭建 FlagScale 框架
git clone https://github.com/flagos-ai/FlagScale.git
pip install . --verbose
git clone https://github.com/flagos-ai/vllm-FL
cd vllm-FL
pip install -e .
# 安装其他依赖项
cd ..
pip install -r requirements.txt
```