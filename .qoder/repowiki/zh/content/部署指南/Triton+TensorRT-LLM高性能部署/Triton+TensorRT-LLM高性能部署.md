# Triton+TensorRT-LLM高性能部署

<cite>
**本文档引用文件**   
- [Dockerfile.server](file://runtime/triton_trtllm/Dockerfile.server)
- [docker-compose.yml](file://runtime/triton_trtllm/docker-compose.yml)
- [run.sh](file://runtime/triton_trtllm/run.sh)
- [run_stepaudio2_dit_token2wav.sh](file://runtime/triton_trtllm/run_stepaudio2_dit_token2wav.sh)
- [requirements.txt](file://runtime/triton_trtllm/requirements.txt)
- [convert_checkpoint.py](file://runtime/triton_trtllm/scripts/convert_checkpoint.py)
- [fill_template.py](file://runtime/triton_trtllm/scripts/fill_template.py)
- [client_grpc.py](file://runtime/triton_trtllm/client_grpc.py)
- [client_http.py](file://runtime/triton_trtllm/client_http.py)
- [offline_inference.py](file://runtime/triton_trtllm/offline_inference.py)
- [streaming_inference.py](file://runtime/triton_trtllm/streaming_inference.py)
- [token2wav.py](file://runtime/triton_trtllm/token2wav.py)
</cite>

## 目录
1. [简介](#简介)
2. [Docker镜像定制化配置](#docker镜像定制化配置)
3. [Docker Compose服务编排](#docker-compose服务编排)
4. [run.sh脚本核心流程解析](#runsh脚本核心流程解析)
5. [DiT架构特殊部署流程](#dit架构特殊部署流程)
6. [资源需求与性能调优](#资源需求与性能调优)
7. [OpenAI兼容API桥接](#openai兼容api桥接)

## 简介
本文档详细阐述了基于NVIDIA Triton Inference Server和TensorRT-LLM的高性能推理部署方案，重点针对CosyVoice语音合成模型。文档深入解析了从Docker镜像构建、服务编排到模型部署的完整流程，涵盖了`Dockerfile.server`的定制化配置、`docker-compose.yml`的服务编排以及`run.sh`和`run_stepaudio2_dit_token2wav.sh`两个核心脚本的六个阶段执行流程。同时，文档提供了资源需求评估、性能调优建议以及OpenAI兼容API的桥接配置，为实现高效、稳定的语音合成服务提供全面指导。

## Docker镜像定制化配置

`Dockerfile.server`文件定义了用于部署CosyVoice模型的自定义Docker镜像。该镜像以NVIDIA官方的Triton Server镜像为基础，通过一系列定制化步骤确保环境满足特定的推理需求。

镜像构建过程首先从`nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3`基础镜像开始，该镜像已预装了Triton Inference Server和TensorRT-LLM运行环境。随后，通过`apt-get`安装了`cmake`工具，为后续的依赖编译提供支持。

核心的定制化步骤是安装特定版本的`torchaudio`依赖。脚本通过`git clone`命令从GitHub仓库克隆`pytorch/audio`项目，并使用`git checkout c670ad8`命令检出`c670ad8`这个特定的提交版本，确保依赖的稳定性和兼容性。接着，通过`python3 setup.py develop`命令在容器内进行开发模式安装。

最后，脚本将项目根目录下的`requirements.txt`文件复制到容器的`/workspace`目录，并使用`pip install -r`命令安装其中列出的所有Python依赖包，确保所有必要的库都已就位。

**Section sources**
- [Dockerfile.server](file://runtime/triton_trtllm/Dockerfile.server#L1-L8)
- [requirements.txt](file://runtime/triton_trtllm/requirements.txt#L1-L15)

## Docker Compose服务编排

`docker-compose.yml`文件定义了使用Docker Compose进行服务编排的配置，简化了容器的部署和管理。

该配置定义了一个名为`tts`的服务，其镜像为`soar97/triton-cosyvoice:25.06`。为了支持模型推理，配置了`shm_size: '1gb'`以增加共享内存大小，这对于处理大型模型和数据至关重要。

服务将容器的8000、8001和8002端口分别映射到宿主机的相同端口，用于Triton Server的HTTP、gRPC和指标服务。环境变量`PYTHONIOENCODING=utf-8`确保了Python进程的输入输出编码为UTF-8，避免了中文文本处理时可能出现的乱码问题。

在`deploy`部分，通过`resources.reservations.devices`配置，明确要求容器使用NVIDIA GPU设备，`device_ids: ['0']`指定了使用第一块GPU，`capabilities: [gpu]`声明了GPU能力。这确保了容器能够访问GPU进行加速计算。

容器启动后执行的命令是一个复杂的`bash`脚本，它首先安装`modelscope`库，然后克隆CosyVoice代码库并更新子模块，最后进入`runtime/triton_trtllm`目录并执行`run.sh`脚本，启动从模型下载到服务部署的完整流程。

**Section sources**
- [docker-compose.yml](file://runtime/triton_trtllm/docker-compose.yml#L1-L20)

## run.sh脚本核心流程解析

`run.sh`脚本是整个部署流程的核心，它通过一个六阶段的流水线，自动化地完成了从模型准备到性能测试的全过程。用户可以通过指定`stage`和`stop_stage`参数来控制执行的起止阶段。

### 阶段0：模型下载
此阶段负责下载所需的模型文件。脚本使用`huggingface-cli download`命令从Hugging Face下载`yuekai/cosyvoice2_llm`模型，并使用`modelscope download`命令从ModelScope下载`iic/CosyVoice2-0.5B`模型。此外，还会下载一个名为`spk2info.pt`的缓存文件，用于存储说话人信息，以加速后续推理。

### 阶段1：Checkpoint转换
此阶段将从Hugging Face下载的模型检查点（checkpoint）转换为TensorRT-LLM格式。首先，通过`scripts/convert_checkpoint.py`脚本将模型权重转换为TensorRT-LLM的权重格式。然后，使用`trtllm-build`命令，基于转换后的权重构建高性能的TensorRT推理引擎。构建时设置了`max_batch_size`和`max_num_tokens`等参数以优化性能。最后，通过`scripts/test_llm.py`脚本对生成的引擎进行功能测试，确保其能正确生成文本。

### 阶段2：模型仓库创建
此阶段创建Triton Server的模型仓库。脚本会复制预定义的模型配置文件夹（如`cosyvoice2`, `tensorrt_llm`, `token2wav`等）到`model_repo_cosyvoice2`目录。随后，使用`scripts/fill_template.py`脚本动态填充这些模型的`config.pbtxt`配置文件，将诸如模型路径、最大批处理大小、解耦模式等运行时参数注入到配置中，实现灵活的部署配置。

### 阶段3：Triton服务启动
此阶段启动Triton Inference Server。通过执行`tritonserver --model-repository $model_repo`命令，以之前创建的模型仓库作为服务根目录启动服务器。服务器启动后，即可通过HTTP或gRPC接口接收推理请求。

### 阶段4：HTTP客户端测试
此阶段执行一个单次请求的HTTP测试，主要用于验证离线TTS模式。脚本`client_http.py`会向服务器发送一个包含参考音频、参考文本和目标文本的请求，并将生成的音频保存到本地文件，以验证整个流程的正确性。

### 阶段5：gRPC性能基准测试
此阶段运行一个并发的gRPC基准测试，以评估服务器的性能。脚本`client_grpc.py`会从Hugging Face数据集中加载多个测试样本，并使用多任务并发的方式向服务器发送推理请求。它会记录每个请求的延迟、吞吐量等指标，并生成详细的性能报告，用于分析系统的并发处理能力。

**Section sources**
- [run.sh](file://runtime/triton_trtllm/run.sh#L1-L143)

## DiT架构特殊部署流程

`run_stepaudio2_dit_token2wav.sh`脚本专为DiT（Diffusion Transformer）架构的模型部署设计，其流程与标准的`run.sh`类似，但在模型和配置上存在显著差异。

该脚本不仅会克隆和下载标准的CosyVoice2模型，还会额外克隆`Step-Audio2`项目，并下载`Step-Audio-2-mini`模型。在阶段1中，`trtllm-build`命令的`--max_batch_size`被设置为64，以适应DiT模型的计算需求。

在阶段2中，脚本会创建一个名为`model_repo_cosyvoice2_dit`的模型仓库，并复制`cosyvoice2_dit`和`token2wav_dit`等特定于DiT架构的模型配置。配置参数也有所不同，例如`BLS_INSTANCE_NUM`被设置为10，`TRITON_MAX_BATCH_SIZE`被设置为1，以优化流式推理性能。

阶段3的启动命令更为复杂，它会并行启动两个服务：一个使用`trtllm-serve`命令启动的LLM服务，另一个使用`tritonserver`命令启动的Token2Wav服务。这种解耦的架构设计允许LLM和声码器在不同的端口上独立运行，提高了系统的灵活性和可扩展性。

后续的基准测试和离线推理阶段也针对这种解耦架构进行了适配，例如在阶段4中，客户端会连接到不同的端口进行测试。

**Section sources**
- [run_stepaudio2_dit_token2wav.sh](file://runtime/triton_trtllm/run_stepaudio2_dit_token2wav.sh#L1-L225)

## 资源需求与性能调优

### 资源需求评估
根据文档中的基准测试结果，部署该系统对硬件资源有较高要求。测试在单块L20 GPU上完成，表明L20 GPU是满足性能需求的推荐选择。L20 GPU拥有48GB的显存，足以容纳大型语言模型和声码器模型的推理引擎。对于显存较小的GPU，可能需要调整`kv_cache_free_gpu_mem_fraction`等参数来降低显存占用，或选择更小的模型。

### 性能调优建议
- **并发数设置**：`BLS_INSTANCE_NUM`（Backend Load Splitter实例数）是影响并发性能的关键参数。在DiT部署中，该值被设置为10，以充分利用GPU的并行计算能力。应根据实际的GPU型号和负载进行调整。
- **缓存策略**：通过设置`use_spk2info_cache=True`，可以缓存说话人的语音特征、声学特征和嵌入向量。这对于固定说话人的应用场景能显著降低首次推理延迟，提升用户体验。
- **批处理大小**：`max_batch_size`参数直接影响吞吐量和延迟。较大的批处理大小可以提高GPU利用率和吞吐量，但会增加延迟。需要在吞吐量和延迟之间找到平衡点。
- **流式与离线模式**：通过设置`DECOUPLED_MODE`为`True`或`False`，可以选择流式（streaming）或离线（offline）TTS模式。流式模式适合实时交互，而离线模式适合批量生成。

**Section sources**
- [run.sh](file://runtime/triton_trtllm/run.sh#L18-L19)
- [run_stepaudio2_dit_token2wav.sh](file://runtime/triton_trtllm/run_stepaudio2_dit_token2wav.sh#L23-L24)
- [README.md](file://runtime/triton_trtllm/README.md#L92-L126)

## OpenAI兼容API桥接

为了与现有生态系统集成，文档提供了将Triton服务桥接为OpenAI兼容API的方法。这通过一个名为`Triton-OpenAI-Speech`的独立项目实现。

首先，需要克隆该项目并安装其依赖。然后，在Triton服务正常运行后，启动`TTS_server.py`脚本。该脚本作为一个FastAPI应用，监听一个指定的端口（如10086），并作为代理将符合OpenAI API规范的请求转发给后端的Triton Server。

例如，一个发送到`http://localhost:10086/v1/audio/speech`的POST请求，会被转换为对Triton Server的gRPC或HTTP调用。这使得开发者可以使用标准的OpenAI SDK来调用CosyVoice模型，极大地简化了客户端的开发工作。需要注意的是，当前该桥接方案主要支持离线TTS模式。

**Section sources**
- [README.md](file://runtime/triton_trtllm/README.md#L126-L142)