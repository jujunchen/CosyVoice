# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入所需的标准库和第三方库
import os
import sys
import argparse
import logging
import io
import subprocess
import tempfile
# 设置matplotlib日志级别为WARNING，避免过多调试信息
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import torch
import torchaudio

# 获取当前脚本所在目录的绝对路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# 将项目根目录添加到Python路径中
sys.path.append('{}/../../..'.format(ROOT_DIR))
# 将第三方库Matcha-TTS目录添加到Python路径中
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

# 创建FastAPI应用实例
app = FastAPI()

# 配置CORS(跨源资源共享)中间件
# 允许所有来源、凭证、方法和请求头，使API可以被任何客户端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

# 定义全局变量，用于存储加载的语音合成模型实例
cosyvoice = None

def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    """
    SFT（Speaker Fine-Tuning）模式语音合成接口
    
    Args:
        tts_text (str): 待合成的文本
        spk_id (str): 说话人ID
        
    Returns:
        StreamingResponse: 流式音频响应
    """
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    """
    Zero-shot语音合成接口（无需训练即可克隆声音）
    
    Args:
        tts_text (str): 待合成的文本
        prompt_text (str): 提示音频对应的文本
        prompt_wav (UploadFile): 提示音频文件
        
    Returns:
        StreamingResponse: 流式音频响应
    """
    # 加载提示音频并确保采样率为16kHz
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    """
    跨语言语音合成接口
    
    Args:
        tts_text (str): 待合成的文本（目标语言）
        prompt_wav (UploadFile): 提示音频文件（源语言语音样本）
        
    Returns:
        StreamingResponse: 流式音频响应
    """
    # 加载提示音频并确保采样率为16kHz
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    """
    指令控制语音合成接口（通过指令文本控制语音情感、风格等）
    
    Args:
        tts_text (str): 待合成的文本
        spk_id (str): 说话人ID
        instruct_text (str): 指令文本，用于控制语音的情感、语调等
        
    Returns:
        StreamingResponse: 流式音频响应
    """
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    """
    基于音频样本和指令的语音合成接口
    
    Args:
        tts_text (str): 待合成的文本
        instruct_text (str): 指令文本，用于控制语音的情感、语调等
        prompt_wav (UploadFile): 提示音频文件
        
    Returns:
        StreamingResponse: 流式音频响应
    """
    # 加载提示音频并确保采样率为16kHz
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加端口参数，默认为50000
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    # 添加模型目录参数，默认为'iic/CosyVoice-300M'
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    # 解析命令行参数
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port)