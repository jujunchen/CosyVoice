# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
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
import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
import random
import librosa
import base64
import io
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
import time
import asyncio
import wave

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

# 导入CosyVoice相关模块
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

# 配置文件日志记录
import logging as std_logging # 为了访问FileHandler, Formatter等，因为'logging'已被占用
log_file_path = os.path.join(ROOT_DIR, "logs/api_service.log") # ROOT_DIR已在上面定义
file_handler = std_logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(std_logging.INFO) # 为此特定处理程序设置级别
formatter = std_logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
file_handler.setFormatter(formatter)

# 将处理程序添加到从cosyvoice.utils.file_utils导入的logger实例
# 这假设'logging'是一个logger对象(例如，来自logging.getLogger())
if hasattr(logging, 'addHandler'):
    logging.addHandler(file_handler)
    # 确保logger本身至少为INFO级别，以便将消息传递给处理程序。
    # 如果当前级别是NOTSET(0)或更高(不太详细)而不是INFO，则将其设置为INFO。
    if hasattr(logging, 'setLevel') and hasattr(logging, 'level'):
        if logging.level == 0 or logging.level > std_logging.INFO: # logging.NOTSET为0
            logging.setLevel(std_logging.INFO)
    elif hasattr(logging, 'setLevel'): # 如果它有setLevel但没有level(对于标准logger不太可能)
        logging.setLevel(std_logging.INFO) # 以防万一进行设置
else:
    # 回退：如果'logging'不是具有addHandler的标准logger对象，
    # 尝试配置根logger。如果'logging'只是模块，可能会发生这种情况。
    std_logger = std_logging.getLogger()
    std_logger.addHandler(file_handler)
    if std_logger.level == 0 or std_logger.level > std_logging.INFO:
        std_logger.setLevel(std_logging.INFO)
    # 如果可能，使用原始logging对象记录此消息，否则使用std_logging
    # if hasattr(logging, 'info'):
    #     logging.info("回退：已为文件输出配置根logger，因为'cosyvoice.utils.file_utils.logging'没有addHandler。")
    # else:
    #     std_logging.info("回退：已为文件输出配置根logger，因为'cosyvoice.utils.file_utils.logging'没有addHandler。")


# 定义请求模型
class SftRequest(BaseModel):
    """预训练音色合成请求模型"""
    tts_text: str              # 待合成文本
    spk_id: str                # 预训练说话人ID
    stream: bool = False       # 是否启用流式合成
    speed: float = 1.0         # 语音速度调节因子
    seed: int = 0              # 随机种子

class ZeroShotRequest(BaseModel):
    """零样本语音合成请求模型"""
    tts_text: str                         # 待合成文本
    prompt_text: str                      # 提示文本
    prompt_audio_base64: Optional[str] = None  # Base64编码的提示音频数据
    stream: bool = False                  # 是否启用流式合成
    speed: float = 1.0                    # 语音速度调节因子
    seed: int = 0                         # 随机种子

class CrossLingualRequest(BaseModel):
    """跨语言语音合成请求模型"""
    tts_text: str                         # 待合成文本
    prompt_audio_base64: Optional[str] = None  # Base64编码的提示音频数据
    stream: bool = False                  # 是否启用流式合成
    speed: float = 1.0                    # 语音速度调节因子
    seed: int = 0                         # 随机种子

class InstructRequest(BaseModel):
    """指令引导语音合成请求模型"""
    tts_text: str             # 待合成文本
    spk_id: str               # 预训练说话人ID
    instruct_text: str        # 指令文本
    stream: bool = False      # 是否启用流式合成
    speed: float = 1.0        # 语音速度调节因子
    seed: int = 0             # 随机种子

class SpeakerDetail(BaseModel):
    """说话人详情模型"""
    spk_id: str     # 说话人ID
    spk_name: str   # 说话人名称

class AvailableSpksResponse(BaseModel):
    """可用说话人列表响应模型"""
    speakers: List[SpeakerDetail]  # 说话人详情列表

class SpeakerPromptSaveRequest(BaseModel):
    """说话人提示保存请求模型"""
    spk_id: str = Field(..., description="保存提示的说话人ID")
    prompt_text: str = Field(..., description="伴随提示音频的文本")
    prompt_audio_base64: str = Field(..., description="Base64编码的提示音频数据")

# 用于存储自定义音色名称
SPEAKER_NAMES_FILE = "speaker_names.json"  # 自定义音色名称存储文件
SPEAKER_NAMES: Dict[str, str] = {}         # 音色名称映射字典

def load_speaker_names():
    """
    从JSON文件加载自定义音色名称
    """
    global SPEAKER_NAMES
    if os.path.exists(SPEAKER_NAMES_FILE):
        try:
            with open(SPEAKER_NAMES_FILE, 'r', encoding='utf-8') as f:
                SPEAKER_NAMES = json.load(f)
            logging.info(f"自定义音色名称已从 {SPEAKER_NAMES_FILE} 加载。")
        except Exception as e:
            logging.error(f"加载自定义音色名称失败: {e}")
    else:
        logging.info(f"{SPEAKER_NAMES_FILE} 未找到，将使用空的音色名称列表。")

def save_speaker_names():
    """
    将自定义音色名称保存到JSON文件
    """
    global SPEAKER_NAMES
    try:
        with open(SPEAKER_NAMES_FILE, 'w', encoding='utf-8') as f:
            json.dump(SPEAKER_NAMES, f, ensure_ascii=False, indent=4)
        logging.info(f"自定义音色名称已保存到 {SPEAKER_NAMES_FILE}。")
    except Exception as e:
        logging.error(f"保存自定义音色名称失败: {e}")

# 音频处理参数
max_val = 0.8       # 音频最大幅度值
prompt_sr = 16000   # 提示音频采样率

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """
    音频后处理函数
    
    Args:
        speech: 输入音频数据
        top_db: 音频修剪的阈值(dB)
        hop_length: 音频帧跳跃长度
        win_length: 音频窗口长度
        
    Returns:
        处理后的音频数据
    """
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech

def decode_audio(base64_audio):
    """
    从Base64字符串解码音频数据
    
    Args:
        base64_audio: Base64编码的音频数据字符串
        
    Returns:
        BytesIO对象，包含解码后的音频数据
    """
    if not base64_audio:
        return None
    
    try:
        audio_bytes = base64.b64decode(base64_audio)
        # 创建临时文件以便torchaudio可以读取
        temp_file = io.BytesIO(audio_bytes)
        temp_file.name = "temp.wav"  # 为BytesIO对象添加名称属性
        return temp_file
    except Exception as e:
        logging.error(f"解码音频失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"音频解码失败: {str(e)}")

async def tts_stream_generator(request: Request, sft_request: SftRequest):
    """流式生成 TTS 音频的生成器函数"""
    start_time = time.time()
    first_token_time = None

    # 关键：遍历生成器时，定期检查连接状态
    for i in cosyvoice.inference_sft(
        sft_request.tts_text,
        sft_request.spk_id,
        stream=sft_request.stream,
        speed=sft_request.speed
    ):
        # 检查客户端是否断开连接
        if await request.is_disconnected():
            logging.info("客户端已断开连接，终止 TTS 生成")
            break

        
        audio_data = i['tts_speech'].numpy().flatten()
        if first_token_time is None:
            first_token_time = time.time()
            latency = first_token_time - start_time
            logging.info(f"---->获取到第一个token: {latency} seconds")

        audio_bytes = convert_to_aac(audio_data, cosyvoice.sample_rate)
        yield audio_bytes

def convert_to_aac(audio_data, sample_rate):
    """
    将numpy数组转换为AAC格式的字节
    
    Args:
        audio_data: 音频数据numpy数组
        sample_rate: 采样率
        
    Returns:
        bytes: AAC格式的音频字节数据
    """
    #限制幅值防止削波
    audio_data = np.clip(audio_data, -1.0, 1.0)

    # 添加淡入淡出效果（可选）
    fade_samples = int(0.01 * sample_rate)  # 10ms的淡入淡出
    if len(audio_data) > 2 * fade_samples:
        # 淡入
        audio_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
        # 淡出
        audio_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    # 将float32转换为int16
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # 创建临时WAV文件
    # wav_buffer = io.BytesIO()
    # with wave.open(wav_buffer, 'wb') as wav_file:
    #     wav_file.setnchannels(1)  # 单声道
    #     wav_file.setsampwidth(2)  # 16位
    #     wav_file.setframerate(sample_rate)
    #     wav_file.writeframes(audio_int16.tobytes())
    
    # # 重置缓冲区指针
    # wav_buffer.seek(0)
    
    # 使用pydub将WAV转换为AAC
    from pydub import AudioSegment
    wav_audio = AudioSegment(
        data=audio_int16.tobytes(),
        sample_width=2,  # 16位=2字节
        frame_rate=sample_rate,
        channels=1  # 单声道
    )
    aac_buffer = io.BytesIO()
    # 使用mp4容器保存AAC音频
    wav_audio.export(aac_buffer, format="adts", codec="aac")
    # wav_audio.export(aac_buffer, format="mp3")
    
    # 返回AAC字节数据
    aac_buffer.seek(0)
    aac_bytes = aac_buffer.read()
    
    return aac_bytes

def convert_to_wav(audio_data, sample_rate):
    """
    将numpy数组转换为WAV格式的字节
    
    Args:
        audio_data: 音频数据numpy数组
        sample_rate: 采样率
        
    Returns:
        bytes: WAV格式的音频字节数据
    """
    # 将float32转换为int16
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # 创建WAV文件头
    buffer = io.BytesIO()
    with io.BytesIO() as wav_file:
        # 创建WAV文件
        wav_writer = wave.open(wav_file, 'wb')
        wav_writer.setnchannels(1)  # 单声道
        wav_writer.setsampwidth(2)  # 16位
        wav_writer.setframerate(sample_rate)
        wav_writer.writeframes(audio_int16.tobytes())
        wav_writer.close()
        
        # 获取WAV文件的字节
        wav_file.seek(0)
        wav_bytes = wav_file.read()
    
    return wav_bytes


def create_app():
    """
    创建FastAPI应用实例
    
    Returns:
        FastAPI应用实例
    """
    app = FastAPI(title="CosyVoice API", description="语音合成API服务")
    
    # 添加CORS中间件，允许跨域请求
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # 允许所有源
        allow_credentials=True,        # 允许凭证
        allow_methods=["*"],          # 允许所有方法
        allow_headers=["*"],          # 允许所有头部
    )
    
    @app.get("/")
    def read_root():
        """
        根路径，返回API服务基本信息
        """
        return {"message": "欢迎使用CosyVoice API服务", "version": "1.0.0"}
    
    @app.get("/available_spks", response_model=AvailableSpksResponse)
    def get_available_spks():
        """
        获取可用的预训练音色列表（包括自定义音色）
        
        Returns:
            AvailableSpksResponse: 可用说话人列表响应
        """
        spk_ids = cosyvoice.list_available_spks() # 这通常包括预设和已通过add_zero_shot_spk添加的
        speakers_details = []
        for spk_id in spk_ids:
            # 优先从SPEAKER_NAMES获取自定义名称，否则默认为spk_id
            spk_name = SPEAKER_NAMES.get(spk_id, spk_id)
            speakers_details.append(SpeakerDetail(spk_id=spk_id, spk_name=spk_name))
        
        # 对于仅存在于SPEAKER_NAMES中但可能未在cosyvoice内部列表的（理论上不应发生如果保存逻辑正确）
        # 但为了完整性，可以考虑合并，不过当前cosyvoice.list_available_spks()应为权威来源
        return {"speakers": speakers_details}

    @app.post("/tts")
    async def tts(request: Request, sft_request: SftRequest):
        """
        使用SFT模型合成语音
        
        Args:
            request: Request对象
            sft_request: SftRequest请求对象
            
        Returns:
            StreamingResponse: aac格式的音频流响应
        """
        # 返回流式响应，媒体类型根据音频格式调整（AAC 用 audio/aac）
        return StreamingResponse(
            tts_stream_generator(request, sft_request),
            media_type="audio/aac",
            headers={"Content-Disposition": "attachment; filename=tts.aac"}
        )
    
    @app.post("/tts/zero_shot")
    async def tts_zero_shot(request: ZeroShotRequest):
        """
        使用3s极速复刻模式合成语音
        
        Args:
            request: ZeroShotRequest请求对象
            
        Returns:
            StreamingResponse: WAV格式的音频流响应
        """
        if not request.prompt_audio_base64:
            raise HTTPException(status_code=400, detail="prompt音频不能为空")
        
        if not request.prompt_text:
            raise HTTPException(status_code=400, detail="prompt文本不能为空")
        
        # 解码Base64音频
        prompt_wav_file = decode_audio(request.prompt_audio_base64)
        if not prompt_wav_file:
            raise HTTPException(status_code=400, detail="prompt音频解码失败")
        
        try:
            # 加载并处理音频
            prompt_speech_16k = postprocess(load_wav(prompt_wav_file, prompt_sr))
            
            if request.seed > 0:
                set_all_random_seed(request.seed)
            else:
                set_all_random_seed(random.randint(1, 100000000))
            
            def generate():
                for i in cosyvoice.inference_zero_shot(request.tts_text, request.prompt_text, prompt_speech_16k, stream=request.stream, speed=request.speed):
                    audio_data = i['tts_speech'].numpy().flatten()
                    audio_bytes = convert_to_aac(audio_data, cosyvoice.sample_rate)
                    yield audio_bytes
                
            return StreamingResponse(generate(), media_type="audio/aac")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    
    @app.post("/tts/cross_lingual")
    async def tts_cross_lingual(request: CrossLingualRequest):
        """
        使用跨语种复刻模式合成语音
        
        Args:
            request: CrossLingualRequest请求对象
            
        Returns:
            StreamingResponse: WAV格式的音频流响应
        """
        if not request.prompt_audio_base64:
            raise HTTPException(status_code=400, detail="prompt音频不能为空")
        
        # 解码Base64音频
        prompt_wav_file = decode_audio(request.prompt_audio_base64)
        if not prompt_wav_file:
            raise HTTPException(status_code=400, detail="prompt音频解码失败")
        
        try:
            # 检查模型是否支持跨语种复刻
            if cosyvoice.instruct is True:
                raise HTTPException(status_code=400, detail=f"当前模型不支持跨语种复刻模式，请使用CosyVoice-300M模型")
            
            # 加载并处理音频
            prompt_speech_16k = postprocess(load_wav(prompt_wav_file, prompt_sr))
            
            if request.seed > 0:
                set_all_random_seed(request.seed)
            else:
                set_all_random_seed(random.randint(1, 100000000))
            
            if request.stream:
                # 流式响应
                def generate():
                    for i in cosyvoice.inference_cross_lingual(request.tts_text, prompt_speech_16k, stream=request.stream, speed=request.speed):
                        audio_data = i['tts_speech'].numpy().flatten()
                        audio_bytes = convert_to_aac(audio_data, cosyvoice.sample_rate)
                        yield audio_bytes
                
                return StreamingResponse(generate(), media_type="audio/aac")
            else:
                # 非流式响应
                audio_segments = []
                for i in cosyvoice.inference_cross_lingual(request.tts_text, prompt_speech_16k, stream=request.stream, speed=request.speed):
                    audio_segments.append(i['tts_speech'].numpy().flatten())
                
                if audio_segments:
                    combined_audio = np.concatenate(audio_segments)
                    audio_bytes = convert_to_aac(combined_audio, cosyvoice.sample_rate)
                    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/aac")
                else:
                    raise HTTPException(status_code=500, detail="音频生成失败")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    
    @app.post("/tts/instruct")
    async def tts_instruct(request: InstructRequest):
        """
        使用自然语言控制模式合成语音
        
        Args:
            request: InstructRequest请求对象
            
        Returns:
            StreamingResponse: WAV格式的音频流响应
        """
        try:
            # 检查模型是否支持自然语言控制
            if cosyvoice.instruct is False:
                raise HTTPException(status_code=400, detail=f"当前模型不支持自然语言控制模式，请使用CosyVoice-300M-Instruct模型")
            
            if not request.instruct_text:
                raise HTTPException(status_code=400, detail="instruct文本不能为空")
            
            if request.seed > 0:
                set_all_random_seed(request.seed)
            else:
                set_all_random_seed(random.randint(1, 100000000))
            
            if request.stream:
                # 流式响应
                def generate():
                    for i in cosyvoice.inference_instruct(request.tts_text, request.spk_id, request.instruct_text, stream=request.stream, speed=request.speed):
                        audio_data = i['tts_speech'].numpy().flatten()
                        aac_bytes = convert_to_aac(audio_data, cosyvoice.sample_rate)
                        yield aac_bytes
                
                return StreamingResponse(generate(), media_type="audio/aac")
            else:
                # 非流式响应
                audio_segments = []
                for i in cosyvoice.inference_instruct(request.tts_text, request.spk_id, request.instruct_text, stream=request.stream, speed=request.speed):
                    audio_segments.append(i['tts_speech'].numpy().flatten())
                
                if audio_segments:
                    combined_audio = np.concatenate(audio_segments)
                    audio_bytes = convert_to_aac(combined_audio, cosyvoice.sample_rate)
                    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/aac")
                else:
                    raise HTTPException(status_code=500, detail="音频生成失败")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    
    # 音色克隆
    @app.post("/clone")
    async def clone(prompt_text: str = Form(...), 
                              spk_id: str = Form(...), 
                              spk_name: Optional[str] = Form(None), # 新增音色名称参数
                              prompt_audio: UploadFile = File(...)):
        """
        保存用户上传的音色prompt
        
        Args:
            prompt_text: 提示文本
            spk_id: 说话人ID
            spk_name: 说话人名称（可选）
            prompt_audio: 提示音频文件
            
        Returns:
            JSONResponse: 保存结果响应
        """
        if not prompt_audio:
            raise HTTPException(status_code=400, detail="prompt_audio 音频文件不能为空")
    
        if not prompt_text:
            raise HTTPException(status_code=400, detail="prompt_text 不能为空")

        if not spk_id:
            raise HTTPException(status_code=400, detail="spk_id 不能为空")

        # 从 UploadFile 对象获取音频
        # 注意：load_wav 需要能够处理 UploadFile.file，它是一个 SpooledTemporaryFile
        # 或者先将 UploadFile 保存到临时文件再读取
        # 为简单起见，这里假设 load_wav 可以直接处理 file-like object
        # 如果不行，需要先 await prompt_audio.read() 然后用 io.BytesIO 包装
        prompt_wav_file = prompt_audio.file

        try:
            # 加载并处理音频
            prompt_speech_16k = postprocess(load_wav(prompt_wav_file, prompt_sr))

            cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, spk_id)

            # Use frontend_zero_shot to get all necessary features including embeddings
            # Pass tts_text='' and zero_shot_spk_id='' to ensure fresh extraction
            # model_input_data = cosyvoice.frontend.frontend_zero_shot(
            #     tts_text='',
            #     prompt_text=prompt_text,
            #     prompt_speech_16k=prompt_speech_16k,
            #     resample_rate=cosyvoice.sample_rate,
            #     zero_shot_spk_id=''
            # )

            # # Remove fields related to tts_text as they are not part of a speaker prompt
            # if 'text' in model_input_data:
            #     del model_input_data['text']
            # if 'text_len' in model_input_data:
            #     del model_input_data['text_len']

            # # Add the 'embedding' key directly for SFT compatibility,
            # # using the llm_embedding (which is the speaker embedding).
            # if 'llm_embedding' in model_input_data:
            #     model_input_data['embedding'] = model_input_data['llm_embedding']
            # else:
            #     # Fallback or error if llm_embedding is somehow missing
            #     logging.error("llm_embedding not found in frontend_zero_shot output. Cannot save speaker for SFT.")
            #     raise HTTPException(status_code=500, detail="音色特征提取失败，缺少llm_embedding")

            # Store the processed information in spk2info
            # cosyvoice.frontend.spk2info[spk_id] = model_input_data

            # Persist the updated spk2info
            cosyvoice.save_spkinfo()

            # 保存或更新自定义音色名称
            actual_spk_name = spk_name if spk_name and spk_name.strip() else spk_id
            SPEAKER_NAMES[spk_id] = actual_spk_name
            save_speaker_names() # 保存自定义音色名称到JSON文件

            return JSONResponse(content={"status": "success", "message": f"音色 '{actual_spk_name}' (ID: {spk_id}) 保存成功", "spk_id": spk_id, "spk_name": actual_spk_name}, status_code=200)
        except Exception as e:
            # print(e)
            logging.error(f"保存音色 {spk_id} 失败 (raw): {e}") # Log with repr for more detail
            # Sanitize the error message for the HTTP response
            detail_message = f"保存音色 {spk_id} 失败:"
            raise HTTPException(status_code=500, detail=detail_message)

    # 文件上传接口，用于上传prompt音频
    @app.post("/upload_audio")
    async def upload_audio(file: UploadFile = File(...)):
        """
        上传音频文件并返回Base64编码
        
        Args:
            file: 上传的音频文件
            
        Returns:
            dict: 包含文件名和Base64编码的字典
        """
        try:
            contents = await file.read()
            # 将文件内容编码为base64
            base64_audio = base64.b64encode(contents).decode('utf-8')
            return {"filename": file.filename, "audio_base64": base64_audio}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")
    
    return app


def main():
    """
    主函数，启动API服务
    """
    import uvicorn
    # import wave # 已移到顶部
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=5000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')  # 模型路径或ModelScope仓库ID
    args = parser.parse_args()
    
    global cosyvoice, default_data
    try:
        cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=True, load_vllm=True, fp16=True, trt_concurrent=4)
    except Exception:
        raise TypeError('no valid model_type!')
    
    default_data = np.zeros(cosyvoice.sample_rate)

    load_speaker_names() # 应用启动时加载自定义音色名称
    
    app = create_app()
    logging.info("API服务已启动, 地址:{}, 端口:{}".format("0.0.0.0", args.port))
    uvicorn.run(app, host="0.0.0.0", port=args.port)  # 启动服务

if __name__ == '__main__':
    main()