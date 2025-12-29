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
import random
import librosa
import base64
import io
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time
import wave
import tempfile
import os
from pydub import AudioSegment
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

# 导入CosyVoice相关模块
from cosyvoice.cli.cosyvoice import CosyVoice3
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

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

class SpeakerPromptSaveRequest(BaseModel):
    """说话人提示保存请求模型"""
    spk_id: str = Field(..., description="保存提示的说话人ID")
    prompt_text: str = Field(..., description="伴随提示音频的文本")
    prompt_audio_base64: str = Field(..., description="Base64编码的提示音频数据")

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

        audio_bytes = convert_to_mp3(audio_data, cosyvoice.sample_rate)
        yield audio_bytes

async def tts_instruct_stream_generator(request: Request, instructRequest: InstructRequest):
    """流式生成 TTS 音频的生成器函数"""

    for i in cosyvoice.inference_instruct2(instructRequest.tts_text, instructRequest.instruct_text, '', instructRequest.spk_id, stream=instructRequest.stream, speed=instructRequest.speed):
        audio_data = i['tts_speech'].numpy().flatten()
        # 检查客户端是否断开连接
        if await request.is_disconnected():
            logging.info("客户端已断开连接，终止 TTS 生成")
            break
        mp3_bytes = convert_to_mp3(audio_data, cosyvoice.sample_rate)
        yield mp3_bytes

def convert_to_mp3(audio_data, sample_rate):
    """
    将numpy数组转换为MP3格式的字节
    
    Args:
        audio_data: 音频数据numpy数组
        sample_rate: 采样率
        
    Returns:
        bytes: MP3格式的音频字节数据
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
    
    # 使用pydub将WAV转换为MP3
    wav_audio = AudioSegment(
        data=audio_int16.tobytes(),
        sample_width=2,  # 16位=2字节
        frame_rate=sample_rate,
        channels=1  # 单声道
    )
    mp3_buffer = io.BytesIO()
    # 导出为MP3格式
    wav_audio.export(mp3_buffer, format="mp3")
    
    # 返回MP3字节数据
    mp3_buffer.seek(0)
    mp3_bytes = mp3_buffer.read()
    
    return mp3_bytes

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
        return {"message": "欢迎使用CosyVoice API服务", "version": "3.0.0"}
    
    @app.get("/available_spks")
    def get_available_spks():
        """
        获取可用的预训练音色列表（包括自定义音色）
        
        Returns:
            AvailableSpksResponse: 可用说话人列表响应
        """
        spk_ids = cosyvoice.list_available_spks() # 这通常包括预设和已通过add_zero_shot_spk添加的  
        
        return {"speakers": spk_ids}

    @app.post("/tts")
    async def tts(request: Request, sft_request: SftRequest):
        """
        使用SFT模型合成语音
        
        Args:
            request: Request对象
            sft_request: SftRequest请求对象
            
        Returns:
            StreamingResponse: mp3格式的音频流响应
        """
        # 返回流式响应，媒体类型根据音频格式调整（MP3 用 audio/mpeg）
        return StreamingResponse(
            tts_stream_generator(request, sft_request),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=tts.mp3"}
        )

    @app.post("/tts/pcm")
    async def tts_pcm(request: Request, sft_request: SftRequest):
        """
        使用SFT模型合成语音，返回PCM格式
        
        Args:
            request: Request对象
            sft_request: SftRequest请求对象
            
        Returns:
            StreamingResponse: PCM格式的音频流响应
        """
        logging.info(f"接收到请求:{datetime.now()}")

        async def pcm_stream_generator(request: Request, sft_request: SftRequest):
            start_time = time.time()
            first_token_time = None
            
            for i in cosyvoice.inference_sft(
                sft_request.tts_text,
                sft_request.spk_id,
                stream=sft_request.stream,
                speed=sft_request.speed
            ):
                if first_token_time is None:
                    first_token_time = time.time()
                    latency = first_token_time - start_time
                    logging.info(f"---->First token latency: {latency} seconds")

                # 检查客户端是否断开连接
                if await request.is_disconnected():
                    logging.info("客户端已断开连接，终止 TTS 生成")
                    break
                
                audio_data = i['tts_speech'].numpy().flatten()
                # 直接返回PCM数据
                audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
                yield audio_int16.tobytes()
        
        return StreamingResponse(
            pcm_stream_generator(request, sft_request),
            media_type="audio/L16",
            headers={
                "Content-Disposition": "attachment; filename=tts.pcm",
                "X-Sample-Rate": str(cosyvoice.sample_rate),
                "X-Channels": "1",
                "X-Bit-Depth": "16"
            }
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
            prompt_wav = load_wav(prompt_wav_file, prompt_sr)
            
            if request.seed > 0:
                set_all_random_seed(request.seed)
            else:
                set_all_random_seed(random.randint(1, 100000000))
            
            def generate():
                for i in cosyvoice.inference_zero_shot(request.tts_text, request.prompt_text, prompt_wav, stream=request.stream, speed=request.speed):
                    audio_data = i['tts_speech'].numpy().flatten()
                    audio_bytes = convert_to_mp3(audio_data, cosyvoice.sample_rate)
                    yield audio_bytes
                
            return StreamingResponse(generate(), media_type="audio/mpeg")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    
    @app.post("/tts/cross_lingual")
    async def tts_cross_lingual(request: CrossLingualRequest):
        """
        使用跨语种复刻模式合成语音(没有调试)
        
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
                        audio_bytes = convert_to_mp3(audio_data, cosyvoice.sample_rate)
                        yield audio_bytes
                
                return StreamingResponse(generate(), media_type="audio/mpeg")
            else:
                # 非流式响应
                audio_segments = []
                for i in cosyvoice.inference_cross_lingual(request.tts_text, prompt_speech_16k, stream=request.stream, speed=request.speed):
                    audio_segments.append(i['tts_speech'].numpy().flatten())
                
                if audio_segments:
                    combined_audio = np.concatenate(audio_segments)
                    audio_bytes = convert_to_mp3(combined_audio, cosyvoice.sample_rate)
                    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")
                else:
                    raise HTTPException(status_code=500, detail="音频生成失败")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    
    @app.post("/tts/instruct")
    async def tts_instruct(request: Request, instructRequest: InstructRequest):
        """
        使用自然语言控制模式合成语音
        
        Args:
            request: InstructRequest请求对象
            
        Returns:
            StreamingResponse: WAV格式的音频流响应
        """
        if not instructRequest.instruct_text:
            sft_request = SftRequest(
                tts_text=instructRequest.tts_text,
                spk_id=instructRequest.spk_id,
                stream=instructRequest.stream,
                speed=instructRequest.speed
            )
            # 调用tts函数
            return await tts(request, sft_request)
            
        if instructRequest.seed > 0:
            set_all_random_seed(instructRequest.seed)
        else:
            set_all_random_seed(random.randint(1, 100000000))
        
        return StreamingResponse(tts_instruct_stream_generator(request, instructRequest), media_type="audio/mpeg", headers={"Content-Disposition": "attachment; filename=tts.mp3"})
    
    # 音色克隆
    @app.post("/clone")
    async def clone(prompt_text: str = Form(...), 
                              spk_id: str = Form(...), 
                              prompt_audio: UploadFile = File(...)):
        """
        保存用户上传的音色prompt
        
        Args:
            prompt_text: 提示文本
            spk_id: 说话人ID
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
        try:
            # 读取上传的音频文件内容
            contents = await prompt_audio.read()
            
            # 创建一个临时文件来存储音频内容
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(prompt_audio.filename)[1]) as temp_file:
                temp_file.write(contents)
                temp_file_path = temp_file.name
            
            # 将临时文件路径传递给add_zero_shot_spk，让前端处理函数内部完成加载和处理
            cosyvoice.add_zero_shot_spk(prompt_text, temp_file_path, spk_id)

            # Persist the updated spk2info
            cosyvoice.save_spkinfo()

            # 删除临时文件
            os.unlink(temp_file_path)

            return JSONResponse(content={"status": "success", "message": f"音色ID: {spk_id} 保存成功", "spk_id": spk_id}, status_code=200)
        except Exception as e:
            # print(e)
            logging.error(f"保存音色 {spk_id} 失败 (raw): {e}") # Log with repr for more detail
            # Sanitize the error message for the HTTP response
            detail_message = f"保存音色 {spk_id} 失败: {str(e)}"
            # 确保临时文件被清理
            if 'temp_file_path' in locals():
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
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
                        default='pretrained_models/Fun-CosyVoice3-0.5B',
                        help='local path or modelscope repo id')  # 模型路径或ModelScope仓库ID
    args = parser.parse_args()
    
    global cosyvoice, default_data
    try:
        cosyvoice = CosyVoice3(args.model_dir, load_trt=True, load_vllm=True, fp16=True, trt_concurrent=2)
    except Exception:
        raise TypeError('no valid model_type!')
    
    default_data = np.zeros(cosyvoice.sample_rate)
    
    app = create_app()
    logging.info("API服务已启动, 地址:{}, 端口:{}".format("0.0.0.0", args.port))
    uvicorn.run(app, host="0.0.0.0", port=args.port)  # 启动服务

if __name__ == '__main__':
    main()