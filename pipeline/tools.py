import copy
import json
import os
import subprocess
import sys
import torch
from logger import FileLogger

# run ffprobe 获取视频元信息
def runffprobe(cmd):
    try:
        cmd[-1]=os.path.normpath(rf'{cmd[-1]}')
        p = subprocess.Popen(['ffprobe']+cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,encoding="utf-8", text=True,
                             creationflags=0 if sys.platform != 'win32' else subprocess.CREATE_NO_WINDOW)
        out, errs = p.communicate()
        if p.returncode == 0:
            return out.strip()
        raise Exception(f'ffprobe error:{str(errs)}')
    except subprocess.CalledProcessError as e:
        raise Exception(f'ffprobe call error:{str(e)}')
    except Exception as e:
        raise Exception(f'ffprobe except error:{str(e)}')

# 执行 ffmpeg
def runffmpeg(
        arg, *, noextname=None,
        disable_gpu=True,  # True=禁止使用GPU解码
        no_decode=False, # False=禁止 h264_cuvid 解码，True=尽量使用硬件解码
        de_format="nv12"): # 硬件输出格式，模型cuda兼容性差，可选nv12
    
    arg_copy=copy.deepcopy(arg)
    cmd = ["ffmpeg", "-hide_banner", "-ignore_unknown"]
    
    # 启用了CUDA 并且没有禁用GPU
    if torch.cuda.is_available() and not disable_gpu:
        cmd.extend(["-hwaccel", 'cuvid', "-hwaccel_output_format", de_format, "-extra_hw_frames", "2"])
        # 如果没有禁止硬件解码，则添加
        if not no_decode:
            cmd.append("-c:v")
            cmd.append("h264_cuvid")
        for i, it in enumerate(arg):
            if i > 0 and arg[i - 1] == '-c:v':
                arg[i] = it.replace('libx264', "h264_nvenc").replace('copy', 'h264_nvenc')
   
    cmd = cmd + arg
    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         encoding="utf-8",
                         text=True, 
                         creationflags=0 if sys.platform != 'win32' else subprocess.CREATE_NO_WINDOW)
    FileLogger.info(f"runffmpeg: {' '.join(cmd)}")

    while True:
        try:
            # 等待0.1未结束则异常
            outs, errs = p.communicate()
            # 如果结束从此开始执行
            if p.returncode == 0:
                return True
            raise Exception(f'ffmpeg error:{str(errs)}')
        except Exception as e:
            #如果启用cuda时出错，则回退cpu
            if torch.cuda.is_available() and not disable_gpu:
                # 切换为cpu
                # disable_gpt=True禁用GPU，no_decode=True禁止h264_cuvid解码，
                return runffmpeg(arg_copy, disable_gpu=True)
            raise Exception(str(e))

    
# 获取视频信息
def get_video_info(mp4_file, *, video_fps=False, video_scale=False, video_time=False):
    out = runffprobe(['-v','quiet','-print_format','json','-show_format','-show_streams',mp4_file])
    if out is False:
        raise Exception(f'ffprobe error:dont get video information')
    out = json.loads(out)
    result = {
        "video_fps": 0,
        "video_codec_name": "h264",
        "audio_codec_name": "aac",
        "width": 0,
        "height": 0,
        "time": 0,
        "streams_len": 0,
        "streams_audio": 0
    }
    if "streams" not in out or len(out["streams"]) < 1:
        raise Exception(f'ffprobe error:streams is 0')

    if "format" in out and out['format']['duration']:
        result['time'] = int(float(out['format']['duration']) * 1000)
    for it in out['streams']:
        result['streams_len'] += 1
        if it['codec_type'] == 'video':
            result['video_codec_name'] = it['codec_name']
            result['width'] = int(it['width'])
            result['height'] = int(it['height'])
            fps, c = it['r_frame_rate'].split('/')
            if not c or c == '0':
                c = 1
                fps = int(fps)
            else:
                fps = round(int(fps) / int(c))
            result['video_fps'] = fps
        elif it['codec_type'] == 'audio':
            result['streams_audio'] += 1
            result['audio_codec_name'] = it['codec_name']

    if video_time:
        return result['time']
    if video_fps:
        return ['video_fps']
    if video_scale:
        return result['width'], result['height']
    return result

# 视频转为 mp4格式 nv12 + not h264_cuvid
def conver_mp4(source_file, out_mp4):
    return runffmpeg([
        '-y',
        '-i',
        os.path.normpath(source_file),
        '-c:v',
        'libx264',
        "-c:a",
        "aac",
        out_mp4
    ], no_decode=True, de_format="nv12")

# m4a 转为 wav cuda + h264_cuvid
def m4a2wav(m4afile, wavfile):
    cmd = [
        "-y",
        "-i",
        m4afile,
        "-ac",
        "1",
        wavfile
    ]
    return runffmpeg(cmd)