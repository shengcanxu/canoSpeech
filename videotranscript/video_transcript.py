import os

from models.chatglm_translate import chatglm_translation
from models.config import VTConfig
from models.denoise_audio import separate_audio
from models.funasr_audio import funasr_audio
from models.opus_translate import opus_translate
from models.whisper_audio import whisper_audio
from models.utils.tools import get_video_info
import shutil
import json
from models.utils.logger import FileLogger
from models.utils.tools import conver_mp4, m4a2wav

def create_task(type:str, source_folder:str, target_folder:str, filename:str, task_folder:str):
    FileLogger.info(f"create {type} task: {filename}")
    task = {
        "type": type,
        "sourcefolder": source_folder,
        "targetfolder": target_folder,
        "filename": filename
    }
    json_str = json.dumps(task)
    task_file = os.path.join(task_folder, type+"_"+filename+".json")
    with open(task_file, "w") as fp:
        fp.write(json_str)

def transform(source_folder:str, target_folder:str, filename:str, task_folder:str):
    # change video to MP4 format
    from_video = os.path.join(source_folder, filename+"_v.m4s")
    to_video = os.path.join(target_folder, filename+".mp4")
    info = get_video_info(from_video)
    if info["video_codec_name"] != "mp4":
        print(f"change {from_video} to mp4")
        succ = conver_mp4(from_video, to_video)
        if not succ: return False
    else:
        shutil.copy(from_video, to_video)

    # change audio to wav format
    from_audio = os.path.join(source_folder, filename+"_a.m4s")
    to_audio = os.path.join(target_folder, filename+".wav")
    info = get_video_info(from_audio)
    if info["audio_codec_name"] != "wav": 
        print(f"change {from_audio} to wav")
        succ = m4a2wav(from_audio, to_audio)
        if not succ: return False
    else: 
        shutil.copy(from_audio, to_audio)

    # check if file exists
    if not os.path.exists(to_audio) or not os.path.exists(to_video):
        return False

    # create the denoise task
    create_task(
        type = "denoise",
        source_folder = target_folder,
        target_folder = target_folder.replace("/transformat/", "/denoise/"),
        filename = filename,
        task_folder = task_folder
    )
    return True

def denoise_audio(source_folder:str, target_folder:str, filename:str, task_folder:str):
    from_audio = os.path.join(source_folder, filename+".wav")
    to_vocal = os.path.join(target_folder, filename+"_vocal.wav")
    to_novocal = os.path.join(target_folder, filename+"_novocal.wav")
    succ = separate_audio(from_audio, to_vocal, to_novocal)
    if not succ: return False

    # check if file exists
    if not os.path.exists(to_vocal) or not os.path.exists(to_novocal):
        return False

    # create the SNR task
    create_task(
        type = "snr",
        source_folder = target_folder,
        target_folder = target_folder.replace("/denoise/", "/snr/"),
        filename = filename,
        task_folder = task_folder
    )
    return True

def snr_audio(source_folder:str, target_folder:str, filename:str, task_folder:str):
    from_audio = os.path.join(source_folder, filename+"_vocal.wav")
    snr_result = ""
    if VTConfig.snr_model == "whisper":
        snr_result = whisper_audio(from_audio)
    elif VTConfig.snr_model == "funasr":
        snr_result = funasr_audio(from_audio)
    else:
        FileLogger.error(f"wrong snr model type: {VTConfig.snr_model}")
        return False

    # save to disk
    to_text = os.path.join(target_folder, filename+".txt")
    with open(to_text, "w") as fp:
        snr_result = snr_result[0]
        for sentence in snr_result["sentence_info"]:
            sentence["spk"] = int(sentence["spk"])  # change the speaker variable type for json dump
        del snr_result["raw_text"]
        del snr_result["timestamp"]

        json_str = json.dumps(snr_result)
        fp.write(json_str)
    if not os.path.exists(to_text):
        return False

    # create the translation task
    create_task(
        type = "translation",
        source_folder = target_folder,
        target_folder = target_folder.replace("/snr/", "/translation/"),
        filename = filename,
        task_folder = task_folder
    )
    return True

def translation_text(source_folder:str, target_folder:str, filename:str, task_folder:str):
    from_text = os.path.join(source_folder, filename+".txt")
    json_str = open(from_text, "r").read()
    text_json = json.loads(json_str)

    if VTConfig.translation_model == "opus":
        translated = opus_translate(text_json["text"])
    elif VTConfig.translation_model == "chatglm":
        translated = chatglm_translation(text_json["text"])
    else:
        FileLogger.error(f"wrong translation model type: {VTConfig.translation_model}")
        return False


def translate(base_path):
    # 更改文件格式
    tasks_path = os.path.join(base_path, "translate/tasks")
    finished_path = os.path.join(base_path, "translate/finished")
    for task_file in os.listdir(tasks_path):
        succ = False
        task_path = os.path.join(tasks_path, task_file)
        with open(task_path, "r") as fp:
            json_str = fp.read()
            task = json.loads(json_str)
            FileLogger.info(f"get {task['type']} task named: {task['filename']}")

            if not os.path.exists(task["targetfolder"]):
                os.mkdir(task["targetfolder"])
            task_type = task['type']
            if task_type == 'transformat':  # 转换成为MP4和WAV
                succ = transform(task["sourcefolder"], task["targetfolder"], task["filename"], tasks_path)
            elif task_type == 'denoise':  # 将wav分离出人声和背景声
                succ = denoise_audio(task["sourcefolder"], task["targetfolder"], task["filename"], tasks_path)
            elif task_type == 'snr':  # 将人声wav识别出文字
                succ = snr_audio(task["sourcefolder"], task["targetfolder"], task["filename"], tasks_path)
            elif task_type == 'translation':  # 翻译成为目标语言
                succ = translation_text(task["sourcefolder"], task["targetfolder"], task["filename"], tasks_path)

        if succ:
            shutil.move(task_path, finished_path)
        else:
            FileLogger.error(f"error on {task['type']} file: {task['filename']}")


if __name__ == "__main__":
    path = "D:/dataset/bilibili"
    translate(path)