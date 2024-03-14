import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

opus_model = None
opus_tokenizer = None
def init_model():
    print("initialize opus translation model...")
    if sys.platform == "win32":
        model_dir = "D:/models/opus-mt/opus-mt-zh-en"
    else:
        model_dir = "/home/cano/models/opus-mt/opus-mt-zh-en"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return model, tokenizer

def opus_translate(text:str):
    global opus_model, opus_tokenizer
    if opus_model is None or opus_tokenizer is None:
        opus_model, opus_tokenizer = init_model()

    # 对中文句子进行分词
    input_ids = opus_tokenizer.encode(text, return_tensors="pt")
    # 进行翻译
    output_ids = opus_model.generate(input_ids)
    # 将翻译结果转换为字符串格式
    english_str = opus_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #如果最后有一个.，则去掉
    if english_str[-1] == '.':
        english_str = english_str[:-1]
    return english_str

if __name__ == "__main__":
    text = "目的：通过几个关键设计来增强从文本到后文本的能力，降低从语音到后文本的复杂性，包括音素预训练、可微时长建模、双向前/后建模以及VAE中的记忆机制。"
    translated = opus_translate(text)
    print(translated)