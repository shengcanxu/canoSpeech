import sys
from modelscope import AutoTokenizer, AutoModel, snapshot_download

chatglm_model = None
chatglm_tokenizer = None
def init_model():
    print("initialize chatglm model...")
    # model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision = "v1.0.2", cache_dir="D:/models")
    if sys.platform == "win32":
        model_dir = "D:/models/ZhipuAI/chatglm3-6b"
    else:
        model_dir = "/home/cano/models/ZhipuAI/chatglm3-6b"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
    model = model.eval()
    return model, tokenizer

# do translation using chatglm model
def chatglm_translation(text_path:str):
    global chatglm_model, chatglm_tokenizer
    if chatglm_model is None or chatglm_tokenizer is None:
        chatglm_model, chatglm_tokenizer = init_model()

    with open(text_path, "r") as fp:
        text = fp.read()
        response, history = chatglm_model.chat(chatglm_tokenizer, text, history=[])
        return response

if __name__ == "__main__":
    pass
    # response, history = model.chat(tokenizer, "你好", history=[])
    # print(response)
    # response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    # print(response)

    # model_dir = snapshot_download("iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn", revision = "v2.0.4", cache_dir="D:/models")
    # print(model_dir)
    #
    # model_dir = snapshot_download("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", revision="v2.0.4", cache_dir="D:/models")
    # print(model_dir)
    #
    # model_dir = snapshot_download("iic/punc_ct-transformer_cn-en-common-vocab471067-large", revision="v2.0.4", cache_dir="D:/models")
    # print(model_dir)
