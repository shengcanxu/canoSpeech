# canoSpeech
text to speech, a repro for producing human natural speech

NaturalTTS is revised from NaturalSpeech2 and change the Diffusion to Flow

use wavenet and attention. As descripted in Figure 4 in NaturalSpeech2 paper, but change a little bit. 
    the changes is: 
        1. keep the flow architecture from VITS
        2. wrape the wavenet, add attention and FiLM to it, as description in Figure 4

# get dataset and process
1. VCTK: run download/VCTK.py

# usefull script
1. re-sample audio file: dataset/resample.py
2. tokenize text and change to phonone: preprocess.py
3. generate pitch from audio using preprocess.gen_pitch.py

# run scripts
all run scripts are in recipes folder. each model one folder

# reference:
#- pitch(f0)
can be generated using three methods, we use 4:
1. using pysptk.sptk.rapt from pysptk project
2. using librosa.pyin from librosa project. util.audio_process.py has the function of compute_f0() for it. but it's very slow.
3. as used in NaturalSpeech2, use pyWord in https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder 
4. FCNF0++, code in https://github.com/interactiveaudiolab/penn, the latest pitch deteck model. 
   1. Model checkpoint: https://huggingface.co/maxrmorrison/fcnf0-plus-plus/tree/main 

#- duration:
1. duration is generated by VITS model, by running the reference of VITS to get the duration. see code in preprocess.gen_audio_stat.py
2. fails to use VITS, switch to Montreal-Forced-Aligner tool, see more detail: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
    2.1. here are some useful introduction of Montreal-Forced-Aligner:
        1) https://techfirst.medium.com/forced-alignment-how-to-match-audio-with-a-transcript-via-machine-learning-dd19da8c0f04

# audio-text align:
1. there are many tools to align audio and text, start from here: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner 

#- speaker:
speaker embedding is generated from  H/ASP model
paper:  Clova baseline system for the voxceleb speaker recognition challenge
speaker enbedding encoder configs:
    model checkpoint: https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar
    configs: https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json 

# phonemizer & symbols
1. about symbols of Chinese and Japanese, see detail in https://github.com/yl4579/StyleTTS/issues/10 

# emotion extractor
1. extract emotion from wav file using wav2vec2 model:  https://github.com/audeering/w2v2-how-to/blob/main/notebook.ipynb
2. 


#- projects
1. https://github.com/yangdongchao/AcademiCodec 
2. https://github.com/CODEJIN/NaturalSpeech2 
3. https://github.com/heatz123/naturalspeech 
4. https://github.com/coqui-ai/TTS  

# resources
1. some usefull course about audio AI: https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0 
2. how to augement audio for AI research: https://www.youtube.com/watch?v=bm1cQfb_pLA&list=PL-wATfeyAMNoR4aqS-Fv0GRmS6bx5RtTW&index=2 
3. audio augementation tools:
   1. librosa
   2. audiomentations
   3. torch-audiomentations
   4. torchaudio.transforms
4. useful checkpoints
   1. H/ASP speaker embedding model checkpoint: https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar
   2. RQV quantizer checkpoint: https://huggingface.co/Dongchao/AcademiCodec/tree/main

# models
1. many models includes in TTS:  https://github.com/mozilla/TTS/wiki/Released-Models 

# findings
1. learning rate非常重要. vits用大learning rate失败, 但是用 2e-4 成功. 可以用fastai里面的find_lr方法获得最佳lr
2. 在刚开始学习的时候，很容易产生loss=NaN，甚至简单的weight_norm(Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)) 都可能让结果变成NaN
3. 尝试多次训练forward KL and backword KL，都失败了，naturalspeech论文上的内容应该是有问题的。从原理上说应该也不需要两个KL


