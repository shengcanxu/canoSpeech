# canoSpeech
text to speech, a repro for producing human natural speech

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
can be generated using three methods, we use 2:
1. using pysptk.sptk.rapt from pysptk project
2. using librosa.pyin from librosa project. util.audio_process.py has the function of compute_f0() for it.
3. as used in NaturalSpeech2, use pyWord in https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder 

#- projects
1. https://github.com/yangdongchao/AcademiCodec 
2. https://github.com/CODEJIN/NaturalSpeech2 
3. https://github.com/heatz123/naturalspeech 
4. https://github.com/coqui-ai/TTS  
5. 
