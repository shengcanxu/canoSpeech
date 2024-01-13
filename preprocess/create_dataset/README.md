# 总体框架  
采用的是libri-light这个facebook创建的库类似的做法来生成的。 一共分为如下几步：
1. 下载音频， 下载的代码保存在crawler项目中的librivox文件夹中。并不是所有的book的音频都下载，只是下载总共大于10分钟的音频
2. 使用whisper来识别出音频对应的文字
3. 根据文字将音频分割成为10S左右的音频

## 下载语音
使用libri-light这个github改过来的代码，代码存在自己的scrawler的项目中

## 识别语音
使用whisper来识别， 具体是使用huggingface的whisper-large-v3 ，详细看 https://huggingface.co/openai/whisper-large-v3
小tips： huggingface里面的模型和配置是可以下载到本地然后从本地加载的


