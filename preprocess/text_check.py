

if __name__ == "__main__":
    lines = []
    with open("D:\\project\\canoSpeech\\filelists\\linux\\all\\vctk_train_filelist.txt.cleaned", encoding="utf-8") as fp:
        pos = 0
        while True:
            line = fp.readline()
            pos += 1
            if line is None or len(line) == 0: break

            texts = line.split("|")[3]
            for text in texts:
                if text.find("/") >= 0:
                    print(text)
                    print(pos)