import tarfile
import os
def tar_folder(folder_path, archive_path):
    # 检查文件夹路径是否存在
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")
    # 创建 tar 文件
    with tarfile.open(archive_path, "w:tar") as tar:
        # 遍历文件夹中的所有文件和子文件夹
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.find(".pkl") > 0:
                    file_path = os.path.join(root, file)
                    # 添加文件到 tar 文件中
                    tar.add(file_path)
                    print(file)



if __name__ == "__main__":
    # 使用函数打包文件夹
    tar_folder("D:\\dataset\\VCTK\\wav48_silence_trimmed", "D:\\dataset\\VCTK\\archive.tar")