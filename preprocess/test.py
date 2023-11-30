import PyPDF2


def split_pdf(file_path, output_folder, output_name):
    # 打开PDF文件
    with open(file_path, 'rb') as file:
        # 创建一个PDF阅读器对象
        reader = PyPDF2.PdfFileReader(file)
        # 获取PDF文件的页数
        num_pages = reader.numPages
        # 循环遍历每一页并保存为单独的文件

        last_page = 1
        writer = PyPDF2.PdfFileWriter()
        for page in range(num_pages):
            writer.addPage(reader.getPage(page))

            if (page+1) % 50 == 0 or page == num_pages - 1:
                with open(output_folder + output_name + "-" + str(last_page) + '-' + str(page) + '.pdf', 'wb') as output:
                    writer.write(output)
                    last_page = page + 1
                    writer = PyPDF2.PdfFileWriter()


# 使用函数拆分PDF文件
split_pdf('D:\\test\\2024 • LEVEL 1 • VOLUME 6.pdf', 'D:\\test\\', '2024-LEVEL1-VOLUME6')
