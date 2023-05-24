import os
import csv


import os

# 定义函数，修改文件名
def rename_files(path, old_str, new_str):
    for file in os.listdir(path):
        # 判断是否为文件
        if os.path.isfile(os.path.join(path, file)):
            # 获取新旧文件名
            old_name = os.path.join(path, file)
            new_name = os.path.join(path, file.replace(old_str, new_str))
            # 修改文件名
            os.rename(old_name, new_name)
            print(f'{old_name} --> {new_name}')

# 调用函数，修改文件名
path = '../input/Spirals/training/HC'
old_str = 'jpg'   # 指定要替换的旧字符串
new_str = 'png'   # 指定新字符串
rename_files(path, old_str, new_str)



