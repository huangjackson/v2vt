# MIT License
#
# Copyright (c) 2024 RVC-Boss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re


def str2bool(str):
    return True if str.lower() == 'true' else False


def get_newest_ckpt(string_list):
    # 定义一个正则表达式模式，用于匹配字符串中的数字
    pattern = r'epoch=(\d+)-step=(\d+)\.ckpt'

    # 使用正则表达式提取每个字符串中的数字信息，并创建一个包含元组的列表
    extracted_info = []
    for string in string_list:
        match = re.match(pattern, string)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))
            extracted_info.append((epoch, step, string))
    # 按照 epoch 后面的数字和 step 后面的数字进行排序
    sorted_info = sorted(
        extracted_info, key=lambda x: (x[0], x[1]), reverse=True)
    # 获取最新的 ckpt 文件名
    newest_ckpt = sorted_info[0][2]
    return newest_ckpt


# 文本存在且不为空时 return True
def check_txt_file(file_path):
    try:
        with open(file_path, 'r') as file:
            text = file.readline().strip()
        assert text.strip() != ''
        return text
    except Exception:
        return False
    return False
