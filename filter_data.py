#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

def filter_file(input_path: str, output_path: str, keywords: list[str]) -> None:
    """
    读取 input_path 文件，保留包含任意一个关键字的行，写入 output_path 文件。
    """
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            # 检查行中是否包含任意一个关键字
            if any(kw in line for kw in keywords):
                fout.write(line)

def main():
    parser = argparse.ArgumentParser(
        description="按关键字过滤文本行，将包含关键字的行输出到新文件。"
    )
    parser.add_argument(
        "input_file",
        help="要过滤的输入文本文件路径"
    )
    parser.add_argument(
        "keywords",
        nargs='+',
        help="一个或多个关键字，至少有一个匹配才保留行"
    )
    parser.add_argument(
        "-o", "--output",
        default="filtered.txt",
        help="输出文件路径，默认为 filtered.txt"
    )
    args = parser.parse_args()

    filter_file(args.input_file, args.output, args.keywords)
    print(f"已将包含关键字 {args.keywords} 的行写入：{args.output}")

if __name__ == "__main__":
    main()
