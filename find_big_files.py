#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_big_files.py —— 找出占用空间最大的文件和文件夹（只扫描，不删除任何东西）

用法（用你本地的 Python 3.9.13 运行）：
    python3.9.13\\python.exe find_big_files.py  D:\\bysj  20
参数：
    第1个：要扫描的目录（默认当前目录）
    第2个：各显示前 N 个最大的文件/目录（默认 20）
"""
import os
import sys
from collections import defaultdict


def human(num):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def scan(root, top_n=20, skip_dirs=(".git", "__pycache__", "python3.9.13")):
    root = os.path.abspath(root)
    direct_files = defaultdict(int)  # 目录 -> 直接包含的文件字节数
    children = defaultdict(list)     # 目录 -> 直接子目录列表
    files = []                       # (size, path)

    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        # 跳过指定的目录（.git / 缓存 / 本地Python安装）
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        total = 0
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                sz = 0
            files.append((sz, fp))
            total += sz
        direct_files[dirpath] = total
        children[dirpath] = [os.path.join(dirpath, d) for d in dirnames]

    # 自底向上累加目录总大小
    sizes = dict(direct_files)
    for d in sorted(direct_files, key=lambda p: p.count(os.sep), reverse=True):
        for sub in children[d]:
            sizes[d] += sizes.get(sub, 0)

    print(f"扫描完成：{root}\n")

    print("=" * 70)
    print(f"最大的 {top_n} 个文件：")
    print("=" * 70)
    for sz, fp in sorted(files, reverse=True)[:top_n]:
        print(f"{human(sz):>12}  {fp}")

    print()
    print("=" * 70)
    print(f"最大的 {top_n} 个目录：")
    print("=" * 70)
    for d, sz in sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)[:top_n]:
        print(f"{human(sz):>12}  {d}")

    print()
    print(f"总大小（不含已跳过目录）：{human(sizes.get(root, 0))}")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    scan(root, top)
