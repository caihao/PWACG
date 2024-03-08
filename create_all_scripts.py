import json
import os
import sys
import jinja2
import glob
from create_code import create_control

def checkDirectoryStructure():
    # 检查目录中是否存在指定的文件和子目录
    fatherDirs = ["output", "result_repo", "rendered_scripts", "run"]
    childDirs = ['output/fit/fit_result_combine', 'output/fit/fit_result_kk', 'output/fit/fit_result_pipi', 'output/error', 'output/pictures/partial_mods_pictures', 'output/draw', 'output/lasso', 'output/pull', 'output/select', 'output/significance']
    requiredDirs = fatherDirs + childDirs
    for subdir in requiredDirs:
        if not os.path.isdir(subdir):
            print(f"目录 {subdir} 不存在！创建新目录")
            os.system("mkdir -p {}".format(subdir))
    return True

if __name__ == "__main__":
    os.system("cp templates/scan_draw.py run/")
    checkDirectoryStructure()

    with open("config/generator_combine.json", encoding='utf-8') as f:
        dict_json = json.loads(f.read())
        create_combine = create_control.Create_Code(dict_json)
        create_combine.initial_prepare()
        create_combine.read_pwa("fit")
        create_combine.jinja_fit()

        create_combine.initial_prepare()
        create_combine.read_pwa("draw")
        create_combine.jinja_draw()

        create_combine.jinja_tensor()

    with open("config/generator_kk.json", encoding='utf-8') as f:
        dict_json = json.loads(f.read())
        create_kk = create_control.Create_Code(dict_json)
        create_kk.initial_prepare()
        create_kk.read_pwa("fit")
        create_kk.jinja_fit()

        create_kk.initial_prepare()
        create_kk.read_pwa("draw")
        create_kk.jinja_draw()

    with open("config/generator_pipi.json", encoding='utf-8') as f:
        dict_json = json.loads(f.read())
        create_pipi = create_control.Create_Code(dict_json)
        create_pipi.initial_prepare()
        create_pipi.read_pwa("fit")
        create_pipi.jinja_fit()

        create_pipi.initial_prepare()
        create_pipi.read_pwa("draw")
        create_pipi.jinja_draw()
