import os
import sys
foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)

from rendered_scripts import batch_object_kk as batch_object

if __name__ == '__main__':

    cal_sig = batch_object.submit("config/generator_{{generator_id}}.json")
    cal_sig.pull_run({{run_num}},{{begin}},{{end}})