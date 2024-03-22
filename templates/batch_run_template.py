import os
import sys
foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)

from rendered_scripts import {{jinja_fit_info.batch.CodeScript|replace('.py','')}} as batch_object

if __name__ == '__main__':
    # cal_sig = batch_object.calculate_significance("config/generator_{{generator_id}}.json")
    # cal_sig.cycle_calculate()

    cal_sig = batch_object.submit("config/generator_{{generator_id}}.json")
    cal_sig.submit()
    # cal_sig.submit_pull()

    # cal_scan = batch_object.scan("config/generator_{{generator_id}}.json")
    # cal_scan.cal_args()
    # cal_scan.cal_fraction_pull()