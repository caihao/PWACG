import os
import sys
foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)

from rendered_scripts import {{jinja_fit_info.batch.CodeScript|replace('.py','')}} as batch_object

if __name__ == '__main__':
    # cal_sig = batch_object.calculate_significance("config/generator_{{generator_id}}.json")
    # cal_sig.cycle_calculate()

    # cal_sig = batch_object.submit("config/generator_{{generator_id}}.json")
    # cal_sig.submit()

    cal_scan = batch_object.scan("config/generator_{{generator_id}}.json")
    # cal_scan.Loop()
    # cal_scan.scan_lh()
    # cal_scan.scan_frac()
    # cal_scan.cal_fraction_error()
    # cal_scan.stepBYstep()
    cal_scan.draw_all()
    # cal_scan.Iterate()
    # cal_scan.scan_select()

    # cal_sig = batch_object.calculate_branch("config/generator_{{generator_id}}.json")
    # cal_sig.cal_branch()

    # cal_sig = batch_object.sort_table()