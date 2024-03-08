{% from "templates/head.py" import head %}
{{head()}}
{% from "dmultiprocess/control.py" import Control with context -%}
{% from "dmultiprocess/multiprocess_interface.py" import args, ProcessReturns, ProcessInitializer with context -%}

{% from "pwa/pwa.py" import PWAFunc with context -%}

class Logger():
    def __init__(self,logger_name):
        self.logger = logging.getLogger(logger_name)
        with open("config/logconfig_fit.json","r") as config:
            LOGGING_CONFIG = json.load(config)
            logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("fit")

{{args()}}

{% call ProcessInitializer(data_path="data/real_data",mc_path="data/mc_int") %}
{% endcall %}

{{ProcessReturns()}}

{% call PWAFunc() %}
{% endcall %}

{% call Control(jinja_fit_info.fit.ResultFile) %}
        num_seed=onp.random.randint(low=0, high=1000000, size=1, dtype='l')
        logger.info("random seed: {}".format(num_seed))
        onp.random.seed(num_seed)
        self.args_float = self.args_list[self.float_list]
        # self.compile_func()
        args_float = self.args_float
        result = self.thread_likelihood(self.args_float)
        print(result)
        min_fcn = [result]
        fvalue = [0]
        ferror = [0]
{% endcall %}