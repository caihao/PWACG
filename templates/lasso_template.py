{% from "templates/head.py" import head %}
{{head()}}
import ROOT
{% from "dmultiprocess/control.py" import Control with context -%}
{% from "dmultiprocess/multiprocess_interface.py" import args, ProcessReturns, ProcessInitializer with context -%}
{% from "pwa/pwa.py" import PWAFunc with context -%}

############################################################################
#                                                                          #
#  this script is to calculate significance of result json`s mods          #
#                                                                          #
############################################################################
class Logger():
    def __init__(self,logger_name):
        self.logger = logging.getLogger(logger_name)
        with open("config/logconfig_lasso.json","r") as config:
            LOGGING_CONFIG = json.load(config)
            logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("lasso")

{{args()}}

{% call ProcessInitializer(data_path="data/real_data",mc_path="data/mc_int") %}
{% endcall %}

{{ProcessReturns()}}

{% call PWAFunc() %}
{% endcall %}

{% call Control(jinja_fit_info.lasso.ResultFile) %}
        num_seed=onp.random.randint(low=0, high=1000000, size=1, dtype='l')
        logger.info("random seed: {}".format(num_seed))
        onp.random.seed(num_seed)
        self.args_float = self.args_list[self.float_list]
        theta_index = onp.array({{args_index_collection.theta}})
        const_index = onp.array({{args_index_collection.const}})
        # self.args_float[theta_index] = onp.pi/2.0
        # self.args_float[const_index] = onp.random.rand(const_index.shape[0])
        self.compile_func()
        min_fcn = 0
        for n in range({{info.fit.Cycles}}):
            {% if info.fit.random %}
            self.args_float[theta_index] = 2*onp.pi*onp.random.rand(theta_index.shape[0]) - onp.pi
            self.args_float[const_index] = -4.0
            disturb = 10.0
            logger.info("disturb of args_float: {}".format(0.5/disturb))
            args_float = self.args_float*((onp.random.rand(self.args_float.shape[0]) - 0.5) / disturb + 1.0)
            {% else %}
            args_float = self.args_float
            {% endif %}
            t1 = time.time()
            # res = minimize(fun=self.thread_likelihood, x0=args_float, method="BFGS", jac=self.thread_grad_likelihood, options={"disp": False,'gtol': 1e-09})
            res = minimize(fun=self.thread_likelihood, x0=args_float, jac=self.thread_grad_likelihood, hessp=self.thread_hvp, method="Newton-CG", callback=self.my_callback, options={"disp":False, "xtol":1e-8})
            t2 = time.time()
            logger.info("{0} {1} {2}".format("="*15,str(n),"="*15))
            logger.info("fcn: {}".format(res.fun))
            logger.info("successs: {}".format(res.success))
            logger.info("message: {}".format(res.message))
            logger.info("Iterations: {}".format(res.nit))
            logger.info("Function evaluations: {}".format(res.nfev))
            logger.info("Gradient evaluations: {}".format(res.njev))
            logger.info("Hessian evaluations: {}".format(res.nhev))
            logger.info("time: {}".format(t2-t1))
            logger.info("{0}".format("="*33))
            values = res.x
            logger.info("result likelihood: {}".format(self.thread_likelihood(values)))
            fvalue = copy.deepcopy(self.args_list)
            for i, locate in enumerate(self.float_list):
                fvalue[locate] = values[i]

            {% if binding_point.goto0 %}
            # 这里的binding的参数回填没有加上value因为在代码生成后会自动加上value(保存的json产生的画图代码会加上value)
            fvalue[{{binding_point.goto0}}] = fvalue[{{binding_point.goto1}}]
            {% endif %}
            self.save_in_json(fvalue,str(n))
            if res.fun < min_fcn:
                min_fcn = res.fun
        ferror = [min_fcn]
        logger.info("{0} the minist value {1} {0}".format("*"*4,ferror))
{% endcall %}