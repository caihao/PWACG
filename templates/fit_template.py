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
        theta_index = onp.array({{args_index_collection.theta}})
        const_index = onp.array({{args_index_collection.const}})
        mass_index = onp.array({{args_index_collection.mass}})
        width_index = onp.array({{args_index_collection.width}})
        mw_index = onp.array({{mw_index}})
        mw_range = {{mw_range}}
        # self.args_float[theta_index] = onp.pi/2.0
        # self.args_float[const_index] = onp.random.rand(const_index.shape[0])
        self.compile_func()
        min_fcn = 40000.0
        result_info = dict()
        for n in range({{info.fit.Cycles}}):
            {% if info.fit.random %}
            theta = 2*onp.pi*onp.random.rand(theta_index.shape[0])
            self.args_float[theta_index] = 1e-1*onp.cos(theta)
            self.args_float[const_index] = 1e-1*onp.sin(theta)
            disturb = 50.0
            logger.info("disturb of args_float: {}".format(0.5/disturb))
            self.args_float = self.args_float*((onp.random.rand(self.args_float.shape[0]) - 0.5) / disturb + 1.0)
            {% endif %}
            args_float = self.args_float
            t1 = time.time()
            res = minimize(fun=self.thread_likelihood, x0=args_float, jac=self.thread_grad_likelihood, hessp=self.thread_hvp, method="Newton-CG", callback=self.my_callback, options={"disp":False, "xtol":1e-8})
            t2 = time.time()
            logger.info("{0} {1} {2}".format("="*15,str(n),"="*15))
            logger.info("fcn: {}".format(res.fun))
            logger.info("success: {}".format(res.success))
            logger.info("message: {}".format(res.message))
            logger.info("Iterations: {}".format(res.nit))
            logger.info("Function evaluations: {}".format(res.nfev))
            logger.info("Gradient evaluations: {}".format(res.njev))
            logger.info("Hessian evaluations: {}".format(res.nhev))
            logger.info("time: {}".format(t2-t1))
            logger.info("{0}".format("="*33))
            result_info["result_"+str(n)] = {"fcn":res.fun,"success":res.success,"Iterations":res.nit,"FunEval":res.nfev,"GrdEval":res.njev,"HesEval":res.nhev,"time":t2-t1}
            values = res.x
            logger.info("result likelihood: {}".format(self.thread_likelihood(values)))
            fvalue = copy.deepcopy(self.args_list)
            for i, locate in enumerate(self.float_list):
                fvalue[locate] = values[i]

            {% if binding_point.goto0 %}
            # 这里的binding的参数回填没有加上value因为在代码生成后会自动加上value(保存的json产生的画图代码会加上value)
            fvalue[{{binding_point.goto0}}] = fvalue[{{binding_point.goto1}}]
            {% endif %}
            # if res.fun < min_fcn and res.success:
            if res.fun < min_fcn:
                min_fcn = res.fun

            args_float_size = (args_float.shape)[0]
            my_hessian = onp.zeros([args_float_size,args_float_size])
            for x in range(args_float_size):
                v = onp.zeros(args_float_size)
                v[x] = 1.0
                my_hessian[:,x] = self.thread_hvp(args_float,v)
            # print("eig :",onp.linalg.eig(my_hessian)[0])
            ferror = onp.sqrt(onp.diag(onp.linalg.inv(my_hessian)))
            fvalue_float = fvalue[self.float_list]
            error = onp.zeros((self.args_list.shape)[0])
            for i, locate in enumerate(self.float_list):
                error[locate] = ferror[i]
            {% if binding_point.goto0 %}
            error[{{binding_point.goto0}}] = error[{{binding_point.goto1}}]
            {% endif %}
            logger.info("error: {}".format(error))
            self.save_in_json(fvalue,error,"{{jinja_fit_info.fit.ResultFile}}",str(n),result_info)
        min_fcn = [min_fcn]
        logger.info("{0} the minist value {1} {0}".format("*"*4,min_fcn))
{% endcall %}