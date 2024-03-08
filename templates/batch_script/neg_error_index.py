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
        mass_index = onp.array({{args_index_collection.mass}})
        width_index = onp.array({{args_index_collection.width}})
        flatte_index = onp.array({{args_index_collection.flatte}})
        # my_args = onp.sort(onp.append(onp.append(mass_index,width_index),flatte_index))
        # my_args = onp.array([0,1,2,7,9])
        # print(my_args)
        
        num_seed=onp.random.randint(low=0, high=1000000, size=1, dtype='l')
        logger.info("random seed: {}".format(num_seed))
        onp.random.seed(num_seed)
        self.args_float = self.args_list[self.float_list]
        self.compile_func()
        result_info = dict()

        args_float = self.args_float

        args_float_size = (args_float.shape)[0]
        print(args_float_size)
        my_hessian = onp.zeros([args_float_size,args_float_size])
        for x in range(args_float_size):
            v = onp.zeros(args_float_size)
            v[x] = 1.0
            my_hessian[:,x] = self.thread_hvp(args_float,v)
        print("hessian size",my_hessian.shape)
        print("hessian :",my_hessian)
        print("hessian diag:",onp.diag(my_hessian))

        hess_inv = onp.linalg.inv(my_hessian)
        print("hess_inv :",onp.diag(hess_inv))
        print("E :",my_hessian@hess_inv)

        hessian_inv = onp.zeros([args_float_size,args_float_size])
        for x in range(args_float_size):
            v = onp.zeros(args_float_size)
            v[x] = 1.0
            hessian_inv[:,x] = onp.linalg.solve(my_hessian,v)
        print(hessian_inv)

        # import mpmath as mmp
        # mmp.mp.dsp = 200
        # hess_mp = mmp.matrix(my_hessian.tolist())
        # hess_inv_mp = hess_mp**-1
        # mmp.nprint(hess_inv_mp,10)

        # nan_index = onp.where(onp.diag(hess_inv)<0)
        # points = onp.arange(0,100,0.5)
        # who = nan_index[0][0]
        # scan_nan = onp.zeros(points.shape[0])
        # print(self.args_float[who])
        # print(nan_index)
        # print(who)
        # x = self.args_float[who] - points.max()/2 + points
        # print(x)
        # for n,p in enumerate(x):
        #     self.args_float[who] = p
        #     # print(self.args_float[who])
        #     scan_nan[n] = self.thread_likelihood(self.args_float)
        # print(scan_nan)
        # self.args_float[who] = 31
        # print(self.thread_likelihood(self.args_float))
        # import matplotlib.pyplot as plt
        # plt.plot(x,scan_nan)
        # plt.savefig("scan_nan") 

        min_fcn = onp.diag(hess_inv)
        fvalue = [0]
        ferror = [0]
{% endcall %}