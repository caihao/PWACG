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

{% call ProcessInitializer(data_path="data/real_data",mc_path="data/mc_truth") %}
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
        my_hessian = onp.zeros([args_float_size,args_float_size])
        for x in range(args_float_size):
            v = onp.zeros(args_float_size)
            v[x] = 1.0
            my_hessian[:,x] = self.thread_hvp(args_float,v)
        print("hessian :",my_hessian)
        ferror = onp.sqrt(onp.diag(onp.linalg.inv(my_hessian)))
        _ferror = onp.diag(onp.linalg.inv(my_hessian))
        print("ferror :",ferror)
        print("_ferror :",_ferror)
        print("sum hessian :",onp.sum(my_hessian))
        print("inv hessian",onp.linalg.inv(my_hessian))
        print("sum inv hessian",onp.sum(onp.linalg.inv(my_hessian)))

        # get only my need args correlation and error
        # re_arr = onp.zeros(args_float_size)
        # re_arr[my_args] = 1.0
        # re_matx = onp.einsum("i,j->ij",re_arr,re_arr)
        # print(re_matx.shape)
        # my_hessian = my_hessian*re_matx
        # mask = onp.all(np.isnan(my_hessian) | np.equal(my_hessian, 0), axis=1)
        # my_hessian = my_hessian[~mask]
        # mask = onp.all(np.isnan(my_hessian) | np.equal(my_hessian, 0), axis=0)
        # my_hessian = my_hessian[:,~mask]
        # ferror = ferror[my_args]
        # print(ferror)

        correlation = onp.linalg.inv(my_hessian)
        error2 = onp.einsum("i,j->ij",ferror,ferror)
        correlation = correlation
        print("correlation :",correlation)

        # args_float = args_float[my_args]
        check_list = list()
        # save og args
        check_list.append(args_float)
        for n in range({{info.fit.Cycles}}):
            # check = onp.random.multivariate_normal(mean=args_float,cov=correlation,check_valid='ignore')
            check = onp.random.multivariate_normal(args_float,correlation)
            check_list.append(check)
        check_array = onp.array(check_list)
        print(check_array.shape)
        print(onp.mean(onp.array(check_array)[:,0]))
        print(onp.mean(onp.array(check_array)[:,1]))
        print(onp.mean(onp.array(check_array)[:,2]))
        print(onp.mean(onp.array(check_array)[:,3]))
        print(onp.mean(onp.array(check_array)[:,4]))
        print(onp.mean(onp.array(check_array)[:,5]))
        print(onp.mean(onp.array(check_array)[:,6]))
        print(onp.mean(onp.array(check_array)[:,7]))
        print(onp.mean(onp.array(check_array)[:,8]))
        onp.save("output/draw/fraction_paras",check_list)

        fvalue = [0]
        ferror = [0]
        min_fcn = [0]
{% endcall %}