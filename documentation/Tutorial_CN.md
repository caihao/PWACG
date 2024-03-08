# Tutorial

### 程序运行环境配置

-   **ROOT** **installation**

    For any Linux distribution and MacOS, ROOT is available as a [conda package](https://anaconda.org/conda-forge/root/ "conda package"). To create a new conda environment containing ROOT and activate it, execute
    ```python
    $ conda config --set channel_priority strict
    $ conda create  --name <my-environment> python=3.9  # 选择一个合适的python版本 
    $ conda activate <my-environment> 
    $ conda install -c conda-forge root
    ```
    Setting `channel_priority` to `strict` is required to avoid conflicts on some platforms, see [the relevant conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html#strict-channel-priority "the relevant conda docs") for more information.

    The conda package uses C++17.

    More instructions about using this package are available in [this blog post](https://iscinumpy.gitlab.io/post/root-conda/ "this blog post").

    Please report any issues with the conda package [here](https://github.com/conda-forge/root-feedstock "here")
-   **JAX Conda installation**
    -   官方readme里的install

        [ GitHub - google/jax: Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more - GitHub - google/jax: Composable transformations of Python+NumPy programs: differentiate, ve... https://github.com/google/jax](https://github.com/google/jax " GitHub - google/jax: Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more - GitHub - google/jax: Composable transformations of Python+NumPy programs: differentiate, ve... https://github.com/google/jax")

        [Installing JAX — JAX documentation](https://jax.readthedocs.io/en/latest/installation.html "Installing JAX — JAX documentation")
-   **other package**
    -   jinja2
    -   numpy
    -   scipy
    -   pandas
    -   SciencePlots
        ```bash
        $ python -m pip install git+https://github.com/garrettj403/SciencePlots.git
        ```
        [ SciencePlots Format Matplotlib for scientific plotting https://pypi.org/project/SciencePlots/1.0.2/](https://pypi.org/project/SciencePlots/1.0.2/ " SciencePlots Format Matplotlib for scientific plotting https://pypi.org/project/SciencePlots/1.0.2/")

### PWACG使用手册

#### 运行拟合程序步骤

-   运行代码生成脚本
    > 代码生成用到的的配置文件在`config` 路径中，具体调参方法可以看后面的→调参方法
    ```bash
    $ python create_all_scripts.py
    ```
-   运行拟合程序
    > 以拟合$\psi(2s) → \phi KK$为例
    ```bash
    $ python run/fit_kk.py 
    ```
-   运行画图程序
    1.  还是要先运行一下代码生成脚本，因为画图的脚本也是用代码生成的
        ```bash
        $ python create_all_scripts.py
        ```
    2.  产生画图需要用到的数据
        > 因为拟合的结果是似然函数的参数，如果想要直观的展示拟合结果可以通过将MC数据输入似然函数计算出MC的weight，通过对MC的直方图加weight 就可以使MC数据具有拟合结果的形状。这一步就是用来产生weight的！
        ```bash
        $ python run/draw_wt_kk.py
        ```
    3.  画图
        > 画图结果保存在`result_repo/xxxx/pictures/partial_mods_pictures`中，其中`xxxx` 是时间戳
        ```bash
        $ python run/dplot_run_kk.py
        ```



#### 调参方法

> 注意！调完参数后一定要再次运行代码生成才能使调参生效 运行代码生成脚本

> 用来调参的`json`文件都在`config` 目录中，下面用拟合$\psi(2s) → \phi KK$举例。

-   **config/generator\_kk.json**
    -   代码生成的配置文件
    -   配置文件中每个key的功能：
        -   `id` 配置脚本的id
        -   `jinja_fit_info` 其内部按照功能模块分成了几类，如拟合脚本"fit"，批量提交作业脚本"batch"等。这些模块内保存的是代码生成用到的模板和输出的脚本的路径，以及功能模块需要保存的该功能的结果文件的路径。
        -   `jinja_draw_info` 与`jinjia_fit_info` 类似，只是这个是与画图相关的路径。
        -   `json_pwa` 分波分析模型配置文件的路径。
        -   上面的路径基本不需要修改。
        -   `annex_info` 需要关注的是"fit"模块：
            -   `use_weight` 是否使用含有weight修正过得真实数据
            -   `write` 拟合结果是否保存到文件中
            -   `Cycles` 拟合重复次数，每次之间没有关联
            -   `total_frac` fraction约束的目标fraction 值
            -   `boundary` 是否使用参数的边界约束功能
            -   `lambda_tfc` fraction约束的强度
            -   `random` 每次拟合开始时是否在初始参数上加入随机偏移
-   **config/parameters.json**
    > 多线程多GPU运行参数配置文件
    -   配置文件中每个key的功能：
        -   `parameters`  多GPU多线程调度的配置接口
            -   base 所有的其他模块都可以继承base里的参数，如果某一个模块（如“fit”）需要特别调整某一个参数可以在该某块下面重写参数，最终会使用的是最新的参数。
                > 我们目前实现了单机多卡的计算，使用的是数据并行的方式！
                -   *run\_config* ： 多进程多线程在多GPU上运行的配置（按默认设置即可）
                    -   total\_gpu\_id ： 例如 \[0，1]，这么写就是告诉程序计算机上有两个GPU，并且他们的序号是 cuda:0 和 cuda:1
                    -   processes\_gpus ：一个进程中可以使用的GPU数
                    -   max\_processes ：最大进程数，如果是1就是单进程
                    -   max\_processes\_memory  ：每个进程占用的最大显存
                    -   thread\_gpus ：线程可以使用的GPU数
                    -   threads\_in\_one\_gpu ：在一个GPU上跑的线程数
                -   *data\_config* ：数据切片，目的是为了配合在多GPU上跑而做的数据并行。
                    -   data\_slices：将数据切分的份数
                    -   mc\_slices：
                    -   mini\_run：
        -   `draw_config` 画图参数设置
        -   `CacheTensor` 预计算的张量计算模块设置
-   **config/pwa\_info\_kk.json**
    > 重点的配置文件，在这里设置分波分析模型！
    -   以 pwa\_info\_kk.json为例
    -   mod\_info ：每一个部分波都成为一个mod，mod\_info 是一个list。可以加的共振态列表[phikk和phipipi的共振态](phikk和phipipi的共振态_vBV77N162gET3QZEaJgbKD.md "phikk和phipipi的共振态")
        -   将mod\_info 中的一个mod拿出来举例： 参数范围设置 [加参数范围的方法](加参数范围的方法_32uuP5gFxfVwfFnLo4omGj.md "加参数范围的方法")
            ```json
            {
                "mod": "phif0_980_kk", 
                # 部分波的名字，phif0是振幅名字，980表示的是 f0(980) 这个共振态 
                
                "amp": "phif0_kk", 
                # 部分波使用的振幅
               
                "prop": {
                    "prop_phi": {
                        "name": "BW",
                        "paras": [
                            "phi_mass",
                            "phi_width",
                            "phi_kk"
                        ]
                    },
                # 传播子设置 phi 的部分
                "prop_f": {
                        "name": "flatte980",
                        "paras": [
                            "kk_f980_mass",
                            "kk_f980_g_kk",
                            "kk_f980_rg",
                            "f_kk"
                        ]
                    }
                },
                # 传播子设置 f 的部分

                "Sbc": {
                    "phi": "phi_kk",
                    "f": "f_kk"
                },
                # 输入的中间共振态名字
                
                "args": {
                    "phi_m": {
                        "value": 1.02, # 初始值
                        "name": "phi_mass",
                        "fix": true, # 是否设置成固定值
                        "error": 0.0  # 这个不用管，不影响拟合
                    },
                    "phi_w": {
                        "value": 0.004,
                        "name": "phi_width",
                        "fix": true,
                        "error": 0.0
                    },
                    # phi 的质量宽度都是fix住的不需要拟合
                    "kk_f980_m": {
                        "value": 0.9791907429163911,
                        "name": "kk_f980_mass",
                        "range": [
                            0.98,
                            10.0
                        ],
                        # 设置范围 [中心值，约束强度] 
                        "error": 0.0018985343438633919
                    },
                    "kk_g_kk": {
                        "value": -0.09047962086358292,
                        "name": "kk_f980_g_kk",
                        "error": 0.0056502434500060265
                    },
                    "kk_rg": {
                        "value": 7.954832705543948,
                        "name": "kk_f980_rg",
                        "error": 0.8648589371836074
                    },
                    # kk_f980_m, kk_g_kk 和 kk_rg 这三个参数是prop_f中flatte980的参数
                    # 除了f980用的是flatte980，大部分部分波都是用的普通的Breit–Wigner，
                    # Breit–Wigner参数使用的是质量和宽度。
                    "kk_f980_c1": {
                        "value": 0.1,
                        "name": "kk_f980_const",
                        "fix": true,
                        "error": 0.0
                    },
                    "kk_f980_c2": {
                        "value": 0.10272328310095176,
                        "name": "kk_f980_const",
                        "error": 0.002280967551017021
                    },
                    "kk_f980_t1": {
                        "value": 0.1,
                        "name": "kk_f980_theta",
                        "fix": true,
                        "error": 0.0
                    },
                    "kk_f980_t2": {
                        "value": 0.039981856462107265,
                        "name": "kk_f980_theta",
                        "error": 0.0018185031895430884
                    }
                }
            }
            ```



#### 批量运行

> **超有用的批量运行脚本！**

-   运行
    ```bash
    $ python run/batch_kk.py
    ```
-   选择批量函数修改 `templates/batch_run_template.py`，这里使用的是在超算批量提交作业的函数
    ```python
    if __name__ == '__main__':
        # cal_sig = batch_object.calculate_significance("config/generator_{{generator_id}}.json")
        # cal_sig.cycle_calculate()

        cal_sig = batch_object.submit("config/generator_{{generator_id}}.json")
        cal_sig.submit()

        # cal_scan = batch_object.scan("config/generator_{{generator_id}}.json")
        # cal_scan.Loop()
        # cal_scan.scan_lh()
        # cal_scan.scan_frac()
        # cal_scan.cal_fraction_error()
        # cal_scan.stepBYstep()
        # cal_scan.draw_all()
        # cal_scan.Iterate()

        # cal_sig = batch_object.calculate_branch("config/generator_{{generator_id}}.json")
        # cal_sig.cal_branch()

        # cal_sig = batch_object.sort_table()
    ```
    -   功能列表：
        -   批量提交作业到超算 submit
            -   在`templates/batch_template.py`里修改`self.cgpwa_dir`的值为PWACG在超算上的绝对路径
                ```python
                class submit(base_batch):
                    def __init__(self, generator_path):
                        super().__init__(generator_path)
                        self.generator_init(self.generator_dict)
                        self.initial_prepare()
                        self.read_pwa("fit")
                        self.cgpwa_dir = "/xxxxx/"#这里修改为相应的路径
                ```
        -   批量运行扫描形式的作业 scan
        -   batch\_script/xxxx-pwa.sbatch，batch\_script/xxxx-pwa.sbatch.conda，batch\_script/xxxx-pwa.sbatch.singularity这三个文件有一部分内容需要修改：
            -   batch/xxxx-pwa.sbatch
                ```python
                singularity exec --nv --bind /project/whoami,/scratch/whoami /scratch/xxxx/singularity/jax.sif python3 {{run_file}}
                ```
            -   batch/xxxx-pwa.sbatch.conda
                ```python
                source /home/xxxx/project/xxxx/anaconda3/bin/activate
                # conda activate jax_env删掉

                ```
            -   pwa.sbatch.singularity
                ```python
                singularity exec --nv --bind /project/`whoami`,/scratch/`whoami` /scratch/xxxx/singularity/jax.sif python3 {{run_file}}
                ```
-   **批量运行脚本参数修改**
    -   扫描fraction的范围在./template/batch\_template.py中修改
        ```python
        def submit(self):
                print("begain to submit job to slurm")
                
                frac_tfc = onp.arange(1.1,1.4,0.05) #在此处修改范围 

                # with open("output/fit/fit_result_kk/fit_result_kk_2/result_info.json",encoding='utf-8') as f:
                #     mydict = json.loads(f.read())
                # arr_lh = onp.array([float(mydict["result_"+str(j)]["fcn"]) for j in range(60)])
                # min_index = list(onp.where(arr_lh == arr_lh.min()))[0][0]

                # for N, tfc in enumerate(frac_tfc):
                #     for n, scanX in mod_info:
                #         self.generatorJob(N,tfc,n,scanX)

                for n, scanX in enumerate(frac_tfc):
                    self.ordinaryJob(n,scanX)
        ```
-   **图像绘制**

    在本地相应的虚拟环境中运行templates/scan\_draw\.py&#x20;

    **注意：**scan\_draw\.py文件中的绘图参数也需要根据fraction的范围进行修改，保存拟合结果的路径也需要修改
    ```python
    def read_scan_tfc_result(filename):
        target_character = "the minist value "
        R0_fcn = list()
        for _n, n in enumerate(frac_tfc):
            R0_fcn.append(find_character_in_file("./R_1"+filename+"/fit_result_"+str(_n)+"/fit_{0}.log".format(_n), target_character))
         #这里和下面的scanSF_dir_list一起修改，修改为当前保存拟合结果的路径 
        #filename=scanSF_dir_list
        draw_R0 = Draw(frac_tfc,R0_fcn)
        return draw_R0
    ```
    ```python
        ...
        # fix dof in different sf
        scanSF_dir_list = ["/fit"]
        # frac_tfc = onp.round(onp.arange(1.0,1.3,0.05), decimals=2)
        frac_tfc = onp.round(onp.arange(0.9,1.4,0.05), decimals=2) #修改fraction的范围 
        print(frac_tfc)
        ...
    ```
    绘制的结果将会保存在./output/pictures/scan/中

    **再次运行****scan\_draw\.py****会将上次运行的结果覆盖，注意保存**



#### 通过weight抽样MC

> 通过拟合或者设置好的json配置文件，得到weight，通weight来抽样出json文件中的共振态形状

-   **运行**
    ```bash
    $ python run/select_kk.py
    ```
-   **说明**
    -   输入的是output中的拟合结果json文件
    -   输出在`data/select/` 中，并替换了`data/real_data`中的npy文件
-   **代码解析**

    **render\_scripts/select\_object\_kk.py**



### 联合拟合

用于同一母粒子不同的两个衰变过程，存在相同的中间共振态的情况。联合的意思就是将两个衰变中将相同的共振态的质量和宽度设置为同一个值。

-   使用方法
    ```bash
    $ python run/fit_combine.py
    ```
-   配置
    -   config/pwa\_info\_kk.json 和 config/pwa\_info\_pipi.json&#x20;
    -   config/pwa\_info\_ctrl.json ， 例如：
        ```json
        {
            "external_binding":
            {
                "pipi_f980_m":{"point":"kk_f980_m","value":0},
                "pipi_g_pipi":{"point":"kk_g_kk","value":0},
                "pipi_rg":{"point":"kk_rg","value":0},

                "pipi_f500_m":{"point":"kk_f500_m","value":0},
                "pipi_f500_w":{"point":"kk_f500_w","value":0},

                "pipi_f1370_m":{"point":"kk_f1370_m","value":0},
                "pipi_f1370_w":{"point":"kk_f1370_w","value":0},

                "pipi_f1500_m":{"point":"kk_f1500_m","value":0},
                "pipi_f1500_w":{"point":"kk_f1500_w","value":0}
            }
        }
        ```
        -   例如`"pipi_f980_m":{"point":"kk_f980_m","value":0}` 表示的是：

            pipi\_f980\_m = kk\_f980\_m + 0 ， 其中 0 是 “value”的值表示偏置多少。这样就将这两个变量变成了一个变量，只有kk\_f980\_m是自由的。


-   **振幅张量和中间态四动量**
    > 将上一步得到的numpy文件的四动量数据计算成分波分析程序要用到的振幅和中间态四动量，用到的代码在目录`PWACG/Tensor/`
    在[RunCacheTensor.py](http://RunCacheTensor.py "RunCacheTensor.py") 里修改需要运行的
    ```bash
    # 产生新脚本
    $ python create_all_scripts.py 

    # 运行
    $ python run/RunCacheTensor.py
    ```



