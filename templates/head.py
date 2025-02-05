{% macro head() %}
import copy
import json
import logging
import logging.config
import os
import sys
import re
import time
import glob
from functools import partial
from multiprocessing import Array, Barrier, Lock, Manager, Pipe, Process, Value
from threading import Thread

import jax.experimental.host_callback as jax_exper
import jax.numpy as np
import numpy as onp
from dlib import dplex
from jax import device_put
from jax import devices as jdevices
from jax import grad, hessian, jit, jvp, value_and_grad, vmap
from jax import config
# from jax.experimental.host_callback import id_print
from scipy.optimize import minimize

from iminuit import minimize as i_minimize
import pynvml

{% endmacro %}
