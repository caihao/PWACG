#!/usr/bin/env python
# coding: utf-8

import jax.numpy as np
import numpy as onp

###########################################################
#                  dplex                                  # 
# 这个包主要用来加速jax的复数计算，jax优化了likelihood的复数计算 #
# 但是likelihood的导数复数计算速度还是极慢                    #
#                                                        #
##########################################################

# 复数张量与复数张量的爱因斯坦求和
def deinsum(subscript, aa, bb):
    real = np.einsum(subscript, aa[0], bb[0]) - np.einsum(subscript, aa[1], bb[1])
    imag = np.einsum(subscript, aa[0], bb[1]) + np.einsum(subscript, aa[1], bb[0])
    return np.stack([real, imag], axis=0)

# 复数张量与实数张量的爱因斯坦求和
def deinsum_ord(subscript, aa, bb):
    real = np.einsum(subscript, aa, bb[0])
    imag = np.einsum(subscript, aa, bb[1])
    return np.stack([real, imag], axis=0)

# 复数张量的模平方
def dabs(aa):
    return aa[0]**2 + aa[1]**2 # 因为是纵向叠加所以aa[0]是纵向上第一个数组

# 复数张量转换成dplex数据格式
def dtomine(aa):
    return np.stack([np.real(aa), np.imag(aa)], axis=0) # 组合而成的数组 axis=0 的第一个数组为实部，第二个数组为虚部

# 用于将计算得到的张量的实部与虚部组合成dplex的数据格式
def dconstruct(aa, bb):
    return np.stack([aa, bb], axis=0) # 纵向叠加数组 第一维为实部，第二维为虚部

# 实数张量与虚数张量的除法
def ddivide(a, bb):
    real = a * bb[0] / dabs(bb)
    imag = -a * bb[1] / dabs(bb)
    return np.stack([real, imag], axis=0)
