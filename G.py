# 本文件主要为该项目“全局变量”集合
#   文件头初始化的全局变量，在具体方法中赋值，随后由另一文件调用，出错，暂不明缘由
#   故建立独立文件初始化所有“全局变量”，之后都从该独立文件调用
# 以及复用率高的方法集合


import numpy as np
import math
import scipy
import pylab
#import time
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

import wx

# 可见光图像
Visimage = np.array([])
RGB = np.array([])
hsv = np.array([])
xyz = np.array([])
Lab = np.array([])

# Hyper图像
# global HyperData2D      # 3D to 2D, 'unfold'
# global HyperDataAve     # 平均吸光度
# global m, n, p          # 原始3D数据尺寸
HyperData2D = np.array([])
HyperDataAve = np.array([])
m, n, p = 0, 0, 0

# 光谱图像波长轴
WaveNumber = np.array([])

# fore，前景传递参数
fore = np.array([])
foreP = np.array([])

# CN，前景提取次数？
CN = 0

# 前景提取算法结果
iteration = np.array([])
homogeneity = np.array([])
distance = np.array([])
local = np.array([])
Otsu = np.array([])
comentropy = np.array([])

# 以threshold为阈值进行分割，计算分割类别的均值,数据量占比
# 目前针对numpy数组
def arr_mean2(arr_a, threshold):
    thp = arr_a >= threshold
    th = np.multiply(arr_a, thp)

    if np.sum(thp)==0:
        mean_h = 0
    else:
        mean_h = np.sum(th)/np.sum(thp)

    thp_r = np.sum(thp)/arr_a.size

    tlp = arr_a < threshold
    tl = np.multiply(arr_a, tlp)
    if np.sum(tlp)==0:
        mean_l = 0
    else:
        mean_l = np.sum(tl)/np.sum(tlp)
    tlp_r = np.sum(tlp)/arr_a.size

    return mean_h, mean_l, thp_r, tlp_r

# 归一化至指定区间
# mapminmax
def mapminmax_(a, min_, max_):
    b = np.min(a)
    c = np.max(a) - np.min(a)
    d = (max_ - min_)*(a - b) / c
    return d

# 前景提取算法6种
# Iteration算法
def Iteration(a, k1, k2, tol):
    # 阈值初始化
    T = k1* (a.min() + a.max())
    done = False

    # 进度条怎么加呢？不知道总次数。。。

    while ~done:
        meanTh, meanTl, p2, p1 = arr_mean2(a, T)

        # 计算新阈值
        Tn = k2 * (meanTh + meanTl)
        done = abs(T - Tn) < tol
        T = Tn
    del meanTl, meanTh, p2, p1
    del done

    # 以阈值T，二值化
    iteration = np.copy(a)
    iteration[iteration < T] = 0
    iteration[iteration >= T] = 1
    return iteration

# homogeneity
def Homogeneity(a, dt):
    Smin = -1
    T = 0
    p = a.min()
    q = a.max()
    m,n = a.shape

    # 进度条
    progressMax = dt+2
    progressdlg = wx.ProgressDialog("The Progress(Homogeneity)", "Time remaining", progressMax,
                                style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 1
    for TT in np.arange(p, q, (q - p) / 1000):
        ave2, ave1, p2, p1 = arr_mean2(a, TT)

        # 计算d1,d2,为两部分与各自均值的差的平方的和
        d1, d2 = -1, -1
        for ii in np.arange(0, m, 1):
            for jj in np.arange(0, n, 1):
                if a[ii, jj] >= TT:
                    d = (a[ii, jj] - ave2) ** 2
                    if d2 == -1:
                        d2 = d
                    else:
                        d2 = d2 + d
                else:
                    d = (a[ii, jj] - ave1) ** 2
                    if d1 == -1:
                        d1 = d
                    else:
                        d1 = d1 + d

        del ave1, ave2

        S = p1 * d1 + p2 * d2

        del p1, p2

        if Smin == -1:
            Smin = S
            T = TT
        else:
            if S < Smin:
                Smin = S
                T = TT

        count = count + 1
        if keepGoing and count < progressMax:
            keepGoing = progressdlg.Update(count)

    #print(T)
    # 以阈值T，二值化
    homogeneity = np.copy(a)
    homogeneity[a >= T] = 1
    homogeneity[a < T] = 0

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    progressdlg.Destroy()
    return homogeneity

# The maximum distance between classes
def Distance(a, dt):
    Smax = 0
    T = 0
    p = a.min()
    q = a.max()

    # 进度条
    progressMax = dt + 2
    progressdlg = wx.ProgressDialog("The Progress(max_distance)", "Time remaining", progressMax,
                                    style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 1

    for TT in np.arange(p, q, (p + q) / dt):
        aveH, aveL, p2, p1 = arr_mean2(a, TT)
        S = ((aveH - TT) * (TT - aveL)) / (aveH - aveL) ** 2
        if S > Smax:
            Smax = S
            T = TT

        count = count + 1
        if keepGoing and count < progressMax:
            keepGoing = progressdlg.Update(count)

    # 以阈值T，二值化
    distance = np.copy(a)
    distance[distance < T] = 0
    distance[distance >= T] = 1

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    progressdlg.Destroy()
    return distance

# Local Threshold
def LocalThreshold(a,ee,kk):
    # 对每个子区域，求阈值并二值化
    m, n = a.shape

    # 进度条
    progressMax = ee*kk + 2
    progressdlg = wx.ProgressDialog("The Progress(Local Thresholding)", "Time remaining", progressMax,
                                style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 1

    for ii in range(ee):
        for jj in range(kk):
            d = a[int(ii * m / ee):int((ii + 1) * m / ee), int(jj * n / kk):int((jj + 1) * n / kk)]
            # 以子区域均值为子区域阈值
            d[d < d.mean()] = 0
            d[d >= d.mean()] = 1

            count = count + 1
            if keepGoing and count < progressMax:
                keepGoing = progressdlg.Update(count)

    progressdlg.Destroy()
    return a

# Otsu
def otsu_(a):
    a = mapminmax_(a, 0, 255)
    max_g = 0
    T = 0

    # 进度条
    progressMax = 256 + 2
    progressdlg = wx.ProgressDialog("The Progress(Otsu)", "Time remaining", progressMax,
                                    style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 1

    for Th in range(0, 256):
        foreAve, bgAve, foreRatio, bgRatio = arr_mean2(a, Th)
        g = foreRatio*bgRatio*(foreAve-bgAve)*(foreAve-bgAve)
        if g > max_g:
            max_g = g
            T = Th

        count = count + 1
        if keepGoing and count < progressMax:
            keepGoing = progressdlg.Update(count)

    # 以T为阈值，二值化
    Otsu = a
    Otsu[Otsu < T] = 0
    Otsu[Otsu >= T] = 1

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    progressdlg.Destroy()
    return Otsu

# maximum entropy
def Comentropy(a, n):
    # 直方图图像
    h = pylab.hist(a.flatten(), n)  # h是tuple，元组

    hh = h[0]  # 取出h[0], 是每一级别的数据个数；h[1]是级别端点值(？)；h[2]似乎是所有的矩形(？)
    length = len(hh)

    H = np.arange(0, length - 1)

    # 进度条
    progressMax = length - 1 + 2
    progressdlg = wx.ProgressDialog("The Progress(max entropy)", "Time remaining", progressMax,
                                    style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 1

    # 遍历所有级别间隔点，分别获取间隔点两侧数据个数，求相关的值(形成向量H)
    for ii in np.arange(0, length - 1, 1):
        if hh[ii] != 0:
            p1 = sum(hh[0:ii + 1])
            print(p1)
            p2 = sum(hh[ii + 1:length])
            print(p2)
        else:
            continue

        H1 = -p1 * math.log(p1)
        H2 = -p2 * math.log(p2)

        H[ii] = H1 + H2
        print(ii)

        count = count + 1
        if keepGoing and count < progressMax:
            keepGoing = progressdlg.Update(count)

    # 对向量H，求最大值索引，可能有多个数，即第几个【前述的间隔点】处计算的相关的值最大
    p = np.argmax(H)  # np.argmax，求最值索引，但只有第一个

    # 向量p最大值有多个，则取第一个间隔最大值(？)
    # r = len(p)
    # dp = np.arange(0, r-1, 1)
    # if r-1:
    # for jj in np.arange(0, r-1, 1):
    #     dp[jj] = p[jj+1]-p[jj]
    # q = np.argmax(dp)
    # print(q)
    # pp = int(sum(hh[0: p*(q(0)+1)-1]))    # pp将被用作索引值，故强制转换为int
    # 向量p最大值仅一个，好办
    # else:
    pp = int(sum(hh[0: p - 1]))  # pp将被用作索引值，故强制转换为int
    #print(pp)

    # HyperDataAve，展开并排序
    m, n = a.shape
    k = np.reshape(a, [1, m * n])
    k = np.sort(k, 1)

    # 以阈值T，二值化
    if pp == 0:
        pass
    else:
        T = k.T[pp - 1]  # 若a是行向量，a.T可实现转置为列向量
        #print(T)

        comentropy = np.copy(a)
        comentropy[comentropy < T] = 0
        comentropy[comentropy >= T] = 1

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    progressdlg.Destroy()

    return comentropy

# 抠图相关-------------------------------------
#lsPointChoose = np.array([])      #列表
#tpPointChoose = ()                #元组
#pointCount = 0                    #计点数
#count = 0                         #记点击数
# 点数上限
#pointsMax = 6
pfore = np.array([])        # 传递变量

# _Limit
def _Limit(a, min_, max_):
    a = int(a)
    if a < min_:
        a = min_
    elif a > max_:
        a = max_
    else:
        a = a
    return a

# 抠图相关-------------------------------------

# 光谱预处理
ref = np.array([])
refN = np.array([])
HyperData = np.array([])
HyperDataN = np.array([])
WaveN = np.array([])
spEndL = 0
spEndR = 0
refp = np.array([])
HyperDatap = np.array([])
HyperDatapAve = np.array([])
SpList = []

# 预处理参数
# Baseline1
BaselineLambda = 0
BaselineP = 0
# airPLS Baseline
BaselineLambda = 0
BaselineOrder = 0
BaselineIter = 0
# S-G Derivative
DWidth = 0
DOrder = 0
Deriv = 0
# S-G Smoothing
SWidth = 0
SOrder = 0

# 预处理算法
# Baseline
# 算法太慢了。。。4w+条光谱好像要13小时多。。。
# 算法没问题，大家都推荐的，就是太慢了。。。
def Masymcorr(x, La, p):
    """
    Method for Baseline From Eilers Anal Chem 2005
    :param x:  data to Baseline
    :param La: Lambda = smoothness
    :param p:  p = asymmetry
    :return:   Baselined data
    """
    m, n = x.shape
    p = min((1-1e-10), p)   #keep p < 1
    SpeyeN = scipy.sparse.eye(n).toarray()     #python中稀疏矩阵需要.toarray()才能调用
    D = np.diff(SpeyeN, 2, 0)                  #D = diff(speye(n), 2);, diff(x,n,dim), python中dim=-1
    DtD = float(La) * np.dot(D.T, D)
    bg = np.copy(x)
    w = np.ones((n, 1))

    # 进度条
    progressMax = m + 2
    progressdlg = wx.ProgressDialog("The progress(Baseline-asymcorr)", "Time remaining", progressMax,
                                style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 1

    for jj in range(m):
        x1 = np.copy(x[jj, :]).reshape(-1, 1)
        if np.all(x1 == 0):
            pass
        else:
            for it in range(20):
                WW = scipy.sparse.spdiags(w.T, 0, n, n).toarray()      #python中稀疏矩阵需要.toarray()才能调用
                C = np.linalg.cholesky(WW + DtD)     # matlab chol(a) = python np.linalg.cholesky(a).T
                z = np.dot(np.linalg.inv(C.T), np.dot(np.linalg.inv(C), np.multiply(w, x1)))
                w_old = np.copy(w)
                w[x1<=z] = 1-p
                w[x1>z] = p
                if np.all(w_old == w):         #w_old == w,返回值是点对点布尔量，0或1; all([])是全为零吗？返回值也是布尔量
                    break
            bg[jj, :] = z.T
            print(jj)

        count = count + 1
        if keepGoing and count < progressMax:
            keepGoing = progressdlg.Update(count)

    xcorr = x -bg
    x = xcorr
    where_are_nan = np.isnan(x)
    #where_are_inf = np.isinf(a)
    x[where_are_nan] = 0
    #a[where_are_inf] = 0

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)
    progressdlg.Destroy()

    return x

# Baseline2
# 好像原理是Baseline1一样的
def baseline_als(y, lam, p):
    # P. Eilers和H. Boelens在2005年有一种名为“Asymmetric Least Squares Smoothing”的算法
    # 该论文是免费的，可以在google上找到。
    """
    :param y: data to Baseline
    :param lam: Lambda = smoothness
    :param p: p = asymmetry
    :return: Baselined data
    """

    m, n = y.shape
    p = min((1 - 1e-10), float(p))  # keep p < 1

    bg = np.copy(y)

    # 进度条
    progressMax = m + 2
    progressdlg = wx.ProgressDialog("The progress(Baseline-als)", "Time remaining", progressMax,
                                    style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 1

    for jj in range(m):
        x1 = np.copy(y[jj, :])#.reshape(-1, 1)
        if np.all(x1 == 0):
            pass
        else:
            D = scipy.sparse.csc_matrix(np.diff(np.eye(n), 2))
            w = np.ones(n)
            for i in range(20):
                W = scipy.sparse.spdiags(w, 0, n, n)
                Z = W + lam * np.dot(D, D.T)
                z = scipy.sparse.linalg.spsolve(Z, w*y)
                w_old = np.copy(w)
                w = p * (y > z) + (1-p) * (y < z)
                if np.all(w_old == w):         #w_old == w,返回值是点对点布尔量，0或1; all([])是全为零吗？返回值也是布尔量
                    break
            bg[jj, :] = z#.T

        count = count + 1
        if keepGoing and count < progressMax:
            keepGoing = progressdlg.Update(count)

    where_are_nan = np.isnan(bg)
    # where_are_inf = np.isinf(a)
    bg[where_are_nan] = 0
    # a[where_are_inf] = 0

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)
    progressdlg.Destroy()

    return bg

# Baseline3
# 这个似乎快了很多
def airPLS(X, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    output
        the fitted background vector
    '''

    ##modified for X(m,n) by Jiaqi, Mei, 20190425

    n, m = X.shape

    # 进度条
    progressMax = n + 2
    progressdlg = wx.ProgressDialog("The Progress(Baseline-airPLS)", "Time remaining", progressMax,
                                style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 1

    for jj in range(n):

        x = np.copy(X[jj, :])
        if np.all(x==0):
            pass
        else:

            #m = x.shape[0]
            w = np.ones(m)

            Z = np.zeros((n, m))

            for i in range(1, itermax + 1):
                z = WhittakerSmooth(x, w, lambda_, porder)
                d = x - z
                dssn = np.abs(d[d < 0].sum())
                if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
                    if (i == itermax): print('WARING max iteration reached!')
                    break
                w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
                w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
                w[0] = np.exp(i * (d[d < 0]).max() / dssn)
                w[-1] = w[0]
            print(jj)
            Z[jj, :] = z

        count = count + 1
        if keepGoing and count < progressMax:
            keepGoing = progressdlg.Update(count)

    #return Z
    x = X-Z
    where_are_nan = np.isnan(x)
    #where_are_inf = np.isinf(a)
    x[where_are_nan] = 0
    #a[where_are_inf] = 0

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    progressdlg.Destroy()
    return x

# Baseline3的配套
def WhittakerSmooth(x, w, lambda_, differences=1):
    '''
    Penalized least squares algorithm for background fitting
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    output
        the fitted background vector
    '''
    X = np.matrix(x)
    m = X.size
    i = np.arange(0, m)
    E = eye(m, format='csc')
    D = E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + lambda_ * D.T * D)
    B = csc_matrix(W * X.T)
    background=spsolve(A, B)
    x = np.array(background)
    where_are_nan = np.isnan(x)
    # where_are_inf = np.isinf(a)
    x[where_are_nan] = 0
    # a[where_are_inf] = 0
    return x

# Baseline4
def baseline_correction(X, block_size=101, polyorder=3, show_graph=False):
    """
    Permit to remove the baseline from data.
    It first realized an Otsu filtering on the data on the block_size,
    then realize the mean of the selected point. The minimum of this baseline
    is then substract to avoid negative value.
    Finally, the data is smooth using a savgol filter of block_size and polyorder parameters.
    Require skimage.filters and scipy.signal.

    https://github.com/leclercsimon74/baseline_correction/blob/master/baseline_correction.py

    Parameters
    ----------
    data : array like, 1d
        data where the correction is apported
    block_size : int, odd number
        block size for the scanning, need to be an odd number (not internally check).
        Default value is 101
    polyorder : int
        value for the smoothin using the savgol_filter, Default is 3
    show_graph : bool
        permit to display the different step of smoothing and final result in a matplotlib plot.
        Default is True
    Return
    --------
    corrected_data : 1D array
        data with the baseline substracted
    """

    #modified for X(m,n) by Jiaqi, Mei, 20190425

    import skimage.filters
    import scipy.signal

    m, n = X.shape
    XX = np.zeros((m, n))

    # 进度条
    progressMax = m + 2
    progressdlg = wx.ProgressDialog("The Progress", "Time remaining", progressMax,
                                style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 1

    for ii in range(m):
        data = np.copy(X[ii, :])
        if np.all(data == 0):
            pass
        else:
            baseline = np.zeros((n))
            for x in range(int(n)):
                min_half = int(block_size / 2)
                max_half = int(block_size / 2)
                if x - min_half < 0:
                    min_half = 0
                if x + max_half > len(data):
                    max_half = len(data)
                selected_data = data[x - min_half:x + max_half]
                thre = skimage.filters.threshold_otsu(np.asarray((selected_data)))  # detect peak in the block
                test = selected_data[selected_data < thre]  # remove the peak value
                baseline[x] = np.mean(test)

            baseline = baseline - np.min(baseline)
            smooth_baseline = scipy.signal.savgol_filter(baseline, block_size, polyorder)  # smooth the result
            corrected_data = data - smooth_baseline

            if show_graph:
                plt.plot(np.arange(len(data)), data, 'b', label='raw data')
                plt.plot(np.arange(len(data)), baseline, 'c', label='baseline1')
                plt.plot(np.arange(len(data)), smooth_baseline, 'g', label='baseline2')
                plt.plot(np.arange(len(data)), corrected_data, 'r', label='corrected')
                plt.legend()
                plt.show()

            # print(ii)
            XX[ii, :] = corrected_data
        count = count + 1
        if keepGoing and count < progressMax:
            keepGoing = progressdlg.Update(count)

    x = XX
    where_are_nan = np.isnan(x)
    # where_are_inf = np.isinf(a)
    x[where_are_nan] = 0
    # a[where_are_inf] = 0

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)
    progressdlg.Destroy()
    return x

# MSC
def Mmsc(x, rs):
    """
    Multiplicative Scatter Correction
    By Cleiton A. Nunes
    UFLA,MG,Brazil
    :param x: (samples x variables) spectra to correct
    :param refmean: (1 x variables) reference spectra (in general mean(x) is used)
    :return: (samples x variables)  corrected spectra
    """

    # 进度条
    progressMax = 6
    progressdlg = wx.ProgressDialog("The Progress(MSC)", "Time remaining", progressMax,
                                    style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 0

    m, n = x.shape
    cw = np.ones((1, n))
    #mz = np.array([])
    mz = np.hstack((cw.reshape(-1, 1), rs.reshape(-1, 1)))
    mm, nm = mz.shape
    wmz = np.multiply(mz, cw.reshape(-1, 1)*np.ones((1, nm)))
    wz = np.multiply(x, np.ones((m, 1))*cw)
    z = np.dot(wmz.T, wmz)

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    U, sigma, VT = np.linalg.svd(z)
    #sd = np.diag(s).T      # Python的SVD中sigma只返回了对角线元素，且舍去了0值
    sd = sigma

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    cn = 10**12

    ms = sd[0]/math.sqrt(cn)     # ms是一个数
    # cs = max(sd, ms) #尺寸不同，以最大尺寸为准，返回每个点最大值
    cs = sd    # sd是行向量
    if cs[0] >= ms:
        pass
    else:
        cs[0] = ms

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    #cz = U*(np.diag(cs))*VT      # 矩阵乘法，还是用np.dot靠谱
    cz = np.dot(np.dot(U, np.diag(cs)) ,VT)
    zi = np.linalg.inv(cz)
    #B = (zi*wmz.T*wz.T).T
    B = (np.dot(np.dot(zi, wmz.T), wz.T)).T

    x_msc = np.copy(x)

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)


    p = B[:, 0]
    kk1 = p.reshape(-1, 1) * np.ones((1, mm))
    x_msc = x_msc - kk1

    p = B[:, 1]
    kk2 = p.reshape(-1, 1)*np.ones((1, mm))
    kk2[kk2 == 0] = np.spacing(1)
    x_msc = x_msc / kk2

    x = x_msc

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    where_are_nan = np.isnan(x)
    # where_are_inf = np.isinf(a)
    x[where_are_nan] = 0
    # a[where_are_inf] = 0

    progressdlg.Destroy()

    return x

# SNV
def Msnv(x):
    """
    Standard Normal Variate
    By Cleiton A. Nunes
    UFLA,MG,Brazil

    :param x: (samples x variables) data to preprocess
    :return: (samples x variables) preprocessed data
    """

    # 进度条
    progressMax = 6
    progressdlg = wx.ProgressDialog("The Progress(SNV)", "Time remaining", progressMax,
                                    style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 0

    n = x.shape[1]
    # x的列均值，复制n列
    rmean = np.mean(x, 1)
    rmeant = np.tile(rmean.reshape(-1, 1), (1, n))

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    # dr^2的列均值
    dr = x - rmeant
    dr2 = np.multiply(dr, dr)
    dr2sum2 = n*np.mean(dr2, 1)

    count = count + 2
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    # 公式
    std_dr = np.sqrt(dr2sum2/(n-1))
    std_drt = np.tile(std_dr.reshape(-1, 1), (1, n))

    # 0值置换为最小浮点数精度值
    std_drt[std_drt == 0] = np.spacing(1)

    x_snv = dr / std_drt

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    # return
    x = x_snv

    where_are_nan = np.isnan(x)
    # where_are_inf = np.isinf(a)
    x[where_are_nan] = 0
    # a[where_are_inf] = 0

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    progressdlg.Destroy()
    return x

# AutoScale
def Mautoscale(x):
    """
    Autoscales matrix to zero mean and unit variance
    By Cleiton A. Nunes
    UFLA,MG,Brazil
    % [ax,mx,stdx] = auto(x)
    % output:
    % ax	autoscaled data
    % mx	means of dataspeye(n)
    % stdx	stantard deviations of data

    :param x: data to autoscale
    :return: autoscaled data
    """
    # 进度条
    progressMax = 4
    progressdlg = wx.ProgressDialog("The Progress(AutoScale)", "Time remaining", progressMax,
                                    style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 0

    m = x.shape[0]
    # x的行均值，复制m行
    mx = np.mean(x, 0)
    mxt = np.tile(mx, (m, 1))
    # x的标准差，复制(m,n)矩阵
    stdx = np.std(x, axis=0, ddof=1)
    stdxt = np.tile(stdx, (m, 1))

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    # 0值置换为最小浮点数精度值
    stdxt[stdxt == 0] = np.spacing(1)

    # 公式
    ax = (x-mxt)/stdxt
    # return
    x = ax

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    where_are_nan = np.isnan(x)
    # where_are_inf = np.isinf(a)
    x[where_are_nan] = 0
    # a[where_are_inf] = 0

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    progressdlg.Destroy()
    return x

# MeanCenter
def Mmeancenter(x):
    """
    Mean center scales matrix to zero mean
    By Cleiton A. Nunes
    UFLA,MG,Brazil
    % [mcx,mx] = center(x)
    % output:
    % mcx	mean center data
    % mx	means of data

    :param x: data to mean center
    :return:  mean center data
    """
    m = x.shape[0]
    # x的行均值，复制m行
    mx = np.mean(x, 0)
    mxt = np.tile(mx, (m, 1))
    # 公式
    ax = (x - mxt)
    # return
    x = ax

    where_are_nan = np.isnan(x)
    # where_are_inf = np.isinf(a)
    x[where_are_nan] = 0
    # a[where_are_inf] = 0

    return x

# S-G Smoothing & Derivative
def MsgSD(x, width, order, deriv):
    """
    Savitsky-Golay smoothing and differentiation
    By Cleiton A. Nunes
    UFLA,MG,Brazil

    :param x:  (samples x variables) data to preprocess
    :param width:  (1 x 1) number of points (optional, default=15)
    :param order:  (1 x 1) polynomial order (optional, default=2)
    :param deriv:  (1 x 1) derivative order (optional, default=0)
    :return: (samples x variables) preprocessed data
    """

    # 进度条
    progressMax = 7
    progressdlg = wx.ProgressDialog("The Progress(Savitsky-Golay)", "Time remaining", progressMax,
                                    style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 0

    m, n = x.shape
    #x_sg = np.copy(x)
    w = max(3, 1+2*round((width-1)/2))
    o = min(max(0, round(order)), 5, w-1)
    d = min(max(0, round(deriv)), o)

    p = (w-1)/2
    pp = np.arange(-p, p+1, 1)
    ppp = pp.reshape(-1, 1)*np.ones((1, 1+o))

    wo = np.ones((1, w)).reshape(-1, 1)*range(o+1)

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    # xc=((-p:p)'*ones(1,1+o)).^(ones(size(1:w))'*(0:o));   #点幂，点对点求幂
    pppm, pppn = ppp.shape
    xc = np.zeros((pppm, pppn))
    for row in range(pppm):
        for col in range(pppn):
            xc[row, col] = ppp[row, col]**wo[row, col]

    we = np.dot(np.linalg.pinv(xc), np.eye(w))

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    bb1 = np.ones((d, 1))*np.arange(1, o+1-d+1, 1)
    bb21 = np.arange(0, d, 1).reshape(-1, 1)
    bb22 = np.ones((1, o+1-d))
    bb2 = bb21*bb22
    bb = bb1+bb2

    b = np.prod(bb, 0)

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    # di=spdiags(ones(n,1)*we(d+1,:)*b(1),p:-1:-p,n,n);
    dis = np.dot(np.ones((n, 1)), we[d, :].reshape(1, -1))*b[0]
    disp = np.arange(p, -p-1, -1)
    di = scipy.sparse.spdiags(dis.T, disp, n, n)
    t = di.toarray()
    # python的spdiags与matlab的参数是不一样的。spdiags(B, diags, k, k)这里的B要用matlab的B参数的转置

    w1 = np.diag(np.array([b]))*we[d:o+1, :]    #matlab中i：i=i，python中i：i+1 =i   matlab[a,b]=python[a-1,b]

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    xca = xc[0:int(p)+1, 0:int(o-d)+1]        #索引值(index)一定要强制int！！！
    xcaw = np.dot(xca, w1)
    xcb = xc[int(p):int(w), 0:int(o-d)+1]
    xcbw = np.dot(xcb, w1)
    t[0:int(w), 0:int(p)+1] = xcaw.T
    t[int(n-w):int(n), int(n-p)-1:int(n)] = xcbw.T

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    x_sg = np.dot(x, t)
    x = x_sg

    where_are_nan = np.isnan(x)
    # where_are_inf = np.isinf(a)
    x[where_are_nan] = 0
    # a[where_are_inf] = 0

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    progressdlg.Destroy()
    return x

# Normalize
def Mnormalize(x, normtype):
    """
    NORMALIZ Normalize rows of matrix.
    This function can be used for pattern normalization, which is useful for
    preprocessing in some pattern recognition applications and also for
    correction of pathlength effects for some quantification applications.
    %  [ndat,norms] = normaliz(dat,normtype);
    %  inputs:
    %  dat = the data matrix
    %  normtype = the type of norm to use {default = 2}.
    %  The following are typical values of normtype:
    %                normtype       description              norm
    %                  1       normalize to unit area       sum(abs(dat))
    %                  2       normalize to unit LENGTH     sqrt(sum(dat^2))
    %                  inf     normalize to maximum value   max(dat)
    %  Generically,
    %      for row i of dat:
    %          norms(i) = sum(abs(dat(i,:)).^normtype)^(1/normtype)
    %  If (normtype) is specified then (out) must be included, although it can be empty [].

    % outputs:
    %    ndat = the matrix of normalized data where the rows have been normalized.
    %   norms = the vector of norms used for normalization of each row.

    :param x: the data matrix
    :return: the matrix of normalized data where the rows have been normalized.
    """
    dat = x

    #a.min()   # 无参，所有中的最小值
    #a.min(0)  # axis=0; 每列的最小值
    #a.min(1)  # axis=1；每行的最小值
    ntype = normtype
    norms = np.zeros((dat.shape[0], 1))

    # 进度条
    progressMax = dat.shape[0] + 2
    progressdlg = wx.ProgressDialog("The Progress(Normalize)", "Time remaining", progressMax,
                                    style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 1

    if ntype == 1 or ntype == 2:
        for ii in range(dat.shape[0]):
            norms[ii] = sum(np.abs(dat[ii, :])**ntype)**(1/ntype)
            if norms[ii] == 0:
                norms[ii] == np.spacing(1)
            dat[ii, :] = dat[ii, :] / norms[ii]

            count = count + 1
            if keepGoing and count < progressMax:
                keepGoing = progressdlg.Update(count)

    elif ntype == 3:
        dat = dat - np.tile(dat.min(1).reshape(-1, 1), (1, dat.shape[1]))
        for ii in range(dat.shape[0]):
            norms[ii] = max(dat[ii, :])
            if norms[ii] == 0:
                norms[ii] == np.spacing(1)
            dat[ii, :] = dat[ii, :] / norms[ii]

            count = count + 1
            if keepGoing and count < progressMax:
                keepGoing = progressdlg.Update(count)

    else:
        pass
        count = dat.shape[0]
        if keepGoing and count < progressMax:
            keepGoing = progressdlg.Update(count)

    x = dat

    where_are_nan = np.isnan(x)
    #where_are_inf = np.isinf(a)
    x[where_are_nan] = 0
    #a[where_are_inf] = 0

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)
    progressdlg.Destroy()
    return x


# Combine
Combine2DAve = np.array([])
Combine2D = np.array([])

# Cluster
labels = np.array([])
ClusterNum = 0
Epidermis = np.zeros((m, n))
VascularBundle = np.zeros((m, n))
Sclerenchyma = np.zeros((m, n))
Parenchyma = np.zeros((m, n))


# Fnnls
def lsqnonneg(C, d):
    '''
    A Python implementation of NNLS algorithm
    References:
    [1]  Lawson, C.L. and R.J. Hanson,
        Solving Least-Squares Problems, Prentice-Hall, Chapter 23, p. 161, 1974.
            https://github.com/stefanopalmieri/lsqnonneg/blob/master/lsqnonneg.py
    Linear least squares with nonnegativity constraints.
    (x, resnorm, residual) = lsqnonneg(C,d) returns the vector x that minimizes norm(d-C*x)
    subject to x >= 0, C and d must be real
    '''
    #modified for 2D ndarray d by Jiaqi, Mei, 20190425
    #C为参考光谱， d为待计算光谱，X为计算结果
    #C, d均为每列1条光谱

    eps = np.spacing(1)
    tol = 10 * eps * np.linalg.norm(C, 1) * (max(C.shape) + 1)
    Cm, Cn = C.shape
    dm, dn = d.shape

    # 进度条
    progressMax = dn + 2
    progressdlg = wx.ProgressDialog("The Progress(Fnnls)", "Time remaining", progressMax,
                                style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    pcount = 1

    if dm != Cm:      #波数轴不等
        pass
    else:
        X = np.zeros((dn, Cn))

        for ii in range(dn):
            P = []
            R = [x for x in range(0, Cn)]  # R ={list} [0,1,2,3,...,Cn-1],遍历C光谱条数
            x = np.zeros(Cn)       # x ={ndarray}[0,0,0,0,...,0], Cn个0
            resid = d[:, ii] - np.dot(C, x)
            w = np.dot(C.T, resid.reshape(-1, 1))

            count = 0

            # outer loop to put variables into set to hold positive coefficients
            while np.any(R) and np.max(w) > tol:
                j = np.argmax(w)
                P.append(j)
                R.remove(j)
                AP = np.zeros(C.shape)
                AP[:, P] = C[:, P]
                s = np.dot(np.linalg.pinv(AP), d[:, ii])
                s[R] = 0
                while np.min(s) < 0:
                    i = [i for i in P if s[i] <= 0]

                    alpha = min(x[i] / (x[i] - s[i]))
                    x = x + alpha * (s - x)

                    j = [j for j in P if x[j] == 0]
                    if j:
                        R.append(*j)
                        P.remove(j)

                    AP = np.zeros(C.shape)
                    AP[:, P] = C[:, P]
                    s = np.dot(np.linalg.pinv(AP), d[:, ii])
                    s[R] = 0
                x = s
                resid = d[:, ii].reshape(1, -1) - np.dot(C, x)
                w = np.dot(C.T, resid.reshape(-1, 1))
            # return (x, sum(resid * resid), resid)
            X[ii, :] = x
            # print(ii)

            pcount = pcount + 1
            if keepGoing and pcount < progressMax:
                keepGoing = progressdlg.Update(pcount)

    progressdlg.Destroy()
    return X

fnnlsR = np.array([])
#fnnlsRmin = []
#fnnlsRmax = []

wildcard_npz = "Numpy array Z(*.npz)|*.npz|" \
               "All files (*.*)|*.*"

wildcard_fig = "PNG files (*.png)|*.png|" \
               "All files (*.*)|*.*"



# wxPython可能有框架限制，对象A实例化对象B时，无法通过参数传递的方式将参数应用于对象B的初始化方法
# 目前发现两种可行的方法：
# 1.在类B的初始化方法之外建立一个“全局变量”，如a=None；
#    从对象A实例化对象B，如A.k=B();
#    从对象A为对象B的“全局变量”赋值，如A.k.a=1;
#       针对上述情况，若有对象C再次实例化对象B，B.a=1这一结论仍然存在，除非有对B.a另行赋值操作
# 2.在类B的初始化方法之外新建方法，如def setParent(self, parent)：self.parent = parent
#    从对象A实例化对象B，如A.k=B()
#    从对象A调用对象B的新建方法，如B.setParent(p), 则B.parent = p
#       针对上述情况，B.parent = p是该对象B的新增属性；若有对象C再次实例化对象B，B.parent这一属性不存在

# 矩阵左除，右除
# 左除：a/b = a*inv(b)
# 右除：a\b = inv(a)*b
# speye(m, n) % 生成m×n的单位稀疏矩阵；speye(n) % 生成n×n的单位稀疏矩阵——————matlab
# The functions spkron, speye, spidentity, lil_eye and lil_diags were removed from scipy.sparse.
# The first three functions are still available as scipy.sparse.kron, scipy.sparse.eye and scipy.sparse.identity.
#starttime = time.time()      #Python time time() 返回当前时间的时间戳（1970纪元后经过的浮点秒数）
#maxtime = 600                # 控制计算用时至多为600s

# 不要随意妄想暂停进程/线程
# 需要子窗口返回值(主窗口暂停执行)就把子窗口设定为Dialog
