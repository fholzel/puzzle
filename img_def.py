
#
# array multiple execution time
#
# np.multiple(array, value) : 1
# array * value             : 1
# np.dot(array, value)      : 2
#

import multiprocessing as mp
import numpy as np
import copy  as cp
import time
from scipy import signal as signal
from itertools import product

# 2 sides of polygon must touch box
# 2 sides of polygon must touch box
def img_create_polygon(image, size, x, y, vertices, debug = 0):
    tstart = time.time()
    xmin = min(vertices[:,0])
    ymin = min(vertices[:,1])
    xmax = max(vertices[:,0])
    ymax = max(vertices[:,1])
    flag = ''
    cntcc = 0
    cntrr = 0
    result = []
    if debug >= 1:
        print("x {:8d}:xmin {:8d}:xmax {:8d}:y {:8d}:ymin {:8d}:ymax {:8d}".format(x, xmin, xmax, y, ymin, ymax))
    # skip if touching all 4 sides
    if xmin == x and xmax == x + size and ymin == y and ymax == y + size:
        return
    # vertical
    if xmin == x and xmax == x + size and ymin != y and ymax != y + size:
        flag = 'V'
        ax0, ay0, bx0, by0 = np.sort([vert for vert in vertices if vert[0] == x],        axis=0).reshape((1,4)).tolist()[0]
        ax1, ay1, bx1, by1 = np.sort([vert for vert in vertices if vert[0] == x + size], axis=0).reshape((1,4)).tolist()[0]
    # top and left
    if xmin == x and xmax != x + size and ymin == y and ymax != y + size:
        flag = 'V'
        ax0, ay0, bx0, by0 = np.sort([vert for vert in vertices if vert[0] == x],        axis=0).reshape((1,4)).tolist()[0]
        ax1, ay1, bx1, by1 = np.sort([vert for vert in vertices if vert[1] == y],        axis=0).reshape((1,4)).tolist()[0]
    # bottom and right
    if xmin != x and xmax == x + size and ymin != y and ymax == y + size:
        flag = 'V'
        ax0, ay0, bx0, by0 = np.sort([vert for vert in vertices if vert[0] == x + size], axis=0).reshape((1,4)).tolist()[0]
        ax1, ay1, bx1, by1 = np.sort([vert for vert in vertices if vert[1] == y + size], axis=0).reshape((1,4)).tolist()[0]
    if flag == 'V':
        ax0, bx0, ax1, bx1 = [px - x for px in [ax0, bx0, ax1, bx1]]
        ay0, by0, ay1, by1 = [py - y for py in [ay0, by0, ay1, by1]]
        dxa, dya, dxb, dyb = [c - d for c,d in [(ax0, ax1), (ay0, ay1), (bx0, bx1), (by0, by1)]]
        ma, mb             = [c / d for c,d in [(dya, dxa), (dyb, dxb)]]
        ba = y + ay0 - (ma * ax0)
        bb = y + by0 - (mb * bx0)
        if debug >= 1:
            print("V0 : {:2d}: {:2d}: {:2d}: {:2d}: dxa {:2d}: dya {:2d}: ma {:8.3f}: ba {:8.3f}".format(ax0, ay0, bx0, by0, dxa, dya, ma, ba))
            print("V1 : {:2d}: {:2d}: {:2d}: {:2d}: dxb {:2d}: dyb {:2d}: mb {:8.3f}: bb {:8.3f}".format(ax1, ay1, bx1, by1, dxb, dyb, mb, bb))
        for a in range(x, x + size + 1):
            za = int(round((a - x) * ma + ba))
            zb = int(round((a - x) * mb + bb))
            cntrr = 0
            for b in range(y, y + size + 1):
                if b >= za and b <= zb:
                    image[a,b] = 1
                    cntcc += 1
                    cntrr += 1
            result.append(cntrr)
        if debug >= 2:
            tend = time.time()
            trun = round((tend - tstart) * 1000)
            print("cntcc : {:8d} : result : {} : trun : {:8d}".format(cntcc, result, trun))
    # horizontal
    if ymin == y and ymax == y + size and xmin != x and xmax != x + size:
        flag = 'H'
        ax0, ay0, bx0, by0 = np.sort([vert for vert in vertices if vert[1] == y]       , axis=0).reshape((1,4)).tolist()[0]
        ax1, ay1, bx1, by1 = np.sort([vert for vert in vertices if vert[1] == y + size], axis=0).reshape((1,4)).tolist()[0]
    # top and right
    if xmin == x and xmax != x + size and ymin != y and ymax == y + size:
        flag = 'H'
        bx0, by0, ax0, ay0 = np.sort([vert for vert in vertices if vert[0] == x],        axis=0).reshape((1,4)).tolist()[0]
        ax1, ay1, bx1, by1 = np.sort([vert for vert in vertices if vert[1] == y + size], axis=0).reshape((1,4)).tolist()[0]
    # bottom and left
    if xmin != x and xmax == x + size and ymin == y and ymax != y + size:
        flag = 'H'
        bx0, by0, ax0, ay0 = np.sort([vert for vert in vertices if vert[0] == x + size], axis=0).reshape((1,4)).tolist()[0]
        ax1, ay1, bx1, by1 = np.sort([vert for vert in vertices if vert[1] == y],        axis=0).reshape((1,4)).tolist()[0]
    if flag == 'H':
        ax0, bx0, ax1, bx1 = [px - x for px in [ax0, bx0, ax1, bx1]]
        ay0, by0, ay1, by1 = [py - y for py in [ay0, by0, ay1, by1]]
        dxa, dya, dxb, dyb = [c - d for c,d in [(ax0, ax1), (ay0, ay1), (bx0, bx1), (by0, by1)]]
        ma, mb             = [c / d for c,d in [(dxa, dya), (dxb, dyb)]]
        ba = x + ax0 - (ma * ay0)
        bb = x + bx0 - (mb * by0)
        if debug >= 1:
            print("H0 : {:2d}: {:2d}: {:2d}: {:2d}: dxa {:2d}: dya {:2d}: ma {:8.3f}: ba {:8.3f}".format(ax0, ay0, bx0, by0, dxa, dya, ma, ba))
            print("H1 : {:2d}: {:2d}: {:2d}: {:2d}: dxb {:2d}: dyb {:2d}: mb {:8.3f}: bb {:8.3f}".format(ax1, ay1, bx1, by1, dxb, dyb, mb, bb))
        for b in range(y, y + size + 1):
            za = int(round((b - y) * ma + ba))
            zb = int(round((b - y) * mb + bb))
            cntrr = 0
            for a in range(x, x + size + 1):
                if a >= za and a <= zb:
                    image[a,b] = 1
                    cntcc += 1
                    cntrr += 1
            result.append(cntrr)
        if debug >= 2:
            tend = time.time()
            trun = round((tend - tstart) * 1000)
            print("cntcc : {:8d} : result : {} : trun : {:8d}".format(cntcc, result, trun))
    return

# anything less than ZB -> 0
# anything more than ZW -> 255
def filter_gray_1(pixel, ZB):
    if pixel < ZB:
        return 1
    else:
        return 0
func_filter_gray_1  = np.vectorize(filter_gray_1)
def img_filter_gray(image, ZB=128, ZW=127):
    tstart = time.time()
    newimage = func_filter_gray_1(image, ZB)
    cntww,cntbb = [(newimage==cnt).sum() for cnt in (0,1)]
    tend = time.time()
    trun = round((tend - tstart) * 1000)
    print("cntbb : {:8d} : cntww : {:8d} : trun : {:8d}".format(cntbb, cntww,trun))
    return newimage

# filter noise : any 1 pixel alone in window 3x3 : value = 1
def filter_black_1(pixel):
    if pixel > 2:
        return 1
    else:
        return 0
func_filter_black_1 = np.vectorize(filter_black_1)
def img_filter_one_black(image):
    tstart = time.time()
    ones3x3 = np.ones((3,3), dtype=int)
    tmpimage0 = signal.convolve2d(image, ones3x3, mode='same')
    tmpimage1 = image * tmpimage0
    newimage = func_filter_black_1(tmpimage1)
    cntww,cntbb = [(newimage==cnt).sum() for cnt in (0,1)]
    tend = time.time()
    trun = round((tend - tstart) * 1000)
    print("cntbb : {:8d} : cntww : {:8d} : trun : {:8d}".format(cntbb, cntww,trun))
    return newimage

# filter noise : any 1 pixel along edge in window 3x3 : value = center + edge
def filter_bin_1(pixel, binA, binB, binC, binD):
    if pixel in [binA, binB, binC, binD]:
        return 1
    else:
        return 0
func_filter_bin_1 = np.vectorize(filter_bin_1)
def img_filter_one_edgeA(image):
    tstart = time.time()
    onesbin = np.array([2**z for z in range(9)]).reshape((3,3))
    binA = sum(onesbin[0:3,0]) + onesbin[1,1]
    binB = sum(onesbin[0:3,2]) + onesbin[1,1]
    binC = sum(onesbin[0,0:3]) + onesbin[1,1]
    binD = sum(onesbin[2,0:3]) + onesbin[1,1]
    tmpimage0 = func_filter_bin_1(signal.convolve2d(image, onesbin, mode='same'), binA, binB, binC, binD)
    newimage = image - tmpimage0
    cntww,cntbb = [(newimage==cnt).sum() for cnt in (0,1)]
    tend = time.time()
    trun = round((tend - tstart) * 1000)
    print("cntbb : {:8d} : cntww : {:8d} : trun : {:8d}".format(cntbb, cntww,trun))
    return newimage

# filter noise : any 1 pixel along edge in window 3x3 : value = center + middle edge + corner
def filter_bin_2(pixel, binA0, binB0, binC0, binD0, binA1, binB1, binC1, binD1):
    if pixel in [binA0, binB0, binC0, binD0, binA1, binB1, binC1, binD1]:
        return 1
    else:
        return 0
func_filter_bin_2 = np.vectorize(filter_bin_2)
def img_filter_one_edgeB(image):
    tstart = time.time()
    onesbin = np.array([2**z for z in range(9)]).reshape((3,3))
    bin_x = [[0,0],[0,0],[0,1],[1,2],[2,2],[2,2],[2,1],[1,0]]
    bin_y = [[0,1],[1,2],[2,2],[2,2],[2,1],[1,0],[0,0],[0,0]]
    bin_xy = zip(bin_x,bin_y)
    bin_list = []
    for x,y in bin_xy:
        bin_list.append(onesbin[x[0],y[0]] + onesbin[x[1],y[1]] + onesbin[1,1])
    binA0, binB0, binC0, binD0, binA1, binB1, binC1, binD1 = bin_list
    tmpimage0 = func_filter_bin_2(signal.convolve2d(image, onesbin, mode='same'), binA0, binB0, binC0, binD0, binA1, binB1, binC1, binD1)
    newimage = image - tmpimage0
    cntww,cntbb = [(newimage==cnt).sum() for cnt in (0,1)]
    tend = time.time()
    trun = round((tend - tstart) * 1000)
    print("cntbb : {:8d} : cntww : {:8d} : trun : {:8d}".format(cntbb, cntww,trun))
    return newimage

# filter noise : any 1 noise along edge in window 3x3 : value = edge + opposite middles
def filter_bin_3(pixel, binA, binB, binC, binD):
    if pixel in [binA, binB, binC, binD]:
        return 1
    else:
        return 0
func_filter_bin_3 = np.vectorize(filter_bin_3)
def img_filter_one_fillA(image):
    tstart = time.time()
    onesbin = np.array([2**z for z in range(9)]).reshape((3,3))
    binA = sum(onesbin[0:3,0]) + onesbin[0,1] + onesbin[2,1]
    binB = sum(onesbin[0:3,2]) + onesbin[0,1] + onesbin[2,1]
    binC = sum(onesbin[0,0:3]) + onesbin[1,0] + onesbin[1,2]
    binD = sum(onesbin[2,0:3]) + onesbin[1,0] + onesbin[1,2]
    tmpimage0 = func_filter_bin_3(signal.convolve2d(image, onesbin, mode='same'), binA, binB, binC, binD)
    newimage = image + tmpimage0
    cntww,cntbb = [(newimage==cnt).sum() for cnt in (0,1)]
    tend = time.time()
    trun = round((tend - tstart) * 1000)
    print("cntbb : {:8d} : cntww : {:8d} : trun : {:8d}".format(cntbb, cntww,trun))
    return newimage

# filter noise : any 1 noise along edge in window 3x3 : value = sum - middle edge + corner
def filter_bin_4(pixel, binA0, binB0, binC0, binD0, binA1, binB1, binC1, binD1):
    if pixel in [binA0, binB0, binC0, binD0, binA1, binB1, binC1, binD1]:
        return 1
    else:
        return 0
func_filter_bin_4 = np.vectorize(filter_bin_4)
def img_filter_one_fillB(image):
    tstart = time.time()
    onesbin = np.array([2**z for z in range(9)]).reshape((3,3))
    onessum = sum(sum(onesbin)) - onesbin[1,1]
    bin_x = [[0,0],[2,2],[1,2],[1,2],[0,0],[2,2],[1,0],[1,0]]
    bin_y = [[1,2],[1,2],[0,0],[2,2],[1,0],[1,0],[0,0],[2,2]]
    bin_xy = zip(bin_x,bin_y)
    bin_list = []
    for x,y in bin_xy:
        bin_list.append(onessum - onesbin[x[0],y[0]] - onesbin[x[1],y[1]])
    binA0, binB0, binC0, binD0, binA1, binB1, binC1, binD1 = bin_list
    tmpimage0 = signal.convolve2d(image, onesbin, mode='same')
    tmpimage1 = func_filter_bin_4(tmpimage0, binA0, binB0, binC0, binD0, binA1, binB1, binC1, binD1)
    newimage = image + tmpimage1
    cntww,cntbb = [(newimage==cnt).sum() for cnt in (0,1)]
    tend = time.time()
    trun = round((tend - tstart) * 1000)
    print("cntbb : {:8d} : cntww : {:8d} : trun : {:8d}".format(cntbb, cntww,trun))
    return newimage

# filter noise : any 1 or 2 noise in window 3x3 : value = sum - middle - 0 or 1
def filter_bin_5(pixel, binA0, binB0):
    if pixel in [binA0, binB0]:
        return 1
    else:
        return 0
func_filter_bin_5 = np.vectorize(filter_bin_5)
def img_filter_one_fillC(image):
    tstart = time.time()
    onesbin = np.ones((3,3), dtype=int)
    onesbin[1,1] = 10
    onessum = sum(sum(onesbin)) - onesbin[1,1]
    binA0 = onessum - 0
    binB0 = onessum - 1
    tmpimage0 = signal.convolve2d(image, onesbin, mode='same')
    tmpimage1 = func_filter_bin_5(tmpimage0, binA0, binB0)
    newimage = image + tmpimage1
    cntww,cntbb = [(newimage==cnt).sum() for cnt in (0,1)]
    tend = time.time()
    trun = round((tend - tstart) * 1000)
    print("cntbb : {:8d} : cntww : {:8d} : trun : {:8d}".format(cntbb, cntww,trun))
    return newimage

# filter noise : any 2 noise in window 1x6 or 6x1 : value = sum - middle[2&3]
def filter_bin_7(pixel, binA0):
    if pixel in [binA0]:
        return 1
    else:
        return 0
func_filter_bin_7 = np.vectorize(filter_bin_7)
def filter_bin_8(pixel):
    if pixel != 0:
        return 1
    else:
        return 0
func_filter_bin_8 = np.vectorize(filter_bin_8)
def img_filter_one_fillD(image):
    tstart = time.time()
    onesbinv = np.ones(( 6, 1), dtype=int)
    onesbinv[2:4,0] = 10
    onessumv = sum(sum(onesbinv)) - sum(onesbinv[2:4,0])
    tmpimagev0 = signal.convolve2d(image, onesbinv, mode='same')
    tmpimagev1 = func_filter_bin_7(tmpimagev0, onessumv)
    onesbinh = np.ones(( 1, 6), dtype=int)
    onesbinh[0,2:4] = 10
    onessumh = sum(sum(onesbinh)) - sum(onesbinh[0,2:4])
    tmpimageh0 = signal.convolve2d(image, onesbinh, mode='same')
    tmpimageh1 = func_filter_bin_7(tmpimageh0, onessumh)
    newimage = func_filter_bin_8(image + tmpimagev1 + tmpimageh1)
    cntww,cntbb = [(newimage==cnt).sum() for cnt in (0,1)]
    tend = time.time()
    trun = round((tend - tstart) * 1000)
    print("cntbb : {:8d} : cntww : {:8d} : trun : {:8d}".format(cntbb, cntww,trun))
    return newimage

# filter noise : any 1 noise in window 1x3 or 3x1 : value = sum - middle
def filter_bin_9(pixel, binA0):
    if pixel in [binA0]:
        return 1
    else:
        return 0
func_filter_bin_9 = np.vectorize(filter_bin_9)
def filter_bin_10(pixel):
    if pixel != 0:
        return 1
    else:
        return 0
func_filter_bin_10 = np.vectorize(filter_bin_10)
def img_filter_one_fillE(image):
    tstart = time.time()
    onesbinv = np.ones(( 3, 1), dtype=int)
    onesbinv[1,0] = 10
    onessumv = sum(sum(onesbinv)) - onesbinv[1,0]
    tmpimagev0 = signal.convolve2d(image, onesbinv, mode='same')
    tmpimagev1 = func_filter_bin_9(tmpimagev0, onessumv)
    onesbinh = np.ones(( 1, 3), dtype=int)
    onesbinh[0,1] = 10
    onessumh = sum(sum(onesbinh)) - onesbinh[0,1]
    tmpimageh0 = signal.convolve2d(image, onesbinh, mode='same')
    tmpimageh1 = func_filter_bin_9(tmpimageh0, onessumh)
    newimage = func_filter_bin_10(image + tmpimagev1 + tmpimageh1)
    cntww,cntbb = [(newimage==cnt).sum() for cnt in (0,1)]
    tend = time.time()
    trun = round((tend - tstart) * 1000)
    print("cntbb : {:8d} : cntww : {:8d} : trun : {:8d}".format(cntbb, cntww,trun))
    return newimage

# filter noise : any 2 pixel along edge in window 5x5 : value = center + edge + center adjacent along edge
def filter_bin_6(pixel, binA0, binB0, binC0, binD0, binA1, binB1, binC1, binD1):
    if pixel in [binA0, binB0, binC0, binD0, binA1, binB1, binC1, binD1]:
        return 1
    else:
        return 0
func_filter_bin_6 = np.vectorize(filter_bin_6)
def img_filter_two_edgeA(image):
    tstart = time.time()
    onesbin = np.array([2**z for z in range(25)]).reshape((5,5))
    onessum = sum(sum(onesbin))
    binA0 = onesbin[2,2] + sum(sum(onesbin[0:5,0:2])) + onesbin[1,2]
    binB0 = onesbin[2,2] + sum(sum(onesbin[0:5,3:5])) + onesbin[1,2]
    binC0 = onesbin[2,2] + sum(sum(onesbin[0:2,0:5])) + onesbin[2,1]
    binD0 = onesbin[2,2] + sum(sum(onesbin[3:5,0:5])) + onesbin[2,1]
    binA1 = onesbin[2,2] + sum(sum(onesbin[0:5,0:2])) + onesbin[3,2]
    binB1 = onesbin[2,2] + sum(sum(onesbin[0:5,3:5])) + onesbin[3,2]
    binC1 = onesbin[2,2] + sum(sum(onesbin[0:2,0:5])) + onesbin[2,3]
    binD1 = onesbin[2,2] + sum(sum(onesbin[3:5,0:5])) + onesbin[2,3]
    tmpimage0 = func_filter_bin_6(signal.convolve2d(image, onesbin, mode='same'), binA0, binB0, binC0, binD0, binA1, binB1, binC1, binD1)
    newimage = image - tmpimage0
    cntww,cntbb = [(newimage==cnt).sum() for cnt in (0,1)]
    tend = time.time()
    trun = round((tend - tstart) * 1000)
    print("cntbb : {:8d} : cntww : {:8d} : trun : {:8d}".format(cntbb, cntww,trun))
    return newimage

# check ABL * ABL squares
# if only 2 entry points, then draw line
def img_filter_line_edge_side(result,z,AB):
    flag,aa,bb,da,db = (0,0,AB,0,0)
    if result.size > 0:
        aa,bb,cc = (result[0]+z,result[-1]+z, result[-1] - result[0] +1)
#       aa, bb = (aa-0, bb+1) if ((bb-aa) % 2) == 1 else (aa-0,bb+0)
#       dba = max(int((bb-aa) / 2 - 2), 0)
#       aa,bb  = (aa+dba,bb-dba)
        aa, bb = (aa-1, bb+1) if  (bb-aa)      <= 2 else (aa-0,bb+0)
        flag = 2 if aa >  z+0 and bb <  z+AB and result.size == cc else 0
        flag = 1 if aa == z+0 or  bb == z+AB                       else flag
        da = result[0]
        db = AB - result[-1]
    return flag, aa, bb, da, db
def img_filter_box_edge_size(result, AB):
    if result.size > 0:
        return result[0], AB - result[-1]
    else :
        return AB, AB
def img_filter_box_edge(image, AB):
    X,Y = image.shape
    qbr,    qtr,    qtl,    qbl    = np.zeros((4), dtype=int)
    flagbr, flagtr, flagtl, flagbl = np.zeros((4), dtype=int)
    sumbr,  sumtr,  sumtl,  sumbl  = np.zeros((4), dtype=int)
    for x in range(X):
        result = np.nonzero(image[x,0:Y])[0]
        ya, yb = img_filter_box_edge_size(result, AB)
        flagbr = 1  if flagbr == 0 and yb == 0 else flagbr
        flagtr = 1  if flagtr == 0 and yb == 0 else flagtr
        flagtl = 1  if flagtl == 0 and ya == 0 else flagtl
        flagbl = 1  if flagbl == 0 and ya == 0 else flagbl
        sumbr += yb if flagbr == 1 and yb >  0 else 0
        sumtr += yb if flagtr == 0 and yb >  0 else 0
        sumtl += ya if flagtl == 0 and ya >  0 else 0
        sumbl += ya if flagbl == 1 and ya >  0 else 0
    return sumbr, sumtr, sumtl, sumbl
def img_filter_line_edgeA(image, mask, maskvalue, start, AB=10, debug = 0, method = ['4S', 'HH', 'VV', 'TL', 'BL', 'TR', 'BR']):
#   image = cp.deepcopy(image)
    tstart = time.time()
    AB2 = int(AB/2)
    cntcc,cntrr, cnt4s,cnthh,cntvv,cnttl,cnttr,cntbl,cntbr = np.zeros((9), dtype=int)
    X,Y = image.shape
    for x in range(start, X - AB2, AB):
        for y in range(start, Y - AB2, AB):
            # already processed or all zero:
            resultrr = ((mask[x-AB2:x+AB2+1,y-AB2:y+AB2+1]) == maskvalue).sum()
            if resultrr == (AB + 1) ** 2:
                cntrr += 1
                continue
            # inner loop init
            vertices_list = []
            flag = 0
            bl,br,at,ab = (y-AB2,y+AB2,x-AB2,x+AB2)
            # left
            resultl = np.nonzero(image[x - AB2:x + AB2+1,bl])[0]
            flagl,xla,xlb,dxla,dxlb = img_filter_line_edge_side(resultl,x-AB2,AB)
            # right
            resultr = np.nonzero(image[x - AB2:x + AB2+1,br])[0]
            flagr,xra,xrb,dxra,dxrb = img_filter_line_edge_side(resultr,x-AB2,AB)
            # top
            resultt = np.nonzero(image[at,y - AB2:y + AB2+1])[0]
            flagt,yta,ytb,dyta,dytb = img_filter_line_edge_side(resultt,y-AB2,AB)
            # bottom
            resultb = np.nonzero(image[ab,y - AB2:y + AB2+1])[0]
            flagb,yba,ybb,dyba,dybb = img_filter_line_edge_side(resultb,y-AB2,AB)
            # horizontal & vertical
            if flagl == 2 and flagr == 2 and flagt == 2 and flagb == 2 and '4S' in method:
                cnt4s += 1
                flag  =  1
                vertices_list.append(np.array([[xla,bl], [xlb,bl], [xrb,br], [xra,br]]))
                vertices_list.append(np.array([[at,yta], [at,ytb], [ab,ybb], [ab,yba]]))
            # horizontal
            if flagl == 2 and flagr == 2 and flagt == 0 and flagb == 0 and 'HH' in method:
                cnthh += 1
                flag  =  1
                vertices_list.append(np.array([[xla,bl], [xlb,bl], [xrb,br], [xra,br]]))
            # vertical
            if flagt == 2 and flagb == 2 and flagl == 0 and flagr == 0 and 'VV' in method:
                cntvv += 1
                flag  =  1
                vertices_list.append(np.array([[at,yta], [at,ytb], [ab,ybb], [ab,yba]]))
            # top and left
            if flagt == 2 and flagl == 2 and flagb == 0 and flagr == 0 and 'TL' in method:
                cnttl += 1
                flag  =  2
                sumbr,sumtr,sumtl,sumbl=img_filter_box_edge(image[x-AB2:x+AB2+1,y-AB2:y+AB2+1], AB)
                vertices_list.append(np.array([[at,yta], [at,ytb], [xla,bl], [xlb,bl]]))
            # bottom and left
            if flagb == 2 and flagl == 2 and flagt == 0 and flagr == 0 and 'BL' in method:
                cntbl += 1
                flag  =  2
                sumbr,sumtr,sumtl,sumbl=img_filter_box_edge(image[x-AB2:x+AB2+1,y-AB2:y+AB2+1], AB)
                vertices_list.append(np.array([[ab,yba], [ab,ybb], [xla,bl], [xlb,bl]]))
            # top and right
            if flagt == 2 and flagr == 2 and flagb == 0 and flagl == 0 and 'TR' in method:
                cnttr += 1
                flag  =  2
                sumbr,sumtr,sumtl,sumbl=img_filter_box_edge(image[x-AB2:x+AB2+1,y-AB2:y+AB2+1], AB)
                vertices_list.append(np.array([[at,yta], [at,ytb], [xra,br], [xrb,br]]))
            # bottom and right
            if flagb == 2 and flagr == 2 and flagt == 0 and flagl == 0 and 'BR' in method:
                cntbr += 1
                flag  =  2
                sumbr,sumtr,sumtl,sumbl=img_filter_box_edge(image[x-AB2:x+AB2+1,y-AB2:y+AB2+1], AB)
                vertices_list.append(np.array([[ab,yba], [ab,ybb], [xra,br], [xrb,br]]))
            # touching corner
            if flagb == 1 or  flagr == 1 or  flagt == 1 or  flagl == 1:
                cntcc += 1
            # valid intersection
            if flag > 0:
                image[x-AB2:x+AB2+1,y-AB2:y+AB2+1] = 0
                mask[x-AB2:x+AB2+1,y-AB2:y+AB2+1] = 1
                for vertices in vertices_list:
                    img_create_polygon(image, AB, x - AB2, y - AB2, vertices, debug=debug)
            if debug >= 3 and flag > 0:
                print(resultl)
                print(resultr)
                print(resultt)
                print(resultb)
                print(vertices)
    cntww,cntbb = [(image==cnt).sum() for cnt in (0,1)]
    tend = time.time()
    trun = round((tend - tstart) * 1000)
    print("cntbb : {:8d} : cntww : {:8d} : cntrr : {:8d} : trun  : {:8d}".format(cntbb, cntww, cntrr, trun))
    print("cntvv : {:8d} : cnthh : {:8d} : cnt4s : {:8d} : cntcc : {:8d}".format(cntvv, cnthh, cnt4s, cntcc))
    print("cnttl : {:8d} : cntbl : {:8d} : cnttr : {:8d} : cntbr : {:8d}".format(cnttl, cntbl, cnttr, cntbr))
    return

