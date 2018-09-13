# -*- coding:utf-8 -*-
"""


"""
import cv2
from difflib import SequenceMatcher
import logging
import os
import sys
import numpy as np
from copy import deepcopy
from imutils import paths
import imutils
import itertools
import time
import HoonUtils as hp
# from matplotlib import pyplot as plt
# from PIL import Image
# from enchant import Dict

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

RED     = (255,   0,   0)
GREEN   = (  0, 255,   0)
BLUE    = (  0,   0, 255)
CYAN    = (  0, 255, 255)
MAGENTA = (255,   0, 255)
YELLOW  = (255, 255,   0)
WHITE   = (255, 255, 255)
BLACK   = (  0,   0,   0)

IMG_EXTS = ['jpg', 'png', 'bmp', 'gif', 'tif', 'tiff']

if False:
    try:
        if sys.platform == 'darwin':
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.quit()
        else:
            from screeninfo import get_monitors
            screen_width  = get_monitors()[0].width
            screen_height = get_monitors()[0].height
        screen_width = 1920 if screen_width > 1920 else screen_width
    except Exception as e:
        print(e)
        pass
else:
    screen_width  = 1920
    screen_height = 1080



class ContourInfo:

    def __init__(self):
        self.contours = []
        self.areas = []
        self.rect_areas = []
        self.rect_area_dims = []
        self.cxs = []
        self.cys = []

    # ------------------------------------------------------------------------------------------------------------------
    def calc_all(self):
        self.areas = []
        self.rect_areas = []
        self.rect_area_dims = []
        self.cxs = []
        self.cys = []
        for k in range(len(self.contours)):
            self.areas.append(int(cv2.contourArea(self.contours[k])))
            _, _, w, h = cv2.boundingRect(self.contours[k])
            self.rect_area_dims.append([w, h])
            self.rect_areas.append(int(w * h))
            M = cv2.moments(self.contours[k])
            self.cxs.append(int(M['m10']/M['m00']) if M['m00'] != 0 else -1)
            self.cys.append(int(M['m01']/M['m00']) if M['m00'] != 0 else -1)

    # ------------------------------------------------------------------------------------------------------------------
    def contour_sorting_by_area(self, reverse=False):
        s_idx = [i[0] for i in sorted(enumerate(self.areas), key=lambda x:x[1], reverse=reverse)]
        contours = deepcopy(self.contours)
        areas = deepcopy(self.areas)
        rect_areas = deepcopy(self.rect_areas)
        cxs = deepcopy(self.cxs)
        cys = deepcopy(self.cys)
        for k in range(len(self.areas)):
            self.contours[k] = contours[s_idx[k]]
            self.areas[k] = areas[s_idx[k]]
            self.rect_areas[k] = rect_areas[s_idx[k]]
            self.cxs[k] = cxs[s_idx[k]]
            self.cys[k] = cys[s_idx[k]]

    # ------------------------------------------------------------------------------------------------------------------
    def delete_contour_info(self, idx):
        del self.contours[idx]
        del self.areas[idx]
        del self.rect_areas[idx]
        del self.rect_area_dims[idx]
        del self.cxs[idx]
        del self.cys[idx]

    # ------------------------------------------------------------------------------------------------------------------
    def check_inclusion(self, canvas, idx_small, idx_big, disp=False):
        dim = canvas.shape
        canvas = np.zeros((dim[0], dim[1], 3), dtype=np.uint8)
        cv2.drawContours(canvas, self.contours, idx_big, WHITE, -1)
        if disp:
            canvas2 = np.zeros((dim[0]+4, dim[1]+4, 3), dtype=np.uint8)
            cv2.drawContours(canvas2, self.contours, idx_big, WHITE, 1)
            cv2.drawContours(canvas2, self.contours, idx_small, RED, 1)
            for k in range(len(self.contours[idx_small])):
                pos = self.contours[idx_small][k][0]
                canvas2[pos[1]][pos[0]] = BLUE
            hp.hp_imshow(canvas2)

        for k in range(len(self.contours[idx_small])):
            pos = self.contours[idx_small][k][0]
            if sum(canvas[pos[1],pos[0]]) != 0:
                return True

        return False

    # ------------------------------------------------------------------------------------------------------------------
    def delete_contour_from_inclusion_rule(self, img, disp=False):
        i = 0
        while i < len(self.contours) - 1:
            if self.check_inclusion(img, i+1,i, disp=disp):
                self.delete_contour_info(i)
            else:
                i += 1
        pass

    # ------------------------------------------------------------------------------------------------------------------
    def delete_contour_from_area_rule(self, thresh=10):
        i = 0
        while i < len(self.contours):
            if self.areas[i] < thresh:
                self.delete_contour_info(i)
            else:
                i += 1

    # ------------------------------------------------------------------------------------------------------------------
    def delete_contour_from_shape_rule(self, mult=5):
        i = 0
        while i < len(self.contours):
            if self.areas[i]  * mult < self.rect_areas[i]:
                self.delete_contour_info(i)
            else:
                i += 1

    # ------------------------------------------------------------------------------------------------------------------
    def delete_contour_from_value_rule(self, img, thresh=128, disp=False):
        i = 0
        while i < len(self.contours):
            dim = img.shape
            canvas = np.zeros((dim[0], dim[1]), dtype=np.uint8)
            cv2.drawContours(canvas, self.contours, i, 255, -1)
            val = 0
            cnt = 0
            if disp:
                hp.hp_imshow(canvas)
            for ky in range(dim[0]):
                for kx in range(dim[1]):
                    if canvas[ky,kx] == 255:
                        val += img[ky,kx]
                        cnt += 1
            if (val / float(cnt)) < thresh:
                self.delete_contour_info(i)
            else:
                i += 1


########################################################################################################################
def get_full_zoom_factor(img, zoom=0.0, factor=4/5.):

    global screen_width, screen_height

    if isinstance(img, str):
        img = hp.hp_imread(img)

    dim = img.shape
    h, w = dim[0], dim[1]
    zoom_w = screen_width  / float(w) * factor
    zoom_h = screen_height / float(h) * factor

    if zoom == 0:
        zoom = min(zoom_h, zoom_w)
    elif zoom < 0:
        if (screen_width < w) or (screen_height < h):
            zoom = min(zoom_h, zoom_w)
        else:
            zoom = 1
    else:
        zoom = 1

    return zoom


def destroyWindow_safe(desc):
    cv2.waitKey(1)
    cv2.destroyWindow(desc)
    for k in range(10):
        cv2.waitKey(1)


########################################################################################################################
def imresize_full(img):

    global screen_width, screen_height

    dim = img.shape
    h, w = dim[0], dim[1]
    if h <= 0 or w <= 0:
        zoom_w, zoom_h = 1, 1
    else:
        zoom_w = screen_width  / float(w) * 9/10.
        zoom_h = screen_height / float(h) * 9/10.
    zoom = min(zoom_h, zoom_w)
    return cv2.resize(img, (0,0), fx=zoom, fy=zoom), zoom


def get_video_stream_info(video_stream):
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, frame_num


def read_frame(video_stream):

    flag, img_full = video_stream.read()
    if not flag:
        print(" @ Error: cannot read input video stream")
        return None
    return img_full


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def my_pause():
    print("")
    input("Press Enter to continue...")


class LocalBreak(Exception):

    def __init(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def find_common_area(px1, py1, px2, py2, qx1, qy1, qx2, qy2):

    area1 = (px2 - px1) * (py2 - py1)
    area2 = (qx2 - qx1) * (qy2 - qy1)

    if px2 < qx1 or qx2 < px1 or py2 < qy1 or qy2 < py1:
        vertex_pts = (0,0,0,0)
    else:
        vertex_pts = (max(px1, qx1), max(py1, qy1), min(px2, qx2), min(py2, qy2))
    '''
    elif px1 < qx1 < px2 < qx2 and py1 < qy1 < py2 < qy2: vertex_pts = (qx1, qy1, px2, py2)
    elif px1 < qx1 < px2 < qx2 and qy1 < py1 < qy2 < py2: vertex_pts = (qx1, py1, px2, qy2)
    elif qx1 < px1 < qx2 < px2 and py1 < qy1 < py2 < qy2: vertex_pts = (px1, qy1, qx2, py2)
    elif qx1 < px1 < qx2 < px2 and qy1 < py1 < qy2 < py2: vertex_pts = (px1, py1, qx2, qy2)

    elif px1 < qx1 < qx2 < px2 and py1 < qy1 < py2 < qy2: vertex_pts = (qx1, qy1, qx2, py2)
    elif qx1 < px1 < px2 < qx2 and qy1 < py1 < qy2 < py2: vertex_pts = (px1, py1, px2, qy2)
    elif qx1 < px1 < qx2 < px2 and py1 < qy1 < qy2 < py2: vertex_pts = (px1, qy1, qx2, qy2)
    elif px1 < qx1 < px2 < qx2 and qy1 < py1 < py2 < qy2: vertex_pts = (qx1, py1, px2, py2)
    elif px1 < qx1 < qx2 < px2 and qy1 < py1 < qy2 < py2: vertex_pts = (qx1, py1, qx2, qy2)
    elif qx1 < px1 < px2 < qx2 and py1 < qy1 < py2 < qy2: vertex_pts = (px1, qy1, px2, py2)
    elif px1 < qx1 < px2 < qx2 and py1 < qy1 < qy2 < py2: vertex_pts = (qx1, qy1, px2, qy2)
    elif qx1 < px1 < qx2 < px2 and qy1 < py1 < py2 < qy2: vertex_pts = (px1, py1, qx2, py2)

    elif px1 < qx1 < qx2 < px2 and py1 < qy1 < qy2 < py2: vertex_pts = (qx1, qy1, qx2, qy2)
    elif qx1 < px1 < px2 < qx2 and qy1 < py1 < py2 < qy2: vertex_pts = (px1, py1, px2, py2)
    '''
    area3 = (vertex_pts[1] - vertex_pts[0]) * (vertex_pts[3] - vertex_pts[2])
    ratio1 = 0. if area3 == 0 else area3 / float(area1)
    ratio2 = 0. if area3 == 0 else area3 / float(area2)

    return vertex_pts, ratio1, ratio2


def configure_logger(filename, name, level=logging.INFO):

    logging.basicConfig(level=level)
    logger = logging.getLogger(name)
    log_handler = logging.FileHandler(filename)
    log_handler.setLevel(level)
    log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(log_handler)

    return logger


def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
            return i
    return -1


def add_prefix_to_all_files(folder, prefix):
    filenames = os.listdir(folder)
    for filename  in filenames:
        if filename.split('.')[0] != "yt":
            os.rename(folder + "/" + filename, folder + "/" + prefix + filename)


def to_str(string_dat):
    if isinstance(string_dat, str):
        string_dat = string_dat.encode('utf-8')
    return string_dat


def to_unicode(unicode_dat):
    if isinstance(unicode_dat, str):
        unicode_dat = unicode_dat.decode('utf-8')
    return unicode_dat


def check_image_file(filename):
    ext = filename.split('.')[-1]
    if ext in ['jpg', 'png', 'bmp', 'gif', 'tiff']:
        return True
    else:
        return False


def filter_image_file_from_list(img_list):
    out_img_list = []
    for idx in range(len(img_list)):
        if check_image_file(img_list[idx]):
            out_img_list.append(img_list[idx])
    return out_img_list


def imwrite_safe(img_file, img, color_fmt='RGB'):

    if color_fmt.upper() == 'RGB':
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img

    cv2.imwrite(img_file, img_rgb)


# ==============================================================================================================================
def overlay_boxes_on_image(img, boxes, color, desc="", display=True):

    if boxes.ndim == 1:
        cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[0] + boxes[2], boxes[1] + boxes[3]), color, 2)
    else:
        for (x, y, w, h) in boxes:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    if display:
        hp.hp_imshow(img, desc=desc, zoom=0)

    return img


# ==============================================================================================================================
def image_thresholding(img, blur=5, method='BINARY'):

    img = hp.hp_imread(img)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if blur is not 0:
        img = cv2.medianBlur(img, blur)

    if method == 'BINARY':
        ret, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    elif method == 'BINARY_INV':
        ret, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    elif method == 'TRUNC':
        ret, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    elif method == 'TOZERO':
        ret, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    elif method == 'TOZERO_INV':
        ret, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    elif method == 'ADAPTIVE_MEAN':
        thresh_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    elif method == 'ADAPTIVE_GAUSSIAN':
        thresh_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif method == 'OTZU':
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        ret3, thresh_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thresh_img = img
        pass

    return thresh_img


def smooth(x,window_len=11,window='hanning'):
    """
    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t = linspace(-2,2,0.1)
    x = sin(t) + randn(len(t))*0.1
    y = smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

    if window == 'flat': # moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')

    return y


def modify_rect_size(rectangle, x, y):
    rect = deepcopy(rectangle)
    rect[0] = 0     if rect[0] <  0 else rect[0]
    rect[0] = x - 1 if rect[0] >= x else rect[0]
    rect[1] = 0     if rect[1] < 0  else rect[1]
    rect[1] = y - 1 if rect[1] >= y else rect[1]
    rect[2] = 0     if rect[2] <  0 else rect[2]
    rect[2] = x - 1 if rect[2] >= x else rect[2]
    rect[3] = y - 1 if rect[3] >= y else rect[3]
    rect[3] = y - 1 if rect[3] >= y else rect[3]
    return rect

'''
    if False:   # Binary image test
        methods = ['BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV', 'ADAPTIVE_MEAN', 'ADAPTIVE_GAUSSIAN',
                   'OTZU', 'NONE']
        for method in methods:
            ref_img = me.image_thresholding(arg.ref_img_file, blur=0, method=method)
            hp.hp_imshow(ref_img, desc=method, zoom=1.0)
'''

# ----------------------------------------------------------------------------------------------------------------------
def convert_pdf_to_img(pdf_filename, img_type, img_filename, resolution=300):
    if os.name == 'nt':
        print(" @ Error: Wand library does not work in Windows OS\n")
        sys.exit()
    else:
        from wand.image import Image
        with Image(filename=pdf_filename, resolution=resolution) as img:
            img.compression = 'no'
            with img.convert(img_type) as converted:
                converted.save(filename=img_filename)

# ----------------------------------------------------------------------------------------------------------------------
def read_pdf(pdf_filename, resolution=300):
    img_filename = 'temp.bmp'
    convert_pdf_to_img(pdf_filename, 'bmp', img_filename, resolution=resolution)
    img = hp.hp_imread(img_filename, color_fmt='RGB')
    os.remove(img_filename)
    return img


def add_box_overlay(img, box, color, alpha):
    """
    Add overlay box to image.

    :param img:
    :param box:
    :param color:
    :param alpha:
    :return:
    """
    over = cv2.rectangle(img.copy(), tuple(box[0:2]), tuple(box[2:]), color, -1)
    over = cv2.addWeighted(img.copy(), alpha, over, 1 - alpha, 0)
    return over

"""
def convert_to_korean_syllables(string):
    if isinstance(string, str):
        utf_str = unicode(string, 'utf-8')
"""


def get_file_list(dir_name, ext='.pdf', sorting_=True):

    if os.path.isdir(dir_name):
        file_list = list(paths.list_files(dir_name, validExts=ext, contains=None))
        if sorting_:
            file_list.sort()
    elif os.path.isfile(dir_name):
        file_list = [dir_name]

    return file_list


def get_color_histogram(img_file, color_fmt='RGB'):

    img = hp.hp_imread(img_file)

    if color_fmt.lower() == 'gray':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        plt.show(hist, "Grayscale histogram", "Bins", "# of pixels", [0, 256], None)
    elif color_fmt.lower() == 'rgb':
        channels = cv2.split(img)
        colors = ('b', 'g', 'r')
        features = []
        plt.figure()
        plt.title("Flattened color histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of pixels")

        for (channel, color) in zip(channels, colors):
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            features.extend(hist)

            plt.plot(hist, color=color)
            plt.xlim([0, 256])

        print("flattened feature vector size : %d" % (np.array(features).flatten().shape))
        plt.show()


def plt_imshow(data_2d, title=None, x_label=None, y_label=None, x_range=None, y_range=None, xticks=[], yticks=[],
               maximize_=True, block=True):
    """Show image via matplotlib.pyplot.

    :param data_2d:
    :param title:
    :param x_label:
    :param y_label:
    :param x_range:
    :param y_range:
    :param xticks:
    :param yticks:
    :param maximize_:
    :param block:
    :return:
    """
    if maximize_:
        if os.name == "nt":     # If Windows OS.
            plt.get_current_fig_manager().window.state('zoomed')
        else:
            plt.get_current_fig_manager().window.showMaximized()

    dim = data_2d.shape
    if len(dim) is 2:
        plt.imshow(data_2d, cmap='gray')
    elif len(dim) is 3:
        if dim[2] is 1:
            plt.imshow(data_2d, cmap='gray')
        else:
            plt.imshow(data_2d)

    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)
    plt.xticks(xticks), plt.yticks(yticks)
    plt.show(block=True)


def plt_imshow2(img1, img2, horizontal_=True, title=(None, None), maximize_=True, block=True):

    fig = plt.figure()

    fig1 = fig.add_subplot(1,2,1) if horizontal_ else fig.add_subplot(2,1,1)
    plt.imshow(img1)
    if title[0]:
        plt.title(title)
    plt.xticks([]), plt.yticks([])

    fig2 = fig.add_subplot(1,2,2) if horizontal_ else fig.add_subplot(2,1,2)
    plt.imshow(img2)
    if title[1]:
        plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.get_current_fig_manager().window.showMaximized()

    if block == 'True':
        plt.show(block=True)
    elif isinstance(block, int):
        plt.show(block=True)
        time.sleep(block)
        plt.close()

# ------------------------------------------------------------------------------------------------------------------------------
def compare_threshold_algorithms(img_gray):

    bw_imgs = [img_gray]
    bw_imgs.append(cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
    bw_imgs.append(cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2))
    bw_imgs.append(cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))

    titles = ['Original image',
              'Global thresholding (v = 127)',
              'Adaptive mean thresholding',
              'Adaptive gaussian thresholding']

    for idx in range(len(bw_imgs)):
        plt.subplot(2, 2, idx + 1), plt.imshow(bw_imgs[idx], 'gray')
        plt.title(titles[idx])
        plt.xticks([]), plt.yticks([])

    plt.show()
    pass


def check_lines_in_img(img_bw):

    img_edge = cv2.Canny(img_bw, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(img_edge, 1, np.pi/180, 100)
    dim = img_edge.shape

    for line in lines:
        img_line = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2RGB)
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x = []
        y = []
        if b is not 0:
            slope = -a / b
            y1 = slope * (-x0) + y0
            if 0 <= y1 < dim[0]:
                x.append(0)
                y.append(y1)
            y1 = slope * (dim[1] - 1 - x0) + y0
            if 0 <= y1 < dim[0]:
                x.append(dim[1] - 1)
                y.append(y1)
            x1 = (-y0) / slope + x0
            if 0 <= x1 < dim[1]:
                x.append(x1)
                y.append(0)
            x1 = (dim[0] -1 - y0) / slope + x0
            if 0 <= x1 < dim[1]:
                x.append(x1)
                y.append(dim[0] - 1)
        else:
            x = [x0, x0]
            y = [0, dim[0]-1]
        angle = (90 - (theta * 180 / np.pi))
        print(" # rotated angle = {:.1f} <- ({:f}, {:f}".format(angle, theta, rho))
        if len(x) is 2:
            cv2.line(img_line, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), RED, 32)
            plt_imshow(img_line)
            # if -5 < angle < 0 or 0 < angle < 5:
            #     plt_imshow(img_line)
        else:
            print(" @ Warning: something wrong.\n")
            pass

    img_edge = cv2.Canny(img_bw, 50, 150, apertureSize=3)
    end_pts_list = cv2.HoughLinesP(img_edge, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)

    for end_pts in end_pts_list:
        img_line = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2RGB)
        cv2.line(img_line, tuple(end_pts[0][0:2]), tuple(end_pts[0][2:]), RED, 10)
        angle = np.arctan2(end_pts[0][3]-end_pts[0][1], end_pts[0][2]-end_pts[0][0]) * 180. / np.pi
        print(" # rotated angle = {:.1f}".format(angle))
        plt_imshow(img_line)
        # if -5 < angle < 0 or 0 < angle < 5:
        #     plt_imshow(img_line)

    plt.imshow(img_line)
    plt.title("Check lines in image")
    plt.xticks([]), plt.yticks([])
    plt.show()
    pass


def derotate_img(img, compare_imgs=0, line_img_file=None, rot_img_file=None, check_lines_in_img_=False, check_time_=False):
    """Derotate image.

    :param img:
    :param compare_imgs:
    :param line_img_file:
    :param rot_img_file:
    :param check_lines_in_img_:
    :param check_time_:
    :return:
    """
    if check_time_: start_time = time.time()
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.amin(img, axis=2)

    check_lines_in_img_ = True
    if check_lines_in_img_:    # Test & check purpose...
        # mu.compare_threshold_algorithms(img_gray)
        _, img_bw = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        check_lines_in_img(img_bw)
        return

    _, img_bw = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edges = cv2.Canny(img_bw, 50, 150, apertureSize=3)
    plt_imshow(edges)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    min_idx = lines[0][:,1].argmin()
    min_angle = (90 - lines[0][min_idx,1] * 180 / np.pi)
    min_angle = int(np.sign(min_angle) * (abs(min_angle) + 0.5))

    if (compare_imgs is not 0) or line_img_file:
        line_img = img.copy()
        rho, theta = lines[0][min_idx]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        dim = img_bw.shape
        x1 = int(x0 - x0 * b)
        y1 = int(y0 + x0 * a)
        x2 = int(x0 + (dim[1] - x0) * b)
        y2 = int(y0 - (dim[1] - x0) * a)
        cv2.line(line_img, (x1, y1), (x2, y2), RED, 10)
        if line_img_file:
            imwrite_safe(line_img_file, line_img, 'RGB')

    rot_img = imutils.rotate(img, -min_angle, (0,0), 1)
    if check_time_:
        print(" ! Time for rotation detection and doc de-rotation if any : {:.2f} sec".format(float(time.time() - start_time)))

    if compare_imgs is not 0:
        # print(" ! angle = {:d} <- {:f}".format(min_angle, theta*180/np.pi))
        print(" ! angle = {:d}".format(min_angle))
        # plt_imshow(np.concatenate((line_img, rot_img), axis=1), title="Comparison of original and de-rotated images")
        hp.hp_imshow(np.concatenate((line_img, rot_img), axis=1), pause_sec=compare_imgs)

    if rot_img_file:
        imwrite_safe(rot_img_file, rot_img, 'RGB')

    return rot_img


# ------------------------------------------------------------------------------------------------------------------------------

def compare_strings(string1, string2, no_match_c='*', match_c='|'):
    if len(string2) < len(string1):
        string1, string2 = string2, string1
    result = ''
    n_diff = 0
    for c1, c2 in itertools.izip(string1, string2):
        if c1 == c2:
            result += match_c
        else:
            result += no_match_c
            n_diff += 1
    delta = len(string2) - len(string1)
    result += delta * no_match_c
    n_diff += delta
    return (result, n_diff)


def template_matching(tar_full_img,
                      tmp_full_img,
                      tar_area=None,
                      tmp_area=None,
                      method=cv2.TM_CCOEFF,
                      pause=-1):
    """
    Template matching algorithm based on opencv.

    :param tar_full_img:
    :param tmp_full_img:
    :param tar_area:
    :param tmp_area:
    :param method:
    :param pause:
    :return:
    """

    # methods
    # cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED

    if tar_area:
        dim = tar_full_img.shape
        tar_area[0] = 0      if tar_area[0] < 0      else tar_area[0]
        tar_area[1] = 0      if tar_area[1] < 0      else tar_area[1]
        tar_area[2] = dim[1] if tar_area[2] > dim[1] else tar_area[2]
        tar_area[3] = dim[0] if tar_area[3] > dim[0] else tar_area[3]
        tar_img = tar_full_img[tar_area[1]:tar_area[3],tar_area[0]:tar_area[2]]
    else:
        tar_img = tar_full_img

    if tmp_area:
        dim = tmp_full_img.shape
        tmp_area[0] = 0      if tmp_area[0] < 0      else tmp_area[0]
        tmp_area[1] = 0      if tmp_area[1] < 0      else tmp_area[1]
        tmp_area[2] = dim[1] if tmp_area[2] > dim[1] else tmp_area[2]
        tmp_area[3] = dim[0] if tmp_area[3] > dim[0] else tmp_area[3]
        tmp_img = tmp_full_img[tmp_area[1]:tmp_area[3],tmp_area[0]:tmp_area[2]]
    else:
        tmp_img = tmp_full_img

    if False:
        dim_tar = tar_img.shape
        dim_tmp = tmp_img.shape
        disp_img = np.zeros((dim_tar[0] * 2, dim_tar[1]), dtype=np.uint8)
        disp_img[:dim_tar[0], :dim_tar[1]] = tar_img
        disp_img[dim_tar[0]+8:dim_tar[0]+dim_tmp[0]+8, 8:dim_tmp[1]+8] = tmp_img
        hp.hp_imshow(disp_img)

    res = cv2.matchTemplate(tar_img, tmp_img, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    if pause >= 0:
        bar_num = 16
        dim = tmp_img.shape
        bottom_right = (top_left[0] + dim[1], top_left[1] + dim[0])
        tar_box_img = cv2.rectangle(cv2.cvtColor(tar_img.copy(), cv2.COLOR_GRAY2RGB), top_left, bottom_right, RED, 4)
        dim_tmp = tmp_img.shape
        dim_tar = tar_box_img.shape
        merge_img = np.zeros((dim_tmp[0]+dim_tar[0]+bar_num, dim_tar[1], dim_tar[2]), dtype=np.uint8)
        merge_img[:dim_tmp[0]:,:dim_tmp[1],0] = tmp_img[:,:]
        merge_img[:dim_tmp[0]:,:dim_tmp[1],1] = tmp_img[:,:]
        merge_img[:dim_tmp[0]:,:dim_tmp[1],2] = tmp_img[:,:]
        merge_img[dim_tmp[0]+bar_num:,:,:] = tar_box_img[:,:,:]
        hp.hp_imshow(merge_img, desc='template matching', pause_sec=pause, loc=(10,10))

    # print(" # min_val = {:d} & max_val = {:d}".format(int(min_val), int(max_val)))

    top_left_rel = deepcopy(top_left)

    if tar_area:
        top_left = (top_left[0] + tar_area[0], top_left[1] + tar_area[1])

    return top_left, top_left_rel, min_val, max_val


# ==============================================================================================================================

def raw_input_safe(in_str):
    try:
        import termios
        time.sleep(1)
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    except:
        pass
    return input(in_str)




def find_box_from_outside(img, pxl_val=1, thresh_ratio=0.1, disp_=False):
    """
    Find rectangle box including something from the outside of BW image.

    :param img:
    :param pxl_val:
    :param thresh_ratio:
    :param disp_:
    :return box:
    """

    dim = img.shape
    box = [-1,] * 4
    thresh = min(dim[0], dim[1]) * thresh_ratio
    MARGIN = 2

    for idx in range(dim[1]):
        if sum(img[:,idx] < pxl_val) > thresh:
            box[0] = max(0, idx - MARGIN)
            break

    for idx in range(dim[0]):
        if sum(img[idx,:] < pxl_val) > thresh:
            box[1] = max(0, idx - MARGIN)
            break

    for idx in range(dim[1]-1, -1, -1):
        if sum(img[:,idx] < pxl_val) > thresh:
            box[2] = min(dim[1], idx)
            break

    for idx in range(dim[0]-1, -1, -1):
        if sum(img[idx,:] < pxl_val) > thresh:
            box[3] = min(dim[0], idx)
            break

    if disp_:
        if len(img.shape) == 2:
            disp_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            disp_img = img.copy()
        cv2.rectangle(disp_img, tuple(box[:2]), tuple(box[2:]), RED, 4)
        hp.hp_imshow(disp_img)

    return box

def crop_box_from_img(img, box, margin=0):
    return img[box[1]-margin:box[3]+margin, box[0]-margin:box[2]+margin]


def draw_box(img, box, color=RED, thickness=1):
    return cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), color=color, thickness=thickness)


def box_dim(box):
    return box[2] - box[0], box[3] - box[1]


def check_box_in_img_dim(box, dim):
    """

    :param box:
    :param dim:
    :return:
    """

    box[0] = 0          if box[0] <  0      else box[0]
    box[2] = dim[0] - 1 if box[2] >= dim[0] else box[2]
    box[1] = 0          if box[1] <  0      else box[1]
    box[3] = dim[1] - 1 if box[3] >= dim[1] else box[3]

    return box


"""
    # ------------------------------------------------------------------------------------------------------------------
    def hough_detection(self, check_roi):

        gray = self.tar_img[check_roi[1]:check_roi[3], check_roi[0]:check_roi[2]]
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * -b)
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * -b)
            y2 = int(y0 - 1000 * a)

            cv2.line(gray, (x1, y1), (x2, y2), mu.RED, 2)

        hp.hp_imshow(gray)

"""
