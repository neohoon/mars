#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import sys
import numpy as np
import logging
import datetime
import cv2
import random
import operator
from copy import deepcopy
from PIL import Image
import glob
from operator import itemgetter
import traceback
import subprocess
import re
import socket
import errno
import time


if False:
    try:
        import Tkinter as tk
    except ImportError:
        import tkinter as tk

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
    except all:
        screen_width = 1920
        screen_height = 1080
        print(" @ Warning in getting screen width and height...\n")
else:
    screen_width = 1920
    screen_height = 1080


RED     = (255,   0,   0)
GREEN   = (  0, 255,   0)
BLUE    = (  0,   0, 255)
CYAN    = (  0, 255, 255)
MAGENTA = (255,   0, 255)
YELLOW  = (255, 255,   0)
WHITE   = (255, 255, 255)
BLACK   = (  0,   0,   0)

IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tif', 'tiff']

COLORS = [RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW]
DEV_NULL = open(os.devnull, 'w')

RECV_BUF_SIZE = 1024


class LoggerWrapper:

    def info(self): pass

    def error(self): pass


def read_pdf(pdf_filename, resolution=300):
    """
    Read pdf file.

    :param pdf_filename:
    :param resolution:
    :return img:
    """
    img_filename = 'temp.bmp'
    convert_pdf_to_img(pdf_filename, 'bmp', img_filename, resolution=resolution)
    img = hp_imread(img_filename, color_fmt='RGB')
    os.remove(img_filename)
    return img


def convert_pdf_to_img(pdf_filename, img_type, img_filename, resolution=300):
    """
    Convert pdf file to image file.

    :param pdf_filename:
    :param img_type:
    :param img_filename:
    :param resolution:
    :return img_filename:
    """
    if os.name == 'nt':
        print(" @ Error: Wand library does not work in Windows OS\n")
        sys.exit()
    else:
        from wand.image import Image
        with Image(filename=pdf_filename, resolution=resolution) as img:
            img.compression = 'no'
            with img.convert(img_type) as converted:
                converted.save(filename=img_filename)
        return img_filename


def imread(img_file, color_fmt='RGB'):
    """
    Read image file.
    Support gif and pdf format.

    :param  img_file:
    :param  color_fmt: RGB, BGR, or GRAY. The default is RGB.
    :return img:
    """
    if not isinstance(img_file, str):
        # print(" % Warning: input is NOT a string for image filename")
        return img_file

    if not os.path.exists(img_file):
        print(" @ Error: image file not found {}".format(img_file))
        sys.exit()

    if not(color_fmt == 'RGB' or color_fmt == 'BGR' or color_fmt == 'GRAY'):
        color_fmt = 'RGB'

    if img_file.split('.')[-1] == 'gif':
        gif = cv2.VideoCapture(img_file)
        ret, img = gif.read()
        if not ret:
            return None
    elif img_file.split('.')[-1] == 'pdf':
        img = read_pdf(img_file, resolution=300)
    else:
        # img = cv2.imread(img_file.encode('utf-8'))
        # img = cv2.imread(img_file)
        # img = np.array(Image.open(img_file.encode('utf-8')).convert('RGB'), np.uint8)
        img = np.array(Image.open(img_file).convert('RGB'), np.uint8)

    if color_fmt.upper() == 'GRAY':
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif color_fmt.upper() == 'BGR':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        return img


def imwrite(img, img_fname, color_fmt='RGB'):
    """
    write image file.

    :param img:
    :param img_fname:
    :param  color_fmt: RGB, BGR, or GRAY. The default is RGB.
    :return img:
    """
    if color_fmt == 'RGB':
        tar = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif color_fmt == 'GRAY':
        tar = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif color_fmt == 'BGR':
        tar = img[:]
    else:
        print(" @ Error: color_fmt, {}, is not correct.".format(color_fmt))
        return False

    cv2.imwrite(img_fname, tar)
    return True


def read_all_images(img_dir,
                    prefix='',
                    exts=IMG_EXTENSIONS,
                    color_fmt='RGB'):
    """
    Read all images with specific filename prefix in an image directory.

    :param img_dir:
    :param prefix:
    :param exts:
    :param color_fmt:
    :color_fmt:
    :return imgs: image list
    """
    imgs = []

    filenames = os.listdir(img_dir)
    filenames.sort()

    for filename in filenames:
        if filename.startswith(prefix) and os.path.splitext(filename)[-1][1:] in exts:
            img = hp_imread(os.path.join(img_dir, filename), color_fmt=color_fmt)
            imgs.append(img)

    if not imgs:
        print(" @ Error: no image filename starting with \"{}\"...".format(prefix))
        sys.exit()

    return imgs


def imread_all_images(img_path,
                      fname_prefix='',
                      img_extensions=IMG_EXTENSIONS,
                      color_fmt='RGB'):
    """
    Read all images in the specific folder.

    :param img_path:
    :param fname_prefix:
    :param img_extensions:
    :param color_fmt:
    :return imgs: image list
    """
    img_filenames = []
    imgs = []

    if os.path.isfile(img_path):
        filenames = [img_path]
    elif os.path.isdir(img_path):
        filenames = [os.path.join(img_path, x) for x in os.listdir(img_path)]
    else:
        print(" @ Error: The input argument is NOT a file nor folder.\n")
        return [], []

    filenames.sort()
    for filename in filenames:
        if os.path.splitext(filename)[1][1:] in img_extensions:
            if os.path.basename(filename).startswith(fname_prefix):
                imgs.append(hp_imread(filename, color_fmt=color_fmt))
                img_filenames.append(filename)

    return imgs, img_filenames


def imshow(img, desc='imshow', zoom=1.0, color_fmt='RGB', skip=False, pause_sec=0, loc=(10,10)):
    """

    :param img:
    :param desc:
    :param zoom:
    :param color_fmt:
    :param skip:
    :param pause_sec:
    :param loc:
    :return:
    """
    global screen_width, screen_height
    if skip or pause_sec < 0:
        return

    if isinstance(img, str):
        img = hp_imread(img)

    if len(img.shape) is 3 and color_fmt == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # def get_full_zoom_factor(img, zoom=0.0, factor=4/5., skip_=False):

    dim = img.shape
    h, w = dim[0], dim[1]
    zoom_w = screen_width  / float(w) * 9/10.
    zoom_h = screen_height / float(h) * 9/10.

    if zoom == 0:
        zoom = min(zoom_h, zoom_w)
    elif (screen_width < w * zoom) or (screen_height < h * zoom):
        zoom = min(zoom_h, zoom_w)
    else:
        pass

    cv2.imshow(desc, cv2.resize(img, (0,0), fx=zoom, fy=zoom))
    if loc[0] >= 0 and loc[1] >= 0:
        cv2.moveWindow(desc, loc[0], loc[1])
    if pause_sec <= 0:
        cv2.waitKey()
    else:
        cv2.waitKey(pause_sec*1000)
    cv2.waitKey(1)
    cv2.destroyWindow(desc)
    for k in range(10):
        cv2.waitKey(1)


def draw_box_on_img(img, box, color=RED, thickness=2, alpha=0.5):
    """
    Draw a box overlay to image.
    box format is either 2 or 4 vertex points.

    :param img:
    :param box:
    :param color:
    :param thickness:
    :param alpha:
    :return:
    """
    if np.array(box).size == 8:
        box = [box[0][0], box[0][1], box[3][0], box[3][1]]

    box_img = cv2.rectangle(deepcopy(img), tuple(box[0:2]), tuple(box[2:]), color, thickness)
    box_img = cv2.addWeighted(img, alpha, box_img, 1-alpha, 0)
    return box_img


def draw_boxes_on_img(img,
                      boxes,
                      color=RED,
                      thickness=2,
                      alpha=0.,
                      margin=0,
                      add_cross_=False ):
    """
    Draw the overlay of boxes to an image.
    box format is either 2 or 4 vertex points.

    :param img:
    :param boxes:
    :param color: color vector such as (R,G,B) or (R,G,B,alpha) or string 'random'
    :param thickness:
    :param alpha:
    :param margin:
    :param add_cross_:
    :return:
    """
    margins = [ x*margin for x in [-1, -1, 1, 1]]

    if isinstance(color, str):
        if color.lower() == 'random':
            box_color = -1
        else:
            box_color = RED
        box_alpha = alpha
    else:
        if len(color) == 4:
            box_color = color[:3]
            box_alpha = color[3]
        else:
            box_color = color
            box_alpha = alpha

    box_img = np.copy(img)
    for cnt, box in enumerate(boxes):
        if np.array(box).size == 8:
            box = [box[0][0], box[0][1], box[3][0], box[3][1]]
        if box_color == -1:
            rand_num = random.randint(0,len(COLORS)-1)
            mod_color = COLORS[rand_num]
        else:
            mod_color = box_color
        mod_box = list(map(operator.add, box, margins))
        box_img = cv2.rectangle(box_img, tuple(mod_box[0:2]), tuple(mod_box[2:]), mod_color, thickness)
        if add_cross_:
            box_img = cv2.line(box_img, (mod_box[0],mod_box[1]), (mod_box[2],mod_box[3]), color=BLACK, thickness=8)
            box_img = cv2.line(box_img, (mod_box[2],mod_box[1]), (mod_box[0],mod_box[3]), color=BLACK, thickness=8)
    disp_img = cv2.addWeighted(np.copy(img), box_alpha, box_img, 1-box_alpha, 0)

    return disp_img


def draw_quadrilateral_on_image(img, vertices, color=RED, thickness=2):
    """
    Draw a quadrilateral on image.
    This function includes the regularization of quadrilateral vertices.

    :param img:
    :param vertices:
    :param color:
    :param thickness:
    :return:
    """
    mod_vertices = regularize_quadrilateral_vertices(vertices)
    disp_img = np.copy(img)

    disp_img = cv2.line(disp_img, tuple(mod_vertices[0]), tuple(mod_vertices[1]), color=color, thickness=thickness)
    disp_img = cv2.line(disp_img, tuple(mod_vertices[0]), tuple(mod_vertices[2]), color=color, thickness=thickness)
    disp_img = cv2.line(disp_img, tuple(mod_vertices[3]), tuple(mod_vertices[1]), color=color, thickness=thickness)
    disp_img = cv2.line(disp_img, tuple(mod_vertices[3]), tuple(mod_vertices[2]), color=color, thickness=thickness)

    return disp_img


def generate_four_vertices_from_ref_vertex(ref, img_sz):
    """
    Generate four vertices from top-left reference vertex.

    :param ref:
    :param img_sz:
    :return:
    """

    pt_tl = [int(img_sz[0] * ref[0]), int(img_sz[1] * ref[1])]
    # pt_tr = [int(img_sz[0]), pt_tl[1]]
    pt_tr = [int(img_sz[0] - pt_tl[0]), pt_tl[1]]
    pt_bl = [pt_tl[0], int(img_sz[1] - pt_tl[1])]
    pt_br = [pt_tr[0], pt_bl[1]]

    return [pt_tl, pt_tr, pt_bl, pt_br]


def crop_image_from_ref_vertex(img, ref_vertex, symm_crop_=True, debug_=False):
    """
    Crop input image with reference vertex.

    :param img:
    :param ref_vertex:
    :param symm_crop_:
    :param debug_:
    :return:
    """
    pts = generate_four_vertices_from_ref_vertex(ref_vertex, img.shape[1::-1])
    if symm_crop_:
        crop_img = img[pts[0][1]:pts[3][1], pts[0][0]:pts[1][0]]
    else:
        crop_img = img[pts[0][1]:pts[3][1], pts[0][0]:img.shape[1]]

    if debug_:
        hp_imshow(draw_box_on_img(img, pts, color=RED, thickness=10, alpha=0.5), desc="original image with frame")
        hp_imshow(crop_img, desc="cropped image")
    return crop_img


def crop_image_with_coordinates(img, crop_coordinates):
    width_point_start = int(img.shape[1] * crop_coordinates[0])
    width_point_end = int(img.shape[1] * crop_coordinates[1])
    height_point_start = int(img.shape[0] * crop_coordinates[2])
    height_point_end = int(img.shape[0] * crop_coordinates[3])
    crop_img = img[height_point_start:height_point_end, width_point_start:width_point_end]

    return crop_img


def get_datetime(fmt="%Y-%m-%d-%H-%M-%S"):
    """ Get datetime with format argument.
    :param fmt:
    :return:
    """
    return datetime.datetime.now().strftime(fmt)


def setup_logger(logger_name,
                 log_prefix_name,
                 level=logging.INFO,
                 folder='.',
                 logger_=True,
                 console_=True):
    """ Setup logger supporting two handlers of stdout and file.
    :param logger_name:
    :param log_prefix_name:
    :param level:
    :param folder:
    :param logger_:
    :param console_:
    :return:
    """

    if not logger_:
        logger = LoggerWrapper()
        logger.info = print
        logger.error = print
        return logger

    if not os.path.exists(folder):
        os.makedirs(folder)

    today_time = str(datetime.date.today())
    log_file = os.path.join(*folder.split('/'), log_prefix_name + today_time + '.log')
    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter('\033[1;32m' + '%(name)-10s | %(asctime)s | %(levelname)-7s | %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    stream_handler = None
    if console_:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(file_handler)
    if console_:
        log_setup.addHandler(stream_handler)
    return logging.getLogger(logger_name)


def check_directory_existence(in_dir, exit_=False, create_=False, print_=True):
    """
    Check if a directory exists or not. If not, create it according to input argument.

    :param in_dir:
    :param exist_:
    :param exit_:
    :param create_:
    :param print_:
    :return:
    """
    if os.path.isdir(in_dir):
        return True
    else:
        if create_:
            try:
                os.makedirs(in_dir)
            except all:
                print(" @ Error: make_dirs in check_directory_existence routine...\n")
                sys.exit()
        else:
            if print_:
                print("\n @ Warning: directory not found, {}.\n".format(in_dir))
            if exit_:
                sys.exit()
        return False


def check_file_existence(filename, print_=False, exit_=False):
    """
    Check if a file exists or not.

    :param filename:
    :param print_:
    :param exit_:
    :return True/Flase:
    """
    if not os.path.isfile(filename):
        if print_ or exit_:
            print("\n @ Warning: file not found, {}.\n".format(filename))
        if exit_:
            sys.exit()
        return False
    else:
        return True


def is_string_nothing(string):
    if string == '' or string is None:
        return True
    else:
        return False


def get_filenames_in_a_directory(dir_name):
    """
    Get names of all the files in a directory.

    :param dir_name:
    :return out_filenames:
    """
    filenames = os.listdir(dir_name)
    out_filenames = []
    for filename in filenames:
        if os.path.isfile(os.path.join(dir_name, filename)):
            out_filenames.append(filename)

    return out_filenames


def transpose_list(in_list):
    """
    Transpose a 2D list variable.

    :param in_list:
    :return:
    """
    try:
        len(in_list[0])
        return list(map(list, zip(*in_list)))
    except TypeError:
        return in_list


def plt_imshow(data_2d,
               title=None,
               x_label=None,
               y_label=None,
               x_range=None,
               y_range=None,
               xticks=None,
               yticks=None,
               maximize_=True,
               block_=True):
    """
    Show image via matplotlib.pyplot.

    :param data_2d:
    :param title:
    :param x_label:
    :param y_label:
    :param x_range:
    :param y_range:
    :param xticks:
    :param yticks:
    :param maximize_:
    :param block_:
    :return:
    """

    maximize_ = maximize_ and False
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
    plt.show(block=block_)


def check_string_in_class(class_name, sub_string):
    for attr in dir(class_name):
        if sub_string in attr:
            print(attr)


def vstack_images(imgs, margin=20):
    """
    Stack images vertically with boundary and in-between margin.

    :param imgs:
    :param margin:
    :return:
    """
    widths = []
    heights = []
    num_imgs = len(imgs)

    if num_imgs == 1:
        return imgs[0]

    color_images = []
    for img in imgs:
        img_sz = img.shape[1::-1]
        widths.append(img_sz[0])
        heights.append(img_sz[1])
        color_images.append(img)

    max_width = max(widths) + 2 * margin
    max_height = sum(heights) + (num_imgs + 1) * margin
    if len(imgs[0].shape) == 3:
        vstack_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    else:
        vstack_image = np.zeros((max_height, max_width), dtype=np.uint8)

    x_offset = margin
    y_offset = margin
    for img in color_images:
        img_sz = img.shape[1::-1]
        vstack_image[y_offset:y_offset+img_sz[1], x_offset:x_offset+img_sz[0]] = img
        y_offset += margin + img_sz[1]

    return vstack_image


def hstack_images(imgs, margin=20):
    """
    Stack images horizontally with boundary and in-between margin.

    :param imgs:
    :param margin:
    :return:
    """
    widths = []
    heights = []
    num_imgs = len(imgs)

    if num_imgs == 1:
        return imgs[0]

    color_images = []
    for img in imgs:
        img_sz = img.shape[1::-1]
        widths.append(img_sz[0])
        heights.append(img_sz[1])
        color_images.append(img)

    max_width = sum(widths) + (num_imgs + 1) * margin
    max_height = max(heights) + 2 * margin
    if len(imgs[0].shape) == 3:
        hstack_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    else:
        hstack_image = np.zeros((max_height, max_width), dtype=np.uint8)

    x_offset = margin
    y_offset = margin
    for img in color_images:
        img_sz = img.shape[1::-1]
        hstack_image[y_offset:y_offset+img_sz[1], x_offset:x_offset+img_sz[0]] = img
        x_offset += margin + img_sz[0]

    return hstack_image


def get_all_files_in_dir_path(dir_path, prefixes='', extensions=''):
    """
    Find all the files starting with prefixes or ending with extensions in the directory path.
    ${dir_path} argument can accept file.

    :param dir_path:
    :param prefixes:
    :param extensions:
    :return:
    """
    filenames = []
    if os.path.isfile(dir_path):
        filenames.append(dir_path)
    for path, _, files in os.walk(dir_path):
        for name in files:
            if name.startswith(tuple(prefixes)):
                filenames.append(os.path.join(path,name))
            if name.endswith(tuple(extensions)):
                filenames.append(os.path.join(path,name))
    if not filenames:
        filenames = glob.glob(dir_path + "*")
    return filenames


def check_box_boundary(box, sz):
    box[0] = 0     if box[0] < 0     else box[0]
    box[1] = 0     if box[1] < 0     else box[1]
    box[2] = sz[0] if box[2] > sz[0] else box[2]
    box[3] = sz[1] if box[3] > sz[1] else box[3]
    return box


def regularize_quadrilateral_vertices(vertices):
    """
    Regularize quadrilateral vertices to the de-facto rule. (LT, RT, LB, RB)

    :param vertices: 2D list of position list of (x, y)
    :return:
    """
    out = sorted(vertices, key=itemgetter(0))
    if out[0][1] > out[1][1]:
        out[0], out[1] = out[1], out[0]
    if out[2][1] > out[3][1]:
        out[2], out[3] = out[3], out[2]
    out[1], out[2] = out[2], out[1]
    return out


def recv_all(connection, timeout_val=10., logger=None, file_prefix=None):
    byte_data = b''
    data_len_list = None    # [16, 15, 527, 837, 842]
    connection.settimeout(timeout_val)
    while True:
        try:
            part = connection.recv(RECV_BUF_SIZE)
            byte_data += part
            if len(part) == RECV_BUF_SIZE:
                if data_len_list:
                    try:
                        if data_len_list.index(len(byte_data)) >= 0:
                            logger.info("Total packet length is {:d}".format(len(byte_data)))
                            return byte_data
                    except ValueError:
                        pass
            else:
                break
        except connection.error as e:
            if logger:
                logger.error("socket error: {}".format(str(e)))
            else:
                print(e)
            break
        except Exception as e:
            if logger:
                logger.error(str(e) + "\n" + traceback.format_exc())
            else:
                print(str(e) + "\n" + traceback.format_exc())
            break

    if file_prefix:
        with open(os.path.join("log", file_prefix + "_" + get_datetime() + ".txt"), "wb") as fid:
            fid.write(byte_data)

    return byte_data


def get_pids(port):
    command = "sudo -S lsof -i :%s | awk '{print $2}'" % port
    pids = subprocess.check_output(command, shell=True)
    pids = pids.strip().decode('utf-8')
    if pids:
        pids = re.sub(' +', ' ', pids)
        for pid in pids.split('\n'):
            try:
                yield str(pid)
            except TypeError:
                pass


def check_addr_and_port_in_use(server_addr, debug_=False):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    in_use = False
    try:
        sock.bind(server_addr)
    except socket.error as e:
        if e.errno == errno.EADDRINUSE:
            if debug_:
                print("# Server - {}:{:d} is in use.".format(server_addr[0], server_addr[1]))
            in_use = True
    sock.close()
    return in_use


def kill_process_with_port(server_addr, debug_=False):
    pids = get_pids(server_addr[1])
    if debug_:
        print("@ Kill the process of {}:{:d}".format(server_addr[0], server_addr[1]))
    os.system('sudo -S kill -9 {}'.format(''.join([str(pid) for pid in pids])))
    return True


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


def hp_imread(): pass


def hp_imshow(): pass


def hp_imread_all_images(): pass


def hp_imwrite(): pass


hp_imread = imread
hp_imshow = imshow
hp_imwrite = imwrite
hp_imread_all_images = imread_all_images
