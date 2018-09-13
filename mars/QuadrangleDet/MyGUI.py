#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import cv2
import numpy as np
import HoonUtils as hp
from operator import itemgetter


g_img_full = []
g_img_crop = []
g_img_zoom = []
g_img_canvas = []

g_cropping  = False
g_cropped   = False
g_selecting = False
g_selected  = False
g_changed   = True
g_quadrilateral = False

g_click_cnt = 0
g_click_pos = []

g_crop_box = []
g_select_box = []

g_scale_full = 0.
g_scale_crop = 0.

xi, yi = -1, -1
xe, ye = -1, -1

MOUSE_OFFSET = (0, 0)


def zoom_and_crop(event, x, y, flags, param):

    global g_img_full, g_img_crop, g_img_zoom, g_img_canvas
    global g_cropping, g_cropped, g_selecting, g_selected
    global g_crop_box, g_select_box
    global g_scale_full, g_scale_crop
    global xi, yi, xe, ye

    if not g_cropped:
        if event == cv2.EVENT_LBUTTONDOWN:
            g_cropping = True
            xi, yi = x, y
            xi -= MOUSE_OFFSET[0]
            yi -= MOUSE_OFFSET[1]
        elif event == cv2.EVENT_MOUSEMOVE:
            if g_cropping is True:
                g_img_canvas = cv2.rectangle(np.copy(g_img_zoom), (xi, yi), (x, y), hp.RED, 2)
        elif event == cv2.EVENT_LBUTTONUP:
            xe, ye = x, y
            xe -= MOUSE_OFFSET[0]
            ye -= MOUSE_OFFSET[1]
            g_cropping = False
            g_cropped  = True
            xi = int(xi / g_scale_full)
            yi = int(yi / g_scale_full)
            xe = int(xe / g_scale_full)
            ye = int(ye / g_scale_full)
            dim = g_img_full.shape
            if 0 <= xi < xe < dim[1] and 0 <= yi < ye < dim[0]:
                g_img_crop = g_img_full[yi:ye,xi:xe]
                g_crop_box = [xi, yi, xe, ye]
                g_img_zoom, g_scale_crop = hp.imresize_full(g_img_crop)
                g_img_canvas = np.copy(g_img_zoom)
                dim = g_img_canvas.shape
                cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
                g_selecting = False
                g_selected  = False
                g_select_box = []
            else:
                g_img_canvas = np.copy(g_img_full)
                dim = g_img_canvas.shape
                cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
                g_cropped  = False
                g_selected = False
    elif g_cropped and not g_selected:
        if event == cv2.EVENT_LBUTTONDOWN:
            g_selecting = True
            xi, yi = x, y
            xi -= MOUSE_OFFSET[0]
            yi -= MOUSE_OFFSET[1]
        elif event == cv2.EVENT_MOUSEMOVE:
            if g_selecting is True:
                g_img_canvas = cv2.rectangle(np.copy(g_img_zoom), (xi, yi), (x, y), hp.BLUE, 2)
        elif event == cv2.EVENT_LBUTTONUP:
            xe, ye = x, y
            xe -= MOUSE_OFFSET[0]
            ye -= MOUSE_OFFSET[1]
            g_selecting = False
            g_selected  = True
            xi -= 4
            yi -= 4
            xe += 4
            ye += 4
            g_img_canvas = cv2.rectangle(g_img_canvas, (xi, yi), (xe, ye), hp.GREEN, 2)
            dim = g_img_canvas.shape
            cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
            xi = int(xi / g_scale_crop)
            yi = int(yi / g_scale_crop)
            xe = int(xe / g_scale_crop)
            ye = int(ye / g_scale_crop)
            g_select_box = [xi, yi, xe, ye]
            g_img_zoom = g_img_canvas


def specify_roi(img, win_loc=(10,10), color_fmt='RGB'):

    global g_img_full, g_img_zoom, g_img_canvas
    global g_cropped, g_selected
    global g_scale_full, g_scale_crop

    if isinstance(img, str):
        img = hp.imread_safe(img)

    cv2.namedWindow('zoom_and_crop')
    cv2.namedWindow('zoom_and_crop', cv2.WINDOW_NORMAL)
    cv2.namedWindow('zoom_and_crop', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('zoom_and_crop', zoom_and_crop)
    roi = []
    g_img_full = np.copy(img)

    print(" # Zoom and crop...")
    disp_img = np.copy(g_img_full)
    g_img_zoom, g_scale_full = hp.imresize_full(disp_img)
    g_img_canvas = g_img_zoom

    first = True
    while True:
        if color_fmt == 'RGB':
            cv2.imshow('zoom_and_crop', cv2.cvtColor(g_img_canvas, cv2.COLOR_RGB2BGR))
        else:
            cv2.imshow('zoom_and_crop', g_img_canvas)
        if first:
            cv2.moveWindow('zoom_and_crop', win_loc[0], win_loc[1])
            first = False
        in_key = cv2.waitKey(1) & 0xFF
        if (in_key == 27):
            print(" @ ESCAPE !! Something wrong...\n")
            sys.exit()
        elif g_selected:
            if in_key == ord('n'):
                disp_img = np.copy(g_img_full)
                g_img_zoom, g_scale_full = hp.imresize_full(disp_img)
                g_img_canvas = g_img_zoom
                dim = g_img_canvas.shape
                cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
                g_cropped  = False
                g_selected = False
            if in_key == ord('y'):
                roi = [g_select_box[0]+g_crop_box[0], g_select_box[1]+g_crop_box[1],
                       g_select_box[2]+g_crop_box[0], g_select_box[3]+g_crop_box[1]]
                g_img_canvas = np.copy(g_img_full)
                dim = g_img_canvas.shape
                cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
                g_cropped  = False
                g_selected = False
                break
            else:
                dim = g_img_canvas.shape
                g_img_canvas = cv2.putText(g_img_canvas,
                                           "Press \'y\' for yes or \'n\' for no",
                                           (10, dim[0]-50),
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           0.5,
                                           hp.RED,
                                           2)
    cv2.destroyAllWindows()
    for i in range(5):
        cv2.waitKey(1)
    return roi, img[roi[1]:roi[3],roi[0]:roi[2]]


# ----------------------------------------------------------------------------------------------------------------------
def specify_roi_line(img):

    global g_img_full, g_img_zoom, g_img_canvas
    global g_cropped, g_selected
    global g_scale_full, g_scale_crop

    cv2.namedWindow('zoom_and_crop')
    cv2.namedWindow('zoom_and_crop', cv2.WINDOW_NORMAL)
    cv2.namedWindow('zoom_and_crop', cv2.WINDOW_KEEPRATIO)
    # cv2.setMouseCallback('zoom_and_crop', zoom_and_crop_line)
    cv2.setMouseCallback('zoom_and_crop', zoom_and_crop)
    roi = []
    g_img_full = np.copy(img)

    print(" # Zoom and crop...")
    disp_img = np.copy(g_img_full)
    g_img_zoom, g_scale_full = hp.imresize_full(g_img_full)
    g_img_canvas = g_img_zoom

    while True:
        cv2.imshow('zoom_and_crop', g_img_canvas)
        in_key = cv2.waitKey(1) & 0xFF
        if (in_key == 27) or (in_key == ord('x')):
            print(" @ Something wrong ? Bye...\n\n")
            return False
        elif in_key == ord('n'):
            g_img_zoom, g_scale_full = hp.imresize_full(np.copy(g_img_full))
            g_img_canvas = g_img_zoom
            # dim = g_img_canvas.shape
            # cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
            g_cropped = False
            g_selected = False
        elif g_selected:
            if in_key == ord('y'):
                for k in range(len(g_select_box)):
                    roi.append([g_select_box[k][0]+g_crop_box[0], g_select_box[k][1]+g_crop_box[1],
                                g_select_box[k][2]+g_crop_box[0], g_select_box[k][3]+g_crop_box[1]])
                g_img_canvas = np.copy(g_img_full)
                dim = g_img_canvas.shape
                cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
                g_cropped = False
                g_selected = 0
                break
            else:
                dim = g_img_canvas
                g_img_canvas = cv2.putText(g_img_canvas,
                                           "Press \'y\' for yes or \'n\' for no",
                                           (10, dim[0]-10),
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           1,
                                           hp.RED,
                                           4)
    cv2.destroyAllWindows()
    return roi


def specify_rois(img, roi_num, win_loc=(10,10)):

    global g_img_full, g_img_zoom, g_img_canvas
    global g_cropped, g_selected
    global g_scale_full, g_scale_crop

    cv2.namedWindow('zoom_and_crop')
    cv2.namedWindow('zoom_and_crop', cv2.WINDOW_NORMAL)
    cv2.namedWindow('zoom_and_crop', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('zoom_and_crop', zoom_and_crop)
    roi = []
    g_img_full = np.copy(img)

    for idx in range(roi_num):

        print(" # Zoom and crop {:d}...".format(idx+1))
        disp_img = np.copy(g_img_full)
        for k1 in range(idx):
            disp_img = cv2.rectangle(disp_img, tuple(roi[k1][:2]), tuple(roi[k1][2:]), hp.RED, -1)
        g_img_zoom, g_scale_full = hp.imresize_full(disp_img)
        g_img_canvas = g_img_zoom

        first = True
        while True:
            cv2.imshow('zoom_and_crop', g_img_canvas)
            if first:
                cv2.moveWindow('zoom_and_crop', win_loc[0], win_loc[1])
                first = False
            in_key = cv2.waitKey(1) & 0xFF
            if (in_key == 27) or (in_key == ord('x')):
                print(" @ Something wrong ? Bye...\n\n")
                return False
            elif in_key == ord('n'):
                disp_img = np.copy(g_img_full)
                for k1 in range(idx):
                    disp_img = cv2.rectangle(disp_img, tuple(roi[k1][:2]), tuple(roi[k1][2:]), hp.RED, -1)
                g_img_zoom, g_scale_full = hp.imresize_full(disp_img)
                g_img_canvas = g_img_zoom
                dim = g_img_canvas.shape
                cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
                g_cropped  = False
                g_selected = False
            elif in_key == ord('y'):
                for k in range(len(g_select_box)):
                    roi.append([g_select_box[k][0]+g_crop_box[0], g_select_box[k][1]+g_crop_box[1],
                                g_select_box[k][2]+g_crop_box[0], g_select_box[k][3]+g_crop_box[1]])
                g_img_canvas = np.copy(g_img_full)
                dim = g_img_canvas.shape
                cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
                g_cropped = False
                g_selected = 0
                break
            else:
                dim = g_img_canvas.shape
                g_img_canvas = cv2.putText(g_img_canvas, "Press \'y\' for yes or \'n\' for no", (10, dim[0]-50),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, hp.RED, 4)
    cv2.destroyAllWindows()
    return roi


def specify_roi_line(img, win_loc=(10,10)):

    global g_img_full, g_img_zoom, g_img_canvas
    global g_cropped, g_selected
    global g_scale_full, g_scale_crop

    cv2.namedWindow('zoom_and_crop')
    cv2.namedWindow('zoom_and_crop', cv2.WINDOW_NORMAL)
    cv2.namedWindow('zoom_and_crop', cv2.WINDOW_KEEPRATIO)
    # cv2.setMouseCallback('zoom_and_crop', zoom_and_crop_line)
    cv2.setMouseCallback('zoom_and_crop', zoom_and_crop)
    roi = []
    g_img_full = np.copy(img)

    print(" # Zoom and crop...")
    disp_img = np.copy(g_img_full)
    g_img_zoom, g_scale_full = hp.imresize_full(g_img_full)
    g_img_canvas = g_img_zoom

    first = True
    while True:
        cv2.imshow('zoom_and_crop', g_img_canvas)
        if first:
            cv2.moveWindow('zoom_and_crop', win_loc[0], win_loc[1])
            first = False
        in_key = cv2.waitKey(1) & 0xFF
        if (in_key == 27) or (in_key == ord('x')):
            print(" @ Something wrong ? Bye...\n\n")
            return False
        elif in_key == ord('n'):
            g_img_zoom, g_scale_full = hp.imresize_full(np.copy(g_img_full))
            g_img_canvas = g_img_zoom
            # dim = g_img_canvas.shape
            # cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
            g_cropped = False
            g_selected = False
        elif g_selected:
            if in_key == ord('y'):
                for k in range(len(g_select_box)):
                    roi.append([g_select_box[k][0]+g_crop_box[0], g_select_box[k][1]+g_crop_box[1],
                                g_select_box[k][2]+g_crop_box[0], g_select_box[k][3]+g_crop_box[1]])
                g_img_canvas = np.copy(g_img_full)
                dim = g_img_canvas.shape
                cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
                g_cropped = False
                g_selected = 0
                break
            else:
                dim = g_img_canvas
                g_img_canvas = cv2.putText(g_img_canvas, "Press \'y\' for yes or \'n\' for no", (10, dim[0]-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 2, hp.RED, 8)
    cv2.destroyAllWindows()
    return roi


def zoom_and_pick(event, x, y, flags, param):

    global g_img_full, g_img_crop, g_img_zoom, g_img_canvas
    global g_cropping, g_cropped, g_selecting, g_selected
    global g_crop_box, g_select_box
    global g_scale_full, g_scale_crop
    global xi, yi, xe, ye
    global g_changed

    if not g_cropped:
        if event == cv2.EVENT_LBUTTONDOWN:
            g_cropping = True
            xi, yi = x, y
            print(" Before zoom: (xi, yi) = {:d} {:d}".format(xi, yi))
        elif event == cv2.EVENT_MOUSEMOVE:
            if g_cropping is True:
                g_img_canvas = cv2.rectangle(np.copy(g_img_zoom), (xi, yi), (x, y), hp.RED, 2)
        elif event == cv2.EVENT_LBUTTONUP:
            xe, ye = x, y
            print(" Before zoom: (xe, ye) = {:d} {:d}".format(xe, ye))
            g_cropping = False
            g_cropped = True
            xi = int(xi / g_scale_full)
            yi = int(yi / g_scale_full)
            xe = int(xe / g_scale_full)
            ye = int(ye / g_scale_full)
            g_img_crop = g_img_full[yi:ye,xi:xe]
            g_crop_box = [xi, yi, xe, ye]
            g_img_zoom, g_scale_crop = hp.imresize_full(g_img_crop)
            g_img_canvas = np.copy(g_img_zoom)
            g_selecting = False
            g_selected  = 0
            g_select_box = []
            g_changed = True
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            g_selecting = True
            xi, yi = x, y
            print(" after zoom: (xi, yi) = {:d} {:d}".format(xi, yi))
        elif event == cv2.EVENT_MOUSEMOVE:
            if g_selecting is True:
                g_img_canvas = cv2.rectangle(np.copy(g_img_zoom), (xi, yi), (x, y), hp.BLUE, 2)
        elif event == cv2.EVENT_LBUTTONUP:
            xe, ye = x, y
            g_selecting = False
            g_selected += 1
            print(" After zoom: (xe, ye) = {:d} {:d}, g_selected = {:d}".format(xe, ye, g_selected))
            xi -= 4
            yi -= 4
            xe += 4
            ye += 4
            g_img_canvas = cv2.rectangle(g_img_canvas, (xi, yi), (xe, ye), hp.GREEN, 2)
            xi = int(xi / g_scale_crop)
            yi = int(yi / g_scale_crop)
            xe = int(xe / g_scale_crop)
            ye = int(ye / g_scale_crop)
            g_select_box.append([xi, yi, xe, ye])
            g_img_zoom = g_img_canvas
            g_changed = True


def zoom_and_click(event, x, y, flags, param):

    global g_img_full
    global g_img_zoom
    global g_img_canvas
    global g_img_crop
    global g_crop_box
    global g_scale_full
    global g_scale_crop
    global g_cropped
    global g_cropping
    global xi, yi
    global xe, ye
    global g_changed
    global g_click_pos

    if not g_cropped:
        if event == cv2.EVENT_LBUTTONDOWN:
            g_cropping = True
            xi, yi = x, y
            print(" # mouse event left button down : {:d} {:d}".format(xi, yi))
        elif event == cv2.EVENT_MOUSEMOVE:
            if g_cropping is True:
                g_img_canvas = cv2.rectangle(np.copy(g_img_zoom), (xi, yi), (x, y), hp.RED, 2)
        elif event == cv2.EVENT_LBUTTONUP:
            xe, ye = x, y
            print(" # mouse event left button up : {:d} {:d}".format(xe, ye))
            g_cropping = False
            g_cropped = True
            xi = int(xi / g_scale_full)
            yi = int(yi / g_scale_full)
            xe = int(xe / g_scale_full)
            ye = int(ye / g_scale_full)
            g_img_crop = g_img_full[yi:ye,xi:xe]
            g_crop_box = [xi, yi, xe, ye]
            g_img_zoom, g_scale_crop = hp.imresize_full(g_img_crop)
            g_img_canvas = np.copy(g_img_zoom)
            g_changed = True
            g_click_pos = []
    else:
        if event == cv2.EVENT_LBUTTONUP:
            g_click_pos.append([x, y])
            print(" # mouse event left button up : {:d} {:d} - {:d}".
                  format(g_click_pos[-1][0], g_click_pos[-1][1], len(g_click_pos)))
            g_img_canvas = cv2.circle(g_img_zoom, tuple(g_click_pos[-1]), 8, hp.RED, -1)
            # g_img_zoom = cv2.circle(g_img_zoom, tuple(g_click_pos[-1]), 8, hp.RED, -1)
            if len(g_click_pos) == 4:
                print(g_click_pos)
                g_click_pos = hp.regularize_quadrilateral_vertices(g_click_pos)
                print(g_click_pos)
                g_img_canvas = hp.draw_quadrilateral_on_image(g_img_canvas, g_click_pos, color=hp.BLUE, thickness=4)
                # g_img_zoom = hp.draw_quadrilateral_on_image(g_img_zoom, g_click_pos, color=hp.BLUE, thickness=4)
                for i in range(4):
                    g_click_pos[i] = [g_crop_box[0] + g_click_pos[i][0] / g_scale_crop,
                                      g_crop_box[1] + g_click_pos[i][1] / g_scale_crop]
            g_changed = True


def init_cv2_window(win_name, callback_func=zoom_and_pick):
    cv2.namedWindow(win_name)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_name, cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(win_name, callback_func)
    return True


def specify_ref_box_roi(ref_img, group_num, check_num):
    """
    Define boxes in an image based on the given information.

    :param ref_img:
    :param group_num:
    :param check_num:
    :return:
    """

    global g_img_full, g_img_zoom, g_img_canvas
    global g_cropped, g_selected
    global g_scale_full, g_scale_crop
    global g_changed

    init_cv2_window('zoom_and_pick')
    """
    cv2.namedWindow('zoom_and_pick')
    cv2.namedWindow('zoom_and_pick', cv2.WINDOW_NORMAL)
    cv2.namedWindow('zoom_and_pick', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('zoom_and_pick', zoom_and_pick)
    """
    ref_box_roi = []
    g_img_full = np.copy(ref_img)

    for idx in range(group_num):

        print(" # Zoom and pick the check-boxes in {:d}-th group...".format(idx+1))
        disp_img = np.copy(g_img_full)
        for k1 in range(idx):
            for k2 in range(check_num[idx]):
                # disp_img = cv2.rectangle(disp_img, tuple(ref_box_roi[k1][k2][:2]), tuple(ref_box_roi[k1][k2][2:]), hp.RED, -1)
                disp_img = hp.add_box_overlay(disp_img, ref_box_roi[k1][k2], hp.RED, 0.5)
        g_img_zoom, g_scale_full = hp.imresize_full(disp_img)
        ref_box_roi.append([])
        g_img_canvas = g_img_zoom

        while True:
            if g_changed:
                cv2.destroyWindow('zoom_and_pick')
                init_cv2_window('zoom_and_pick')
                g_changed = False
            cv2.imshow('zoom_and_pick', cv2.cvtColor(g_img_canvas, cv2.COLOR_RGB2BGR))
            in_key = cv2.waitKey(1) & 0xFF
            if (in_key == 27) or (in_key == ord('x')):
                print(" @ Something wrong ? Bye...\n\n")
                return False
            elif in_key == ord('n'):
                disp_img = np.copy(g_img_full)
                for k1 in range(idx):
                    for k2 in range(check_num[idx]):
                        # disp_img = cv2.rectangle(disp_img, tuple(ref_box_roi[k1][k2][:2]), tuple(ref_box_roi[k1][k2][2:]),
                        # hp.RED, -1)
                        disp_img = hp.add_box_overlay(disp_img, ref_box_roi[k1][k2], hp.RED, 0.5)
                g_img_zoom, g_scale_full = hp.imresize_full(disp_img)
                g_img_canvas = g_img_zoom
                g_cropped = False
                g_selected = 0
                g_changed = True
            elif g_selected == check_num[idx]:
                if in_key == ord('y'):
                    for k in range(len(g_select_box)):
                        ref_box_roi[idx].append([g_select_box[k][0]+g_crop_box[0], g_select_box[k][1]+g_crop_box[1],
                                                 g_select_box[k][2]+g_crop_box[0], g_select_box[k][3]+g_crop_box[1]])
                    g_img_canvas = np.copy(g_img_full)
                    g_cropped = False
                    g_selected = 0
                    g_changed = True
                    break
                else:
                    g_img_canvas = cv2.putText(g_img_canvas, "Press \'y\' for the next step or \'n\' for previous step",
                                               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, hp.RED, 8)
    cv2.destroyAllWindows()

    return ref_box_roi


def define_quadrilateral(ref_img):
    """
    Define a quadrilateral in an image by clicking four points.

    :param ref_img:
    :return:
    """

    global g_img_full, g_img_zoom, g_img_canvas
    global g_cropped, g_selected
    global g_scale_full, g_scale_crop
    global g_changed
    global g_click_pos

    init_cv2_window('zoom_and_click')
    g_img_full = np.copy(ref_img)

    g_img_zoom, g_scale_full = hp.imresize_full(g_img_full)
    g_img_canvas = g_img_zoom

    g_changed = True
    g_cropped = False
    g_click_pos = []
    while True:
        if g_changed:
            cv2.destroyWindow('zoom_and_click')
            init_cv2_window('zoom_and_click', callback_func=zoom_and_click)
            g_changed = False
        if len(g_click_pos) == 4:
            g_img_canvas = cv2.putText(g_img_canvas,
                                       "\'x\', \'s\' \'n\' \'y\'",
                                       (50, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       1,
                                       hp.RED,
                                       4)
        else:
            g_img_canvas = cv2.putText(g_img_canvas,
                                       "\'x\', \'s\'",
                                       (50, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       1,
                                       hp.BLUE,
                                       4)
        cv2.imshow('zoom_and_click', cv2.cvtColor(g_img_canvas, cv2.COLOR_RGB2BGR))
        in_key = cv2.waitKey(1) & 0xFF
        if (in_key == 27) or (in_key == ord('x')):
            cv2.destroyWindow('zoom_and_click')
            cv2.waitKey(1)
            return -1
        if in_key == ord('s'):
            cv2.destroyWindow('zoom_and_click')
            cv2.waitKey(1)
            return 0
        if len(g_click_pos) == 4:
            if in_key == ord('n'):
                g_img_zoom, g_scale_full = hp.imresize_full(g_img_full)
                g_img_canvas = g_img_zoom
                g_cropped = False
                g_changed = True
                g_click_pos = []
            elif in_key == ord('y'):
                break
            else:
                pass
    cv2.destroyAllWindows()

    return g_click_pos
