#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
import os
import sys
import argparse
import numpy as np
import cv2
import json
import MyGUI as mg
import HoonUtils as hp


CV2_FONT = cv2.FONT_HERSHEY_SIMPLEX


def main(args):
    hp.check_directory_existence(args.out_path, exit_=False, create_=True)
    img_filenames = hp.get_all_files_in_dir_path(args.img_path, extensions=hp.IMG_EXTENSIONS)
    img_filenames = sorted(img_filenames)

    for img_filename in img_filenames:
        img = hp.imread(img_filename)
        print(" # {} image file processing...".format(img_filename))
        pts1 = mg.define_quadrilateral(img)
        if pts1 == -1:
            sys.exit(1)
        elif pts1 == 0:
            continue
        pts1 = np.float32(pts1)
        h1_len = np.sqrt((pts1[0][0] - pts1[1][0])**2 + (pts1[0][1] - pts1[1][1])**2)
        h2_len = np.sqrt((pts1[2][0] - pts1[3][0])**2 + (pts1[2][1] - pts1[3][1])**2)
        v1_len = np.sqrt((pts1[0][1] - pts1[3][1])**2 + (pts1[0][1] - pts1[3][1])**2)
        v2_len = np.sqrt((pts1[1][1] - pts1[2][1])**2 + (pts1[1][1] - pts1[2][1])**2)
        h_len = int((h1_len + h2_len) / 2)
        v_len = int((v1_len + v2_len) / 2)
        pts2 = np.float32([[0, 0], [h_len, 0], [0, v_len], [h_len, v_len]])
        mtx = cv2.getPerspectiveTransform(pts1, pts2)
        warp_img = cv2.warpPerspective(img, mtx, (h_len, v_len))
        out_img = np.zeros((v_len+args.boundary_margin*2, h_len+args.boundary_margin*2, 3), dtype=np.uint8)
        out_img[args.boundary_margin:args.boundary_margin+v_len,
                args.boundary_margin:args.boundary_margin+h_len] = warp_img
        guide_on_ = False
        while True:
            if not guide_on_:
                img_zoom, _ = hp.imresize_full(out_img)
                disp_img = cv2.putText(img_zoom,
                                       "Press \'r\', \'f\', \'m\', \'y\', or \'s\'",
                                       (50, 60),
                                       CV2_FONT,
                                       1,
                                       hp.RED,
                                       4)
                cv2.imshow('warping', cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR))
            in_key = cv2.waitKey(1) & 0xFF
            if in_key == ord('y'):
                # ans = input(" % Enter the image filename : ")
                break
            elif in_key == ord('m'):
                out_img = cv2.flip(out_img, 1)
                guide_on_ = False
            elif in_key == ord('f'):
                out_img = cv2.flip(out_img, 0)
                guide_on_ = False
            elif in_key == ord('r'):
                out_img = cv2.transpose(out_img)
                out_img = cv2.flip(out_img, 0)
                guide_on_ = False
            elif in_key == ord('s'):
                continue
            else:
                pass
        cv2.destroyAllWindows()
        for i in range(10):
            cv2.waitKey(1)
        while True:
            core_fname = os.path.splitext(os.path.basename(img_filename))[0]
            out_img_fname = os.path.join(args.out_path, core_fname + "--crop.jpg")
            out_json_fname = os.path.join(args.out_path, core_fname + "--crop.json")
            out_info_dict = {'image_filename': img_filename,
                             'vertices': pts1.astype(np.int16).tolist(), }
            print(" # Default filename is {}".format(out_img_fname))
            ans = input(" % Enter the image filename (Enter to use the default filename) : ")
            ans = out_img_fname if ans == '' else ans
            if ans.split('.')[-1] in hp.IMG_EXTENSIONS:
                if os.path.isfile(out_img_fname):
                    print(" @ Warning: the same filename exists, {}.".format(out_img_fname))
                else:
                    hp.imwrite(out_img, out_img_fname)
                    with open(out_json_fname, 'w') as f:
                        json.dump(out_info_dict, f)
                    break
            else:
                print(" @ Error: Image file extension required, {}".format(ans.split('.')[-1]))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", required=True, help="image filename or directory")
    parser.add_argument("--boundary_margin", type=int, default=64, help="boundary margin")
    parser.add_argument("--out_path", default='.', help="output image path")
    parser.add_argument("--info_path", default='.', help="Vertices information path")
    return parser.parse_args(argv)


SELF_TEST_ = True
IMG_PATH = "DB_1.jpg"
RST_PATH = '.'


if __name__ == "__main__":

    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--img_path", IMG_PATH])
            sys.argv.extend(["--boundary_margin", '64'])
            sys.argv.extend(["--out_path", RST_PATH])
            sys.argv.extend(["--info_path", RST_PATH])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))
