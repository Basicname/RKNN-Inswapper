import argparse
import retinaface
import cv2
import inswapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RetinaFace Python Demo', add_help=True)
    parser.add_argument('--source', type=str, help='source image')
    parser.add_argument('--target', type=str, help='target image')
    parser.add_argument('--result', type=str, help='result image')
    args = parser.parse_args()
    source_img = cv2.imread(args.source)
    target_img = cv2.imread(args.target)
    source_ret = retinaface.get_faces(source_img)
    target_ret = retinaface.get_faces(target_img)
    assert len(source_ret) == 1
    assert len(target_ret) == 1
    face_fake = inswapper.get(target_img, target_ret[0]['face'], source_ret[0]['face'], target_ret[0]['M'])
    cv2.imwrite(args.result, face_fake)
