import onnxruntime
import cv2
import numpy as np
from numpy.linalg import norm as l2norm
from rknnlite.api import RKNNLite

rknn = RKNNLite()
input_size = (112, 112)

def get_feat(img):
    input_mean = 127.5
    input_std = 127.5
    blob = cv2.dnn.blobFromImages([img], 1.0 / input_std, input_size,(input_mean, input_mean, input_mean), swapRB=True)
    blob = blob.transpose(0,2,3,1)
    rknn.load_rknn('rec.rknn')
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    net_out = rknn.inference(inputs=[blob])[0]
    rknn.release()
    return net_out

def get_emb(img):
    embedding = get_feat(img).flatten()
    embedding_norm = l2norm(embedding)
    source_face_emb = embedding / embedding_norm
    source_face_emb = np.expand_dims(source_face_emb, axis=0)
    return source_face_emb

