# RKNN-Inswapper
Deploy faceswap(retinaface + arcface + inswapper) to RK3588S, optimized for rknpu.
1.Model conversion
(1) RetinaFace: Please refer to rknn model zoo
(2) arcface: Download from <https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip>
(3) Inswapper: Download from google :) Inswapper model's link from insightface is no longer available, so just google it.
Then use rknn-toolkit(2) to convert it. The model for RK3588S and RKNN-Toolkit-Lite 1.6.0 are alreay in the repository.
Performance:
(1) Set NPU frequency to maximum
(2) export RKNN_LOG_LEVEL=0 (optional)
(3) Run main.py. Every run costs ~1.8s
