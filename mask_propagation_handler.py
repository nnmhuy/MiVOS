"""
ModelHandler defines a custom model handler.
"""

import zipfile
from ts.torch_handler.base_handler import BaseHandler

import os
import io
from os import path
import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2
import base64
from PIL import Image
import time
import logging


logger = logging.getLogger(__name__)


class MaskPropagationHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.prop_net = None
        self.device = None
        self.MIN_SIDE = 480
        self.INPUT_WIDTH = 768
        self.INPUT_HEIGHT = 480


    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """

        self._context = context
        self.manifest = context.manifest


        # unzip package
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        if not torch.cuda.is_available() or properties.get("gpu_id") is None :
            raise RuntimeError("This model is not supported on CPU machines.")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")))

        with zipfile.ZipFile(model_dir + '/mask-propagation-package.zip', 'r') as zip_ref:
            zip_ref.extractall(model_dir)
          

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = model_dir + "/smart-annotation-platform-credential.json"


        # load models
        # load stcn mask propagation model
        from model.propagation.prop_net import PropagationNetwork

        prop_model_pth_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            prop_model_pth_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(prop_model_pth_path):
            raise RuntimeError("Missing the stcn.pth file")

        prop_net = PropagationNetwork()
        prop_net.load_state_dict(torch.load(
            prop_model_pth_path, map_location=self.device))
        prop_net = prop_net.to(self.device).eval()
        torch.set_grad_enabled(False)

        self.prop_net = prop_net

        # load stcn mask fusion model
        from model.fusion_net import FusionNet

        fuse_model_pth_path = model_dir + "saves/fusion_stcn.pth"
        if not os.path.isfile(fuse_model_pth_path):
            raise RuntimeError("Missing the fusion_stcn.pth file")

        fuse_net = FusionNet()
        fuse_net.load_state_dict(torch.load(fuse_model_pth_path, map_location=self.device))
        fuse_net = fuse_net.to(self.device).eval()
        torch.set_grad_enabled(False)

        self.fuse_net = fuse_net

        # TODO: with torch.cuda.amp.autocast(enabled=not args.no_amp)?

        self.initialized = True


    # def _parse_request(self, data):
    #     annotation_object_id = data.get("annotation_object_id")
    #     dataset_id = data.get("dataset_id")
    #     video_name = data.get("video_name")
    #     key_frames = data.get("key_frames")
    #     propagating_frames = data.get("propagating_frames")
    #     frame_frequency = data.get("frame_frequency")

    #     return data

    def _load_images(self, dataset_id, video_name, frames):
        """
        Load all images required in propagation processing including key and propagating frames
        :param dataset_id
        :param video_name
        :param frames: array of frame indexes
        :return: dict of images by indexes
        """
        from util.gcloud_util import upload_to_google_storage, download_image_from_google_storage

        root_dir = 'dataset/{}/frames/{}/'.format(dataset_id, video_name)
        images = [download_image_from_google_storage(root_dir + '%05d.jpg' % idx) for idx in frames]

        return images

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        start_time = time.time()

        metrics = self._context.metrics

        model_input, pad, output_path = self.preprocess(data)
        model_output = self.inference(model_input)
        final_output =  self.postprocess(model_output, pad, output_path)
        # 1. Get key frames index, propagating frame indexes from request
        # 2. Load images, masks of key frame, propagating frames
        # 3. Pre-processing: images and masks to model input (padding, ...)
        # 4. Run models (with fusion or not)
        # 5. Post-processing: model output to image data, save to cloud
        # 6. Return propagated mask URLs

        stop_time = time.time()
        metrics.add_time('HandlerTime', round((stop_time - start_time) * 1000, 2), None, 'ms')

        return final_output
