# %%
import copy
import os
import io

import PIL

import detectron2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# import some common libraries
import numpy as np
import torch

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image
NUM_OBJECTS = 36

# %%


def get_vg_classes(objects_path="data/1600-400-20/objects_vocab.txt"):
    vg_classes = []
    with open(objects_path) as f:
        for object in f.readlines():
            vg_classes.append(object.split(',')[0].lower().strip())
    return vg_classes


def build_predictor(config_path="data/faster_rcnn_R_101_C4_caffe.yaml"):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.PROPOSAL_GENERATOR.HID_CHANNELS = 512
    cfg.MODEL.ROI_BOX_HEAD.RES5HALVE = False
    cfg.merge_from_file(config_path)
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # VG Weight
    cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
    predictor = DefaultPredictor(cfg)
    return predictor


def doit(raw_image, predictor):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        print("Original image size: ", (raw_height, raw_width))

        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        print("Transformed image size: ", image.shape[:2])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        print('Pooled features size:', feature_pooled.shape)

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]

        # Note: BUTD uses raw RoI predictions,
        #       we use the predicted boxes instead.
        # boxes = proposal_boxes[0].tensor

        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:],
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS:
                break

        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        print(instances)

        return instances, roi_features


def normalize_pretrained_boxes(image, boxes, features, num_boxes=36):
    img_h, img_w = image.shape[:2]
    # boxes is num_boxes x 4
    # feathres is num_boxes x 2048
    while boxes.shape[0] < num_boxes:
        zeros = torch.zeros(1, boxes.shape[1])
        boxes = torch.cat((boxes, zeros), 0)
    while features.shape[0] < num_boxes:
        features = torch.cat((features, torch.zeros(1, features.shape[1])), 0)
    reshaped_boxes = boxes.reshape((num_boxes, -1))  # [36, 4]
    features = features.reshape((num_boxes, -1))  # [36, 2048]
    boxes = copy.deepcopy(reshaped_boxes)
    boxes[:, [0, 2]] /= int(img_w)
    boxes[:, [1, 3]] /= int(img_h)
    areas = (boxes[:, 2] - boxes[:, 0]) * \
        (boxes[:, 3] - boxes[:, 1])
    return np.c_[boxes, areas], features


def get_features_and_object_labels(images):
    vg_classes = get_vg_classes()
    predictor = build_predictor()

    full_class_labels = []
    normed_boxes = []
    normed_features = []
    for image in images:
        instances, features = doit(image, predictor)
        pred_classes = [item.item() for item in instances.pred_classes]
        classes_int = list(dict.fromkeys(pred_classes))
        class_labels = [vg_classes[i] for i in classes_int]
        full_class_labels.append(class_labels)
        boxes_normed, features_normed = normalize_pretrained_boxes(image, instances.pred_boxes.tensor, features)
        normed_boxes.append(boxes_normed)
        normed_features.append(features_normed)

    return normed_boxes, normed_features, full_class_labels


# %%
if __name__ == "__main__":
    image = PIL.Image.open("data/input.jpg")
    image = np.array(image)

    normed_boxes, normed_features, full_class_labels = get_features_and_object_labels([image])
    # print(features.shape)
    print(normed_boxes)
    print(normed_features)
    # print(instances.pred_classes)
    print(full_class_labels)
