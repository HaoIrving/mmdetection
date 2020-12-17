# adapted from https://github.com/Media-Smart/vedadet.git

import numpy as np
from numpy import random
from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class RandomSquareCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding boxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).

    Note:
        The keys for bboxes, labels and masks should be paired. That is, \
        `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and \
        `gt_bboxes_ignore` to `gt_labels_ignore` and `gt_masks_ignore`.
    """

    def __init__(self, crop_ratio_range=None, crop_choice=None):

        self.crop_ratio_range = crop_ratio_range
        self.crop_choice = crop_choice

        assert (self.crop_ratio_range is None) ^ (self.crop_choice is None)
        if self.crop_ratio_range is not None:
            self.crop_ratio_min, self.crop_ratio_max = self.crop_ratio_range

        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def __call__(self, results):
        """Call function to crop images and bounding boxes with minimum IoU
        constraint.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images and bounding boxes cropped, \
                'img_shape' key is updated.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert 'bbox_fields' in results
        boxes = [results[key] for key in results['bbox_fields']]
        boxes = np.concatenate(boxes, 0)
        h, w, c = img.shape

        while True:

            if self.crop_ratio_range is not None:
                scale = np.random.uniform(self.crop_ratio_min,
                                          self.crop_ratio_max)
            elif self.crop_choice is not None:
                scale = np.random.choice(self.crop_choice)

            # print(scale, img.shape[:2], boxes)
            # import cv2
            # cv2.imwrite('aaa.png', img)

            for i in range(250):
                short_side = min(w, h)
                cw = int(scale * short_side)
                ch = cw

                # TODO +1
                left = random.uniform(w - cw)
                top = random.uniform(h - ch)

                patch = np.array(
                    (int(left), int(top), int(left + cw), int(top + ch)))

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                # adjust boxes
                def is_center_of_bboxes_in_patch(boxes, patch):
                    # TODO >=
                    center = (boxes[:, :2] + boxes[:, 2:]) / 2
                    mask = ((center[:, 0] > patch[0]) *
                            (center[:, 1] > patch[1]) *
                            (center[:, 0] < patch[2]) *
                            (center[:, 1] < patch[3]))
                    return mask

                mask = is_center_of_bboxes_in_patch(boxes, patch)
                if not mask.any():
                    continue
                for key in results.get('bbox_fields', []):
                    boxes = results[key].copy()
                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    boxes = boxes[mask]
                    boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                    boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                    boxes -= np.tile(patch[:2], 2)

                    results[key] = boxes
                    # labels
                    label_key = self.bbox2label.get(key)
                    if label_key in results:
                        results[label_key] = results[label_key][mask]

                    # mask fields
                    mask_key = self.bbox2mask.get(key)
                    if mask_key in results:
                        results[mask_key] = results[mask_key][mask.nonzero()
                                                              [0]].crop(patch)

                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                results['img'] = img
                results['img_shape'] = img.shape

                # seg fields
                for key in results.get('seg_fields', []):
                    results[key] = results[key][patch[1]:patch[3],
                                                patch[0]:patch[2]]
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_ious={self.min_iou}, '
        repr_str += f'crop_size={self.crop_size})'
        return repr_str