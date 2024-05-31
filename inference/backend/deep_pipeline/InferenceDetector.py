# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import mmcv
import mmengine
import numpy as np
import torch.nn as nn
from mmcv.transforms import LoadImageFromFile
from mmengine.config import ConfigDict
from mmengine.dataset import Compose
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file)
from mmengine.infer.infer import BaseInferencer, ModelType
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import _load_checkpoint_to_model
from rich.progress import track
from utils.structures import DetDataSample

InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = List[DetDataSample]
ConfigType = Union[ConfigDict, dict]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


class DetInferencer(BaseInferencer):
    """Object Detection Inferencer.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "rtmdet-s" or 'rtmdet_s_8xb32-300e_coco' or
            "configs/rtmdet/rtmdet_s_8xb32-300e_coco.py".
            If model is not specified, user must provide the
            `weights` saved by MMEngine which contains the config string.
            Defaults to None.
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        scope (str, optional): The scope of the model. Defaults to mmdet.
        show_progress (bool): Control whether to display the progress
            bar during the inference process. Defaults to True.
    """

    preprocess_kwargs: set = set()
    forward_kwargs: set = set()

    def __init__(
        self,
        model: Optional[Union[ModelType, str]] = None,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        scope: Optional[str] = "mmdet",
    ) -> None:
        # A global counter tracking the number of images processed, for
        # naming of the output images
        self.num_predicted_imgs = 0
        init_default_scope(scope)
        super().__init__(model=model, weights=weights, device=device, scope=scope)
        self.model = revert_sync_batchnorm(self.model)
        self.show_progress = True

    def _load_weights_to_model(
        self, model: nn.Module, checkpoint: Optional[dict], cfg: Optional[ConfigType]
    ) -> None:
        """Loading model weights and meta information from cfg and checkpoint.

        Args:
            model (nn.Module): Model to load weights and meta information.
            checkpoint (dict, optional): The loaded checkpoint.
            cfg (Config or ConfigDict, optional): The loaded config.
        """

        if checkpoint is not None:
            _load_checkpoint_to_model(model, checkpoint)
            checkpoint_meta = checkpoint.get("meta", {})
            # save the dataset_meta in the model for convenience
            if "dataset_meta" in checkpoint_meta:
                # mmdet 3.x, all keys should be lowercase
                model.dataset_meta = {
                    k.lower(): v for k, v in checkpoint_meta["dataset_meta"].items()
                }
            elif "CLASSES" in checkpoint_meta:
                # < mmdet 3.x
                classes = checkpoint_meta["CLASSES"]
                model.dataset_meta = {"classes": classes}
            else:
                model.dataset_meta = {"classes": ["tamgi"]}
        else:
            model.dataset_meta = {"classes": get_classes("coco")}

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        # For inference, the key of ``img_id`` is not used.
        if "meta_keys" in pipeline_cfg[-1]:
            pipeline_cfg[-1]["meta_keys"] = tuple(
                meta_key
                for meta_key in pipeline_cfg[-1]["meta_keys"]
                if meta_key != "img_id"
            )

        load_img_idx = self._get_transform_idx(
            pipeline_cfg, ("LoadImageFromFile", LoadImageFromFile)
        )
        if load_img_idx == -1:
            raise ValueError("LoadImageFromFile is not found in the test pipeline")
        pipeline_cfg[load_img_idx]["type"] = "mmdet.InferencerLoader"
        return Compose(pipeline_cfg)

    def _get_transform_idx(
        self, pipeline_cfg: ConfigType, name: Union[str, Tuple[str, type]]
    ) -> int:
        """Returns the index of the transform in a pipeline.

        If the transform is not found, returns -1.
        """
        for i, transform in enumerate(pipeline_cfg):
            if transform["type"] in name:
                return i
        return -1

    def _inputs_to_list(self, inputs: InputsType) -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - str:
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.

        Args:
            inputs (InputsType): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        if isinstance(inputs, str):
            backend = get_file_backend(inputs)
            if hasattr(backend, "isdir") and isdir(inputs):
                # Backends like HttpsBackend do not implement `isdir`, so only
                # those backends that implement `isdir` could accept the inputs
                # as a directory
                filename_list = list_dir_or_file(
                    inputs, list_dir=False, suffix=IMG_EXTENSIONS
                )
                inputs = [join_path(inputs, filename) for filename in filename_list]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

    def preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs):
        """Process the inputs into a model-feedable format.

        Customize your preprocess by overriding this method. Preprocess should
        return an iterable object, of which each item will be used as the
        input of ``model.test_step``.

        ``BaseInferencer.preprocess`` will return an iterable chunked data,
        which will be used in __call__ like this:

        .. code-block:: python

            def __call__(self, inputs, batch_size=1, **kwargs):
                chunked_data = self.preprocess(inputs, batch_size, **kwargs)
                for batch in chunked_data:
                    preds = self.forward(batch, **kwargs)

        Args:
            inputs (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1.

        Yields:
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
        """
        chunked_data = self._get_chunk_data(inputs, batch_size)
        yield from map(self.collate_fn, chunked_data)

    def _get_chunk_data(self, inputs: Iterable, chunk_size: int):
        """Get batch data from inputs.

        Args:
            inputs (Iterable): An iterable dataset.
            chunk_size (int): Equivalent to batch size.

        Yields:
            list: batch data.
        """
        inputs_iter = iter(inputs)
        while True:
            try:
                chunk_data = []
                for _ in range(chunk_size):
                    inputs_ = next(inputs_iter)
                    if isinstance(inputs_, dict):
                        if "img" in inputs_:
                            ori_inputs_ = inputs_["img"]
                        else:
                            ori_inputs_ = inputs_["img_path"]
                        chunk_data.append(
                            (ori_inputs_, self.pipeline(copy.deepcopy(inputs_)))
                        )
                    else:
                        chunk_data.append((inputs_, self.pipeline(inputs_)))
                yield chunk_data
            except StopIteration:
                if chunk_data:
                    yield chunk_data
                break

    def __call__(
        self,
        inputs: InputsType,
        batch_size: int = 1,
        pred_score_thr: float = 0.3,
        return_datasamples: bool = False,
        no_save_pred: bool = True,
        out_dir: str = "",
        **kwargs,
    ) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            batch_size (int): Inference batch size. Defaults to 1.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            return_datasamples (bool): Whether to return results as
                :obj:`DetDataSample`. Defaults to False.
            no_save_pred (bool): Whether to force not to save prediction
                results. Defaults to True.
            out_dir: Dir to save the inference results.
            If left as empty, no file will be saved.
                Defaults to ''.
            **kwargs: Other keyword arguments passed to :meth:`preprocess`,
                :meth:`forward`, and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``,
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference results.
        """
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        ori_inputs = self._inputs_to_list(inputs)

        inputs = self.preprocess(ori_inputs, batch_size=batch_size, **preprocess_kwargs)

        results_dict = {"predictions": []}
        for ori_imgs, data in (
            track(inputs, description="Inference") if self.show_progress else inputs
        ):
            preds = self.forward(data, **forward_kwargs)
            results = self.postprocess(
                preds,
                return_datasamples=return_datasamples,
                no_save_pred=no_save_pred,
                pred_out_dir=out_dir,
                **postprocess_kwargs,
            )
            results_dict["predictions"].extend(results["predictions"])
        return results_dict

    def postprocess(
        self,
        preds: PredType,
        return_datasamples: bool = False,
        no_save_pred: bool = False,
        pred_out_dir: str = "",
        **kwargs,
    ) -> Dict:
        """Process the predictions from ``forward``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions results and return them.
        3. Dump or log the predictions.

        Args:
            preds (List[:obj:`DetDataSample`]): Predictions of the model.
            return_datasamples (bool): Whether to use Datasample to store
                inference results. If False, dict will be used.
            no_save_pred (bool): Whether to force not to save prediction
                results. Defaults to False.
            pred_out_dir: Dir to save the inference results. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Inference results with key ``predictions``.

            - ``predictions`` (dict or DataSample): Returned by
                :meth:`forward` and processed in :meth:`postprocess`.
                If ``return_datasamples=False``, it usually should be a
                json-serializable dict containing only basic data elements such
                as strings and numbers.
        """
        if no_save_pred is True:
            pred_out_dir = ""

        result_dict = {}
        results = preds
        if not return_datasamples:
            results = []
            for pred in preds:
                result = self.pred2dict(pred, pred_out_dir)
                results.append(result)
        elif pred_out_dir != "":
            warnings.warn(
                "Currently does not support saving datasample "
                "when return_datasamples is set to True. "
                "Prediction results are not saved!"
            )
        # Add img to the results after printing and dumping
        result_dict["predictions"] = results

        return result_dict

    def pred2dict(self, data_sample: DetDataSample, pred_out_dir: str = "") -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (:obj:`DetDataSample`): Predictions of the model.
            pred_out_dir: Dir to save the inference results.
            If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Prediction results.
        """
        is_save_pred = True
        if pred_out_dir == "":
            is_save_pred = False

        if is_save_pred and "img_path" in data_sample:
            img_path = osp.basename(data_sample.img_path)
            img_path = osp.splitext(img_path)[0]

            out_json_path = osp.join(pred_out_dir, "preds", img_path + ".json")
        elif is_save_pred:
            out_json_path = osp.join(
                pred_out_dir, "preds", f"{self.num_predicted_imgs}.json"
            )
            self.num_predicted_imgs += 1

        result = {}
        if "pred_instances" in data_sample:
            pred_instances = data_sample.pred_instances.numpy()
            result = {
                "scores": pred_instances.scores.tolist(),
            }
            if "bboxes" in pred_instances:
                result["bboxes"] = pred_instances.bboxes.tolist()

        if is_save_pred:
            mmengine.dump(result, out_json_path)

        return result

    def visualize(
        self, inputs: list, preds: Any, show: bool = False, **kwargs
    ) -> List[np.ndarray]: ...


import os
import time

import cv2
from tqdm import tqdm

if __name__ == "__main__":
    det_enferencer = DetInferencer(
        "/home/ana/University/Tamgi/src/mmdetection/work_dirs/retinanet_r50_fpn_tamgi/retinanet_r50_fpn_tamgi.py",
        "/home/ana/University/Tamgi/src/mmdetection/work_dirs/retinanet_r50_fpn_tamgi/best_coco_bbox_mAP_epoch_70.pth",
        "cuda",
    )
    input_dir = "/home/ana/University/Tamgi/data/dataset/render"
    output_dir = "/home/ana/University/Tamgi/data/dataset/crop"
    start = time.time()
    cnt = 0
    for ddir in tqdm(os.listdir(input_dir)):
        for img in os.listdir(os.path.join(input_dir, ddir)):
            try:
                det = det_enferencer(os.path.join(input_dir, ddir, img))["predictions"][
                    0
                ]["bboxes"][0]
                cnt += 1
                det = list(map(int, det))
                image = cv2.imread(os.path.join(input_dir, ddir, img))
                crop = image[det[1] : det[3], det[0] : det[2]]
                os.makedirs(os.path.join(output_dir, ddir), exist_ok=True)
                cv2.imwrite(os.path.join(output_dir, ddir, img), crop)
                image = cv2.imread(os.path.join(input_dir, ddir, img))
                crop = image[det[1] : det[3], det[0] : det[2]]
                os.makedirs(os.path.join(output_dir, ddir), exist_ok=True)
                cv2.imwrite(os.path.join(output_dir, ddir, img), crop)
            except:
                ...

    print("average time", (time.time() - start) / 1000 / cnt)
    print(cnt)
