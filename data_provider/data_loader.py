"""
Dataset loader and utilities for battery life prediction.

This module provides dataset classes and collate functions for loading and processing
battery charge/discharge cycle data from multiple datasets for training and evaluation.
"""

import os
import copy
import json
import pickle
import shutil
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm

from denseweight import DenseWeight
from data_provider.data_split_recorder import split_recorder
from utils.augmentation import BatchAugmentation_battery_revised


# Mapping of dataset names to unique integer IDs
datasetName2ids: dict[str, int] = {
    "CALCE": 0,
    "HNEI": 1,
    "HUST": 2,
    "MATR": 3,
    "RWTH": 4,
    "SNL": 5,
    "MICH": 6,
    "MICH_EXP": 7,
    "Tongji1": 8,
    "Stanford": 9,
    "ISU-ILCC": 11,
    "XJTU": 12,
    "ZN-coin": 13,
    "UL-PUR": 14,
    "Tongji2": 15,
    "Tongji3": 16,
    "CALB": 17,
    "ZN42": 22,
    "ZN2024": 23,
    "CALB42": 24,
    "CALB2024": 25,
    "NA-ion": 27,
    "NA-ion42": 28,
    "NA-ion2024": 29,
}


def my_collate_fn_withId(
    samples: list[dict[str, Any]]
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Collate function with dataset IDs for evaluation mode.
    
    Batches samples together and includes dataset IDs for tracking which dataset
    each sample comes from during evaluation.
    
    Args:
        samples: List of sample dictionaries containing cycle data, masks, labels, etc.
        
    Returns:
        tuple: (
            cycle_curve_data: Battery charge/discharge curves [B, seq_len, num_vars, curve_len],
            curve_attn_mask: Attention mask for valid cycles [B, seq_len],
            labels: Target labels (remaining useful life) [B],
            life_class: Life class labels [B],
            scaled_life_class: Scaled life class labels [B],
            weights: Sample weights for loss [B],
            dataset_ids: Dataset identifier for each sample [B],
            seen_unseen_ids: Whether sample is from seen/unseen data [B]
        )
    """
    cycle_curve_data = torch.vstack(
        [i["cycle_curve_data"].unsqueeze(0) for i in samples]
    )
    curve_attn_mask = torch.vstack([i["curve_attn_mask"].unsqueeze(0) for i in samples])
    life_class = torch.Tensor([i["life_class"] for i in samples])
    labels = torch.Tensor([i["labels"] for i in samples])
    scaled_life_class = torch.Tensor([i["scaled_life_class"] for i in samples])
    weights = torch.Tensor([i["weight"] for i in samples])
    dataset_ids = torch.Tensor([i["dataset_id"] for i in samples])
    seen_unseen_ids = torch.Tensor([i["seen_unseen_id"] for i in samples])

    # Mask out unseen cycles by setting them to zero
    tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(
        cycle_curve_data
    )
    cycle_curve_data[tmp_curve_attn_mask == 0] = 0
    
    return (
        cycle_curve_data,
        curve_attn_mask,
        labels,
        life_class,
        scaled_life_class,
        weights,
        dataset_ids,
        seen_unseen_ids,
    )


def my_collate_fn_baseline(
    samples: list[dict[str, Any]]
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Baseline collate function for training and standard evaluation.
    
    Batches samples together without dataset IDs. Used for standard training
    and validation where dataset tracking is not required.
    
    Args:
        samples: List of sample dictionaries containing cycle data, masks, labels, etc.
        
    Returns:
        tuple: (
            cycle_curve_data: Battery charge/discharge curves [B, seq_len, num_vars, curve_len],
            curve_attn_mask: Attention mask for valid cycles [B, seq_len],
            labels: Target labels (remaining useful life) [B],
            life_class: Life class labels [B],
            scaled_life_class: Scaled life class labels [B],
            weights: Sample weights for loss [B],
            seen_unseen_ids: Whether sample is from seen/unseen data [B]
        )
    """
    cycle_curve_data = torch.vstack(
        [i["cycle_curve_data"].unsqueeze(0) for i in samples]
    )
    curve_attn_mask = torch.vstack([i["curve_attn_mask"].unsqueeze(0) for i in samples])
    life_class = torch.Tensor([i["life_class"] for i in samples])
    labels = torch.Tensor(np.array([i["labels"] for i in samples]))
    scaled_life_class = torch.Tensor([i["scaled_life_class"] for i in samples])
    weights = torch.Tensor([i["weight"] for i in samples])
    seen_unseen_ids = torch.Tensor([i["seen_unseen_id"] for i in samples])

    # Mask out unseen cycles by setting them to zero
    tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(
        cycle_curve_data
    )
    cycle_curve_data[tmp_curve_attn_mask == 0] = 0
    
    return (
        cycle_curve_data,
        curve_attn_mask,
        labels,
        life_class,
        scaled_life_class,
        weights,
        seen_unseen_ids,
    )


class Dataset_original(Dataset):
    """
    Dataset class for loading battery life prediction data.
    
    Loads and preprocesses battery charge/discharge cycle data from multiple datasets
    for training, validation, and testing. Supports data augmentation, weighted sampling,
    and seen/unseen data tracking.
    
    Attributes:
        life_classes: Mapping of battery life to class labels
        args: Configuration arguments
        flag: Dataset split ('train', 'val', or 'test')
        dataset: Name of the dataset to load
        total_charge_discharge_curves: All charge/discharge curves
        total_labels: All labels (remaining useful life)
        label_scaler: StandardScaler for normalizing labels
    """
    
    def __init__(
        self,
        args: Any,
        flag: str = "train",
        label_scaler: StandardScaler | None = None,
        eval_cycle_max: int | None = None,
        eval_cycle_min: int | None = None,
        life_class_scaler: StandardScaler | None = None,
        use_target_dataset: bool = False,
    ) -> None:
        """
        Initialize the Dataset_original class.
        
        Args:
            args: Model configuration parameters
            flag: Dataset split - 'train', 'val', or 'test'
            label_scaler: Pre-fitted scaler for labels (None for training)
            eval_cycle_max: Maximum evaluation cycle
            eval_cycle_min: Minimum evaluation cycle
            life_class_scaler: Pre-fitted scaler for life classes (None for training)
            use_target_dataset: Whether to use target dataset for domain adaptation
        """
        self.life_classes = json.load(open("data_provider/life_classes.json"))
        self.eval_cycle_max = eval_cycle_max
        self.eval_cycle_min = eval_cycle_min
        self.args = args
        self.root_path = args.root_path
        self.seq_len = args.seq_len
        self.charge_discharge_len = args.charge_discharge_length
        self.flag = flag
        self.dataset = args.target_dataset if use_target_dataset else args.dataset
        self.early_cycle_threshold = args.early_cycle_threshold
        self.KDE_samples = []

        self.need_keys = [
            "current_in_A",
            "voltage_in_V",
            "charge_capacity_in_Ah",
            "discharge_capacity_in_Ah",
            "time_in_s",
        ]
        self.aug_helper = BatchAugmentation_battery_revised()
        assert flag in ["train", "test", "val"]
        if self.dataset == "exp":
            self.train_files = split_recorder.Stanford_train_files[:3]
            self.val_files = (
                split_recorder.Tongji_val_files[:2] + split_recorder.HUST_val_files[:2]
            )
            self.test_files = (
                split_recorder.Tongji_test_files[:2]
                + split_recorder.HUST_test_files[:2]
            )
        elif self.dataset == "Tongji":
            self.train_files = split_recorder.Tongji_train_files
            self.val_files = split_recorder.Tongji_val_files
            self.test_files = split_recorder.Tongji_test_files
        elif self.dataset == "HUST":
            self.train_files = split_recorder.HUST_train_files
            self.val_files = split_recorder.HUST_val_files
            self.test_files = split_recorder.HUST_test_files
        elif self.dataset == "MATR":
            self.train_files = split_recorder.MATR_train_files
            self.val_files = split_recorder.MATR_val_files
            self.test_files = split_recorder.MATR_test_files
        elif self.dataset == "SNL":
            self.train_files = split_recorder.SNL_train_files
            self.val_files = split_recorder.SNL_val_files
            self.test_files = split_recorder.SNL_test_files
        elif self.dataset == "MICH":
            self.train_files = split_recorder.MICH_train_files
            self.val_files = split_recorder.MICH_val_files
            self.test_files = split_recorder.MICH_test_files
        elif self.dataset == "MICH_EXP":
            self.train_files = split_recorder.MICH_EXP_train_files
            self.val_files = split_recorder.MICH_EXP_val_files
            self.test_files = split_recorder.MICH_EXP_test_files
        elif self.dataset == "UL_PUR":
            self.train_files = split_recorder.UL_PUR_train_files
            self.val_files = split_recorder.UL_PUR_val_files
            self.test_files = split_recorder.UL_PUR_test_files
        elif self.dataset == "RWTH":
            self.train_files = split_recorder.RWTH_train_files
            self.val_files = split_recorder.RWTH_val_files
            self.test_files = split_recorder.RWTH_test_files
        elif self.dataset == "HNEI":
            self.train_files = split_recorder.HNEI_train_files
            self.val_files = split_recorder.HNEI_val_files
            self.test_files = split_recorder.HNEI_test_files
        elif self.dataset == "CALCE":
            self.train_files = split_recorder.CALCE_train_files
            self.val_files = split_recorder.CALCE_val_files
            self.test_files = split_recorder.CALCE_test_files
        elif self.dataset == "Stanford":
            self.train_files = split_recorder.Stanford_train_files
            self.val_files = split_recorder.Stanford_val_files
            self.test_files = split_recorder.Stanford_test_files
        elif self.dataset == "ISU_ILCC":
            self.train_files = split_recorder.ISU_ILCC_train_files
            self.val_files = split_recorder.ISU_ILCC_val_files
            self.test_files = split_recorder.ISU_ILCC_test_files
        elif self.dataset == "XJTU":
            self.train_files = split_recorder.XJTU_train_files
            self.val_files = split_recorder.XJTU_val_files
            self.test_files = split_recorder.XJTU_test_files
        elif self.dataset == "MIX_large":
            self.train_files = split_recorder.MIX_large_train_files
            self.val_files = split_recorder.MIX_large_val_files
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == "ZN-coin":
            self.train_files = split_recorder.ZNcoin_train_files
            self.val_files = split_recorder.ZNcoin_val_files
            self.test_files = split_recorder.ZNcoin_test_files
        elif self.dataset == "CALB":
            self.train_files = split_recorder.CALB_train_files
            self.val_files = split_recorder.CALB_val_files
            self.test_files = split_recorder.CALB_test_files
        elif self.dataset == "ZN-coin42":
            self.train_files = split_recorder.ZN_42_train_files
            self.val_files = split_recorder.ZN_42_val_files
            self.test_files = split_recorder.ZN_42_test_files
        elif self.dataset == "ZN-coin2024":
            self.train_files = split_recorder.ZN_2024_train_files
            self.val_files = split_recorder.ZN_2024_val_files
            self.test_files = split_recorder.ZN_2024_test_files
        elif self.dataset == "CALB42":
            self.train_files = split_recorder.CALB_42_train_files
            self.val_files = split_recorder.CALB_42_val_files
            self.test_files = split_recorder.CALB_42_test_files
        elif self.dataset == "CALB2024":
            self.train_files = split_recorder.CALB_2024_train_files
            self.val_files = split_recorder.CALB_2024_val_files
            self.test_files = split_recorder.CALB_2024_test_files
        elif self.dataset == "NAion":
            self.train_files = split_recorder.NAion_2021_train_files
            self.val_files = split_recorder.NAion_2021_val_files
            self.test_files = split_recorder.NAion_2021_test_files
        elif self.dataset == "NAion42":
            self.train_files = split_recorder.NAion_42_train_files
            self.val_files = split_recorder.NAion_42_val_files
            self.test_files = split_recorder.NAion_42_test_files
        elif self.dataset == "NAion2024":
            self.train_files = split_recorder.NAion_2024_train_files
            self.val_files = split_recorder.NAion_2024_val_files
            self.test_files = split_recorder.NAion_2024_test_files
        else:
            raise Exception(f"Dataset is not recognized. Got {self.dataset}")

        if flag == "train":
            self.files = [i for i in self.train_files]
        elif flag == "val":
            self.files = [i for i in self.val_files]
        elif flag == "test":
            self.files = [i for i in self.test_files]
            if self.dataset == "ZN-coin42":
                self.unseen_seen_record = json.load(
                    open(f"{self.root_path}/seen_unseen_labels/cal_for_test_ZN42.json")
                )
            elif self.dataset == "ZN-coin2024":
                self.unseen_seen_record = json.load(
                    open(
                        f"{self.root_path}/seen_unseen_labels/cal_for_test_ZN2024.json"
                    )
                )
            elif self.dataset == "CALB42":
                self.unseen_seen_record = json.load(
                    open(
                        f"{self.root_path}/seen_unseen_labels/cal_for_test_CALB42.json"
                    )
                )
            elif self.dataset == "CALB2024":
                self.unseen_seen_record = json.load(
                    open(
                        f"{self.root_path}/seen_unseen_labels/cal_for_test_CALB2024.json"
                    )
                )
            elif self.dataset == "NAion":
                self.unseen_seen_record = json.load(
                    open(
                        f"{self.root_path}/seen_unseen_labels/cal_for_test_NA2021.json"
                    )
                )
            elif self.dataset == "NAion42":
                self.unseen_seen_record = json.load(
                    open(f"{self.root_path}/seen_unseen_labels/cal_for_test_NA42.json")
                )
            elif self.dataset == "NAion2024":
                self.unseen_seen_record = json.load(
                    open(
                        f"{self.root_path}/seen_unseen_labels/cal_for_test_NA2024.json"
                    )
                )
            else:
                self.unseen_seen_record = json.load(
                    open(f"{self.root_path}/seen_unseen_labels/cal_for_test.json")
                )
            # self.unseen_seen_record = json.load(open(f'{self.root_path}/cal_for_test.json'))

        (
            self.total_charge_discharge_curves,
            self.total_curve_attn_masks,
            self.total_labels,
            self.unique_labels,
            self.class_labels,
            self.total_dataset_ids,
            self.total_cj_aug_charge_discharge_curves,
            self.total_seen_unseen_IDs,
        ) = self.read_data()

        self.KDE_samples = copy.deepcopy(self.total_labels) if flag == "train" else []

        self.weights = self.get_loss_weight()
        if np.any(np.isnan(self.total_charge_discharge_curves)):
            raise Exception("Nan in the data")
        if np.any(np.isnan(self.unique_labels)):
            raise Exception("Nan in the labels")
        # K-means to classify the battery life

        self.raw_labels = copy.deepcopy(self.total_labels)
        if flag == "train" and label_scaler is None:
            self.label_scaler = StandardScaler()
            self.life_class_scaler = StandardScaler()
            self.label_scaler.fit(np.array(self.unique_labels).reshape(-1, 1))
            self.life_class_scaler.fit(np.array(self.class_labels).reshape(-1, 1))
            self.total_labels = self.label_scaler.transform(
                np.array(self.total_labels).reshape(-1, 1)
            )
            self.scaled_life_classes = np.array(self.class_labels) - 1
            # self.scaled_life_classes = self.life_class_scaler.transform(np.array(self.class_labels).reshape(-1, 1))
        else:
            # validation set or testing set
            assert label_scaler is not None
            self.label_scaler = label_scaler
            self.life_class_scaler = life_class_scaler
            self.total_labels = self.label_scaler.transform(
                np.array(self.total_labels).reshape(-1, 1)
            )
            self.scaled_life_classes = np.array(self.class_labels) - 1
            # self.scaled_life_classes = self.life_class_scaler.transform(np.array(self.class_labels).reshape(-1,1))

    def get_loss_weight(self, method: str = "KDE") -> NDArray[np.floating]:
        """
        Calculate sample weights for weighted loss.
        
        Computes weights based on label distribution to handle class imbalance.
        Supports three methods: '1/n' (inverse frequency), '1/log(x+1)' (log-scaled),
        and 'KDE' (kernel density estimation).
        
        Args:
            method: Weighting method - '1/n', '1/log(x+1)', or 'KDE'
            
        Returns:
            Array of weights for each sample
        """
        if self.args.weighted_loss and self.flag == "train":
            if method == "1/n":
                indices = list(range(self.__len__()))
                df = pd.DataFrame()
                df["label"] = self.total_dataset_ids
                df.index = indices
                df = df.sort_index()

                label_to_count = df["label"].value_counts()

                weights = 1.0 / label_to_count[df["label"]].values
            elif method == "1/log(x+1)":
                indices = list(range(self.__len__()))
                df = pd.DataFrame()
                df["label"] = self.total_dataset_ids
                df.index = indices
                df = df.sort_index()

                label_to_count = df["label"].value_counts()

                x = label_to_count[df["label"]].values
                normalized_x = np.log(x / np.min(x) + 1)
                weights = 1 / normalized_x
            elif method == "KDE":
                # Define DenseWeight
                dw = DenseWeight(alpha=1.0)
                # Fit DenseWeight and get the weights for the 1000 samples
                dw.fit(self.KDE_samples)
                # Calculate the weight for an arbitrary target value
                weights = []
                for label in self.KDE_samples:
                    single_sample_weight = dw(label)[0]
                    weights.append(single_sample_weight)
            else:
                raise Exception("Not implemented")
            return weights
        else:
            return np.ones(len(self.total_charge_discharge_curves))

    def get_center_vector_index(self, file_name: str) -> int:
        """
        Get the center vector index based on dataset type.
        
        Args:
            file_name: Name of the data file
            
        Returns:
            Index (0 for MATR/HUST/LFP datasets, 1 for others)
        """
        prefix = file_name.split("_")[0]
        if prefix in ["MATR", "HUST"] or "LFP" in file_name:
            return 0
        else:
            return 1

    def return_label_scaler(self) -> StandardScaler:
        """Return the fitted label scaler."""
        return self.label_scaler

    def return_life_class_scaler(self) -> StandardScaler:
        """Return the fitted life class scaler."""
        return self.life_class_scaler

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.total_labels)

    def read_data(self) -> tuple[
        list[Any],
        list[Any],
        NDArray[np.floating],
        list[float],
        list[int],
        list[int],
        list[Any],
        list[int],
    ]:
        """
        Read all data from files.
        
        Loads battery cycle data from all files in the dataset split,
        extracts features and labels, and tracks metadata.
        
        Returns:
            tuple: (
                total_charge_discharge_curves: List of charge/discharge curves,
                total_curve_attn_masks: List of attention masks,
                total_labels: Array of labels (remaining useful life),
                unique_labels: List of unique labels per battery,
                class_labels: List of life class labels,
                total_dataset_ids: List of dataset IDs,
                total_cj_aug_charge_discharge_curves: List of augmented curves,
                total_seen_unseen_IDs: List of seen/unseen indicators
            )
        """

        total_charge_discharge_curves = []
        total_curve_attn_masks = []
        total_labels = []  # RUL
        unique_labels = []
        class_labels = []  # the pseudo class for samples
        total_dataset_ids = []
        total_cj_aug_charge_discharge_curves = []
        total_seen_unseen_IDs = []

        for file_name in tqdm(self.files):
            if (
                file_name not in split_recorder.MICH_EXP_test_files
                and file_name not in split_recorder.MICH_EXP_train_files
                and file_name not in split_recorder.MICH_EXP_val_files
            ):
                dataset_id = datasetName2ids[file_name.split("_")[0]]
            else:
                dataset_id = datasetName2ids["MICH_EXP"]

            (
                charge_discharge_curves,
                attn_masks,
                labels,
                eol,
                cj_aug_charge_discharge_curves,
            ) = self.read_samples_from_one_cell(file_name)
            if eol is None:
                # This battery has not reached end of life
                continue

            for class_label, life_range in self.life_classes.items():
                if eol >= life_range[0] and eol < life_range[1]:
                    class_label = int(class_label)
                    class_labels += [
                        class_label for _ in range(len(charge_discharge_curves))
                    ]
                    break

            total_charge_discharge_curves += charge_discharge_curves
            total_cj_aug_charge_discharge_curves += cj_aug_charge_discharge_curves
            total_curve_attn_masks += attn_masks
            total_labels += labels
            total_dataset_ids += [dataset_id for _ in range(len(labels))]
            # total_center_vector_indices += [center_vector_index for _ in range(len(labels))]
            unique_labels.append(eol)
            if self.flag == "test":
                seen_unseen_id = self.unseen_seen_record[file_name]
                if seen_unseen_id == "unseen":
                    total_seen_unseen_IDs += [0 for _ in range(len(labels))]
                elif seen_unseen_id == "seen":
                    total_seen_unseen_IDs += [1 for _ in range(len(labels))]
                else:
                    raise Exception("Check the bug!")
            else:
                total_seen_unseen_IDs += [
                    1 for _ in range(len(labels))
                ]  # 1 indicates seen. This is not used on training or evaluation set

        return (
            total_charge_discharge_curves,
            total_curve_attn_masks,
            np.array(total_labels),
            unique_labels,
            class_labels,
            total_dataset_ids,
            total_cj_aug_charge_discharge_curves,
            total_seen_unseen_IDs,
        )

    def read_cell_data_according_to_prefix(
        self, file_name
    ) -> tuple[dict[str, Any], int | None]:
        """
        Read the battery data and eol according to the file_name
        The dataset is indicated by the prefix of the file_name

        Args:
            file_name: which file needs to be read
        Returns:
            data: The loaded battery data dictionary
            eol: The end-of-life cycle number (None if not reached)
        """
        prefix = file_name.split("_")[0]
        if prefix.startswith("MATR"):
            data = pickle.load(open(f"{self.root_path}/MATR/{file_name}", "rb"))
        elif prefix.startswith("HUST"):
            data = pickle.load(open(f"{self.root_path}/HUST/{file_name}", "rb"))
        elif prefix.startswith("SNL"):
            data = pickle.load(open(f"{self.root_path}/SNL/{file_name}", "rb"))
        elif prefix.startswith("CALCE"):
            data = pickle.load(open(f"{self.root_path}/CALCE/{file_name}", "rb"))
        elif prefix.startswith("HNEI"):
            data = pickle.load(open(f"{self.root_path}/HNEI/{file_name}", "rb"))
        elif prefix.startswith("MICH"):
            if not os.path.isdir(f"{self.root_path}/total_MICH/"):
                self.merge_MICH(f"{self.root_path}/total_MICH/")
            data = pickle.load(open(f"{self.root_path}/total_MICH/{file_name}", "rb"))
        elif prefix.startswith("OX"):
            data = pickle.load(open(f"{self.root_path}/OX/{file_name}", "rb"))
        elif prefix.startswith("RWTH"):
            data = pickle.load(open(f"{self.root_path}/RWTH/{file_name}", "rb"))
        elif prefix.startswith("UL-PUR"):
            data = pickle.load(open(f"{self.root_path}/UL_PUR/{file_name}", "rb"))
        elif prefix.startswith("SMICH"):
            data = pickle.load(open(f"{self.root_path}/MICH_EXP/{file_name[1:]}", "rb"))
        elif prefix.startswith("BIT2"):
            data = pickle.load(open(f"{self.root_path}/BIT2/{file_name}", "rb"))
        elif prefix.startswith("Tongji"):
            data = pickle.load(open(f"{self.root_path}/Tongji/{file_name}", "rb"))
        elif prefix.startswith("Stanford"):
            data = pickle.load(open(f"{self.root_path}/Stanford/{file_name}", "rb"))
        elif prefix.startswith("ISU-ILCC"):
            data = pickle.load(open(f"{self.root_path}/ISU_ILCC/{file_name}", "rb"))
        elif prefix.startswith("XJTU"):
            data = pickle.load(open(f"{self.root_path}/XJTU/{file_name}", "rb"))
        elif prefix.startswith("ZN-coin"):
            data = pickle.load(open(f"{self.root_path}/ZN-coin/{file_name}", "rb"))
        elif prefix.startswith("CALB"):
            data = pickle.load(open(f"{self.root_path}/CALB/{file_name}", "rb"))
        elif prefix.startswith("NA-ion"):
            data = pickle.load(open(f"{self.root_path}/NA-ion/{file_name}", "rb"))

        if prefix == "MICH":
            with open(f"{self.root_path}/Life labels/total_MICH_labels.json") as f:
                life_labels = json.load(f)
        elif prefix.startswith("Tongji"):
            file_name = file_name.replace("--", "-#")
            with open(f"{self.root_path}/Life labels/Tongji_labels.json") as f:
                life_labels = json.load(f)
        else:
            with open(f"{self.root_path}/Life labels/{prefix}_labels.json") as f:
                life_labels = json.load(f)
        if file_name in life_labels:
            eol = life_labels[file_name]
        else:
            eol = None
        return data, eol

    def read_cell_df(
        self, file_name: str
    ) -> tuple[
        pd.DataFrame | None,
        NDArray[np.floating] | None,
        int | None,
        float | None,
        torch.Tensor | None,
    ]:
        """
        Read and process cell dataframe from a single battery file.
        
        Loads raw cycle data, processes it into a pandas DataFrame, extracts
        charge/discharge curves, and applies data augmentation.
        
        Args:
            file_name: Name of the battery data file to read
            
        Returns:
            tuple: (
                df: Processed DataFrame with cycle data (None if battery hasn't reached EOL),
                charge_discharge_curves: Resampled charge/discharge curves [cycles, 3, curve_len],
                eol: End-of-life cycle number (None if not reached),
                nominal_capacity: Battery nominal capacity in Ah,
                cj_aug_charge_discharge_curves: Cutout-jitter augmented curves
            )
        """
        data, eol = self.read_cell_data_according_to_prefix(file_name)
        if eol is None:
            # Battery has not reached end-of-life
            return None, None, None, None, None
        
        # Get nominal capacity (dataset-specific hardcoded values for some datasets)
        if file_name.startswith("RWTH"):
            nominal_capacity = 1.85
        elif file_name.startswith("SNL_18650_NCA_25C_20-80"):
            nominal_capacity = 3.2
        else:
            nominal_capacity = data["nominal_capacity_in_Ah"]

        cycle_data = data["cycle_data"]  # List of cycle data dictionaries

        # Process each cycle into a DataFrame
        total_cycle_dfs = []
        for correct_cycle_index, sub_cycle_data in enumerate(cycle_data):
            cycle_df = pd.DataFrame()
            for key in self.need_keys:
                cycle_df[key] = sub_cycle_data[key]
            cycle_df["cycle_number"] = correct_cycle_index + 1
            
            # Remove outliers (negative capacity values)
            cycle_df.loc[cycle_df["charge_capacity_in_Ah"] < 0] = np.nan
            cycle_df.loc[cycle_df["discharge_capacity_in_Ah"] < 0] = np.nan
            cycle_df.bfill(inplace=True)  # Backfill NaN values
            total_cycle_dfs.append(cycle_df)

            correct_cycle_number = correct_cycle_index + 1
            # Stop if we've reached the early cycle threshold or EOL
            if (
                correct_cycle_number > self.early_cycle_threshold
                or correct_cycle_number > eol
            ):
                break

        df = pd.concat(total_cycle_dfs)
        
        # Extract and resample charge/discharge curves
        charge_discharge_curves = self.get_charge_discharge_curves(
            file_name, df, self.early_cycle_threshold, nominal_capacity
        )
        
        # Apply cutout-jitter augmentation
        cj_aug_charge_discharge_curves, _ = (
            self.aug_helper.batch_aug(charge_discharge_curves)
        )

        return (
            df,
            charge_discharge_curves,
            eol,
            nominal_capacity,
            cj_aug_charge_discharge_curves,
        )

    def read_samples_from_one_cell(
        self, file_name: str
    ) -> tuple[
        list[NDArray[np.floating]] | None,
        list[NDArray[np.floating]] | None,
        list[int] | None,
        int | None,
        list[torch.Tensor] | None,
    ]:
        """
        Extract all samples from a single battery cell file.
        
        Creates multiple samples from one battery by using progressively more
        early-life cycles (sliding window approach). Each sample predicts the
        same EOL from different amounts of early cycle data.
        
        Args:
            file_name: Name of the battery data file to read
            
        Returns:
            tuple: (
                charge_discharge_curves: List of charge/discharge curve arrays,
                attn_masks: List of attention masks indicating valid cycles,
                labels: List of labels (EOL) for each sample,
                eol: End-of-life cycle number (None if invalid),
                total_cj_aug_charge_discharge_curves: List of augmented curves
            )
            Returns (None, None, None, None, None) if battery hasn't reached EOL
            or has insufficient early cycles.
        """
        (
            df,
            charge_discharge_curves_data,
            eol,
            _,
            cj_aug_charge_discharge_curves,
        ) = self.read_cell_df(file_name)
        
        if df is None or eol <= self.early_cycle_threshold:
            # Skip if battery hasn't reached EOL or has too short a life
            return None, None, None, None, None

        # Initialize lists to store samples
        charge_discharge_curves = []
        total_cj_aug_charge_discharge_curves = []
        attn_masks = []
        labels = []
        
        # Extract early-life data (first N cycles)
        early_charge_discharge_curves_data = charge_discharge_curves_data[
            : self.early_cycle_threshold
        ]
        early_cj_aug_charge_discharge_curves = cj_aug_charge_discharge_curves[
            : self.early_cycle_threshold
        ]
        
        # Validate data integrity
        if np.any(np.isnan(early_charge_discharge_curves_data)):
            raise Exception(
                f"Failure in {file_name} | Early data contains NaN! Cycle life is {eol}!"
            )
        
        # Create samples with progressively more early cycles
        for i in range(self.seq_len, self.early_cycle_threshold + 1):
            if i >= eol:
                # Don't include samples that go beyond battery's actual life
                break

            # Create attention mask (1 = valid cycle, 0 = masked/padding)
            tmp_attn_mask = np.zeros(self.early_cycle_threshold)
            tmp_attn_mask[:i] = 1

            # Filter by evaluation cycle range if specified (for val/test sets)
            if self.eval_cycle_max is not None and self.eval_cycle_min is not None:
                if not (self.eval_cycle_min <= i <= self.eval_cycle_max):
                    continue

            # Add sample (all samples from same cell have same EOL label)
            labels.append(eol)
            charge_discharge_curves.append(early_charge_discharge_curves_data)
            total_cj_aug_charge_discharge_curves.append(
                early_cj_aug_charge_discharge_curves
            )
            attn_masks.append(tmp_attn_mask)

        return (
            charge_discharge_curves,
            attn_masks,
            labels,
            eol,
            total_cj_aug_charge_discharge_curves,
        )

    def get_charge_discharge_curves(
        self,
        file_name: str,
        df: pd.DataFrame,
        early_cycle_threshold: int,
        nominal_capacity: float,
    ) -> NDArray[np.floating]:
        """
        Extract and resample charge/discharge curves from cycle data.
        
        Processes raw voltage, current, and capacity measurements for each cycle,
        separates charge and discharge phases, resamples to fixed length, and
        normalizes the values.
        
        Args:
            file_name: Name of the battery data file (used to determine dataset type)
            df: DataFrame containing cycle data with voltage, current, capacity, time
            early_cycle_threshold: Number of early cycles to extract
            nominal_capacity: Battery nominal capacity in Ah for normalization
            
        Returns:
            Array of shape [early_cycle_threshold, 3, charge_discharge_len] containing
            normalized voltage, current, and capacity curves for charge and discharge.
            For cycles that don't exist, returns zeros.
        """
        curves = []
        prefix = file_name.split("_")[0]
        
        # Special handling for CALB dataset naming
        if prefix == "CALB":
            prefix = file_name.split("_")[:2]
            prefix = "_".join(prefix)

        # Process each cycle
        for cycle in range(1, early_cycle_threshold + 1):
            if cycle in df["cycle_number"].unique():
                cycle_df = df.loc[df["cycle_number"] == cycle]

                # Extract raw measurements
                voltage_records = cycle_df["voltage_in_V"].values
                current_records = cycle_df["current_in_A"].values
                current_records_in_C = current_records / nominal_capacity  # Normalize to C-rate
                charge_capacity_records = cycle_df["charge_capacity_in_Ah"].values
                discharge_capacity_records = cycle_df["discharge_capacity_in_Ah"].values
                time_in_s_records = cycle_df["time_in_s"].values

                # Find charge phase end (where current >= 0.01 C-rate)
                # Includes constant-voltage charge data
                cutoff_voltage_indices = np.nonzero(current_records_in_C >= 0.01)
                charge_end_index = cutoff_voltage_indices[0][-1]

                # Find discharge phase end (where current <= -0.01 C-rate)
                cutoff_voltage_indices = np.nonzero(current_records_in_C <= -0.01)
                discharge_end_index = cutoff_voltage_indices[0][-1]

                # Dataset-specific charge/discharge ordering
                # Some datasets discharge first, then charge
                if prefix in ["RWTH", "OX", "ZN-coin", "CALB_0", "CALB_35", "CALB_45"]:
                    # These datasets: Discharge first, then charge
                    discharge_voltages = voltage_records[:discharge_end_index]
                    discharge_capacities = discharge_capacity_records[:discharge_end_index]
                    discharge_currents = current_records[:discharge_end_index]
                    discharge_times = time_in_s_records[:discharge_end_index]

                    charge_voltages = voltage_records[discharge_end_index:]
                    charge_capacities = charge_capacity_records[discharge_end_index:]
                    charge_currents = current_records[discharge_end_index:]
                    charge_times = time_in_s_records[discharge_end_index:]
                    charge_current_in_C = charge_currents / nominal_capacity

                    # Filter out rest periods (current < 0.01 C-rate)
                    charge_voltages = charge_voltages[np.abs(charge_current_in_C) > 0.01]
                    charge_capacities = charge_capacities[np.abs(charge_current_in_C) > 0.01]
                    charge_currents = charge_currents[np.abs(charge_current_in_C) > 0.01]
                    charge_times = charge_times[np.abs(charge_current_in_C) > 0.01]
                else:
                    # Standard datasets: Charge first, then discharge
                    discharge_voltages = voltage_records[charge_end_index:]
                    discharge_capacities = discharge_capacity_records[charge_end_index:]
                    discharge_currents = current_records[charge_end_index:]
                    discharge_times = time_in_s_records[charge_end_index:]
                    discharge_current_in_C = discharge_currents / nominal_capacity

                    # Filter out rest periods (current < 0.01 C-rate)
                    discharge_voltages = discharge_voltages[np.abs(discharge_current_in_C) > 0.01]
                    discharge_capacities = discharge_capacities[np.abs(discharge_current_in_C) > 0.01]
                    discharge_currents = discharge_currents[np.abs(discharge_current_in_C) > 0.01]
                    discharge_times = discharge_times[np.abs(discharge_current_in_C) > 0.01]

                    charge_voltages = voltage_records[:charge_end_index]
                    charge_capacities = charge_capacity_records[:charge_end_index]
                    charge_currents = current_records[:charge_end_index]
                    charge_times = time_in_s_records[:charge_end_index]

                # Resample to fixed length using linear interpolation
                discharge_voltages, discharge_currents, discharge_capacities = (
                    self.resample_charge_discharge_curves(
                        discharge_voltages, discharge_currents, discharge_capacities
                    )
                )
                charge_voltages, charge_currents, charge_capacities = (
                    self.resample_charge_discharge_curves(
                        charge_voltages, charge_currents, charge_capacities
                    )
                )

                # Concatenate charge and discharge phases
                voltage_records = np.concatenate(
                    [charge_voltages, discharge_voltages], axis=0
                )
                current_records = np.concatenate(
                    [charge_currents, discharge_currents], axis=0
                )
                capacity_in_battery = np.concatenate(
                    [charge_capacities, discharge_capacities], axis=0
                )

                # Normalize values
                voltage_records = voltage_records.reshape(
                    1, self.charge_discharge_len
                ) / max(voltage_records)  # Normalize by cutoff voltage
                
                current_records = (
                    current_records.reshape(1, self.charge_discharge_len)
                    / nominal_capacity
                )  # Normalize to C-rate
                
                capacity_in_battery = (
                    capacity_in_battery.reshape(1, self.charge_discharge_len)
                    / nominal_capacity
                )  # Normalize by nominal capacity

                # Stack voltage, current, and capacity as features
                curve_data = np.concatenate(
                    [voltage_records, current_records, capacity_in_battery], axis=0
                )
            else:
                # Fill with zeros for missing cycles
                curve_data = np.zeros((3, self.charge_discharge_len))

            curves.append(
                curve_data.reshape(1, curve_data.shape[0], self.charge_discharge_len)
            )

        curves = np.concatenate(curves, axis=0)  # [L, 3, fixed_len]
        return curves

    def resample_charge_discharge_curves(
        self,
        voltages: NDArray[np.floating],
        currents: NDArray[np.floating],
        capacity_in_battery: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """
        Resample charge and discharge curves to fixed length.
        
        Uses linear interpolation to resample voltage, current, and capacity
        curves to a standardized length for model input.
        
        Args:
            voltages: Voltage measurements during charge or discharge
            currents: Current measurements during charge or discharge
            capacity_in_battery: Remaining capacity in the battery
            
        Returns:
            tuple: (interpolated_voltages, interpolated_currents, interpolated_capacity)
        """
        charge_discharge_len = self.charge_discharge_len // 2
        raw_bases = np.arange(1, len(voltages) + 1)
        interp_bases = np.linspace(
            1, len(voltages) + 1, num=charge_discharge_len, endpoint=True
        )
        interp_voltages = np.interp(interp_bases, raw_bases, voltages)
        interp_currents = np.interp(interp_bases, raw_bases, currents)
        interp_capacity_in_battery = np.interp(
            interp_bases, raw_bases, capacity_in_battery
        )
        return interp_voltages, interp_currents, interp_capacity_in_battery

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get a single sample by index.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing sample data with keys:
                - cycle_curve_data: Charge/discharge curves
                - curve_attn_mask: Attention mask
                - labels: Target label
                - life_class: Life class label
                - scaled_life_class: Scaled life class
                - weight: Sample weight
                - dataset_id: Dataset identifier
                - cj_cycle_curve_data: Cutout-jitter augmented curves
                - seen_unseen_id: Seen/unseen indicator
        """
        sample = {
            "cycle_curve_data": torch.Tensor(self.total_charge_discharge_curves[index]),
            "curve_attn_mask": torch.Tensor(self.total_curve_attn_masks[index]),
            "labels": self.total_labels[index],
            "life_class": self.class_labels[index],
            "scaled_life_class": self.scaled_life_classes[index],
            "weight": self.weights[index],
            "dataset_id": self.total_dataset_ids[index],
            "cj_cycle_curve_data": self.total_cj_aug_charge_discharge_curves[index],
            "seen_unseen_id": self.total_seen_unseen_IDs[index],
        }
        return sample

    def merge_MICH(self, merge_path: str) -> None:
        """
        Merge MICH and MICH_EXP dataset files into a single directory.
        
        Args:
            merge_path: Target directory path for merged files
        """
        os.makedirs(merge_path, exist_ok=True)
        source_path1 = f"{self.root_path}/MICH/"
        source_path2 = f"{self.root_path}/MICH_EXP/"
        source1_files = [i for i in os.listdir(source_path1) if i.endswith(".pkl")]
        source2_files = [i for i in os.listdir(source_path2) if i.endswith(".pkl")]

        for file in source1_files:
            shutil.copy(os.path.join(source_path1, file), merge_path)
        for file in source2_files:
            shutil.copy(os.path.join(source_path2, file), merge_path)


