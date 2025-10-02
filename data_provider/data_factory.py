from typing import Literal
from data_provider.data_loader import Dataset_original, my_collate_fn_baseline, my_collate_fn_withId
from torch.utils.data import DataLoader, RandomSampler

dataset_types = Literal['baseline', 'evaluate', 'da']
ds_types = Literal['train', 'val', 'test']

def data_provider(
    args,
    flag: ds_types,
    mode: dataset_types = 'baseline',
    label_scaler=None,
    eval_cycle_min=None,
    eval_cycle_max=None,
    life_class_scaler=None,
) -> tuple[Dataset_original, DataLoader] | tuple[Dataset_original, DataLoader, Dataset_original, DataLoader]:
    """
    Unified data provider for training, validation, testing, and domain adaptation.

    Args:
        args: Configuration arguments.
        flag (str): 'train', 'val', or 'test'.
        mode (str): Determines the function's behavior.
                    - 'baseline': Standard data loader.
                    - 'evaluate': Uses a collate function that includes IDs.
                    - 'da': Enables domain adaptation logic with a target dataset.
        label_scaler: Scaler for labels.
        eval_cycle_min: Minimum evaluation cycle.
        eval_cycle_max: Maximum evaluation cycle.
        life_class_scaler: Scaler for life classes.
    """
    if flag in ['test', 'val']:
        shuffle_flag = False
        drop_last = False
    else: # 'train'
        shuffle_flag = True
        drop_last = True

    # Determine which collate function to use based on the mode
    collate_function = my_collate_fn_withId if mode == 'evaluate' else my_collate_fn_baseline

    # Determine if the main dataset should be loaded from the target files
    # This is only True for the DA mode during validation/testing
    use_target_for_main_dataset = (mode == 'da' and flag in ['test', 'val'])

    data_set = Dataset_original(
        args=args,
        flag=flag,
        label_scaler=label_scaler,
        eval_cycle_min=eval_cycle_min,
        eval_cycle_max=eval_cycle_max,
        life_class_scaler=life_class_scaler,
        use_target_dataset=use_target_for_main_dataset,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=collate_function,
    )

    # Specific logic for Domain Adaptation (DA) during the training phase
    if mode == 'da' and flag == 'train':
        target_data_set = Dataset_original(
            args=args,
            flag=flag,
            label_scaler=data_set.return_label_scaler(),
            eval_cycle_min=eval_cycle_min,
            eval_cycle_max=eval_cycle_max,
            life_class_scaler=data_set.return_life_class_scaler(),
            use_target_dataset=True, # The target loader always uses the target dataset
        )

        # Create a sampler to oversample the smaller target dataset to match the source dataset size
        target_sampler = RandomSampler(
            target_data_set,
            replacement=True,
            num_samples=len(data_set),
        )

        target_resampled_dataloader = DataLoader(
            target_data_set,
            batch_size=args.batch_size,
            sampler=target_sampler,
            num_workers=args.num_workers,
            collate_fn=my_collate_fn_baseline,
        )
        return data_set, data_loader, target_data_set, target_resampled_dataloader

    return data_set, data_loader