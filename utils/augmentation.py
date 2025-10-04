"""
Data augmentation utilities for battery life prediction.

This module provides augmentation techniques for battery charge/discharge curves
to improve model generalization and robustness.
"""

import torch
import numpy as np
import copy
from numpy.typing import NDArray


class BatchAugmentation_battery_revised:
    """
    Batch-level data augmentation for battery charge/discharge curves.
    
    This class applies two types of augmentation techniques:
    1. Cutout with jitter: Randomly masks segments and adds noise
    2. Frequency masking: Masks frequency components in the Fourier domain
    
    Args:
        aug_rate (float): Rate for general augmentation. Default: 0.5
        cut_rate (float): Rate for cutout augmentation. Default: 0.5
        holes (int): Number of cutout holes to apply. Default: 10
        length (int): Length of each cutout segment. Default: 5
        std (float): Standard deviation for jitter noise. Default: 0.02
    """
    
    def __init__(
        self,
        aug_rate: float = 0.5,
        cut_rate: float = 0.5,
        holes: int = 10,
        length: int = 5,
        std: float = 0.02
    ) -> None:
        self.cut_rate = cut_rate
        self.aug_rate = aug_rate
        self.cutout_aug = Cutout_jitter_aug(holes, length, std=std)

    def freq_mask(self, x: torch.Tensor, rate: float = 0.25, dim: int = 1) -> torch.Tensor:
        """
        Apply frequency masking in the Fourier domain.
        
        Masks out random frequency components while preserving dominant frequencies.
        This helps the model learn robust features that are invariant to certain
        frequency perturbations.
        
        Args:
            x (Tensor): Input tensor of shape [B*L, charge_discharge_len]
            rate (float): Probability of masking each frequency component. Default: 0.25
            dim (int): Dimension along which to apply FFT. Default: 1
            
        Returns:
            Tensor: Frequency-masked tensor with the same shape as input
        """
        xy = x
        xy_f = torch.fft.rfft(xy, dim=dim)
        m = torch.ones(xy_f.shape, dtype=x.dtype, device=x.device)
        m = m.uniform_(0, 1) < rate
        
        amp = abs(xy_f)
        _, index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 5
        m = torch.bitwise_and(m, dominant_mask)
        
        freal = xy_f.real.masked_fill(m, 0)
        fimag = xy_f.imag.masked_fill(m, 0)
        xy_f = torch.complex(freal, fimag)
        xy = torch.fft.irfft(xy_f, dim=dim)
        return xy

    def batch_aug(self, x: NDArray[np.floating]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply batch augmentation to battery charge/discharge curves.
        
        This method creates two augmented versions of the input:
        1. Cutout-jitter augmentation: Masks random segments and adds noise
        2. Frequency masking: Masks frequency components in Fourier domain
        
        The capacity records are preserved without augmentation as they represent
        the actual battery state.
        
        Args:
            x (ndarray): Input array of shape [B*L, num_var, charge_discharge_len]
                        where num_var includes voltage, current, and capacity
            
        Returns:
            tuple: (cut_aug_x, freqmask_aug_x)
                - cut_aug_x (Tensor): Cutout-jitter augmented data
                - freqmask_aug_x (Tensor): Frequency-masked augmented data
        """
        x_tensor: torch.Tensor = torch.from_numpy(x)
        voltage_records: torch.Tensor = x_tensor[:, 0, :]
        current_records: torch.Tensor = x_tensor[:, 1, :]
        capacity_records: torch.Tensor = x_tensor[:, -1, :].unsqueeze(1)
        
        # Cutout-jitter augmentation
        cut_aug_voltage: torch.Tensor = self.cutout_aug(voltage_records.unsqueeze(1))
        cut_aug_current: torch.Tensor = self.cutout_aug(current_records.unsqueeze(1))
        cut_aug_x: torch.Tensor = torch.cat([cut_aug_voltage, cut_aug_current, capacity_records], dim=1)

        # Frequency masking augmentation
        freqmask_aug_voltage: torch.Tensor = self.freq_mask(voltage_records)
        freqmask_aug_current: torch.Tensor = self.freq_mask(current_records)
        freqmask_aug_voltage = freqmask_aug_voltage.unsqueeze(1)
        freqmask_aug_current = freqmask_aug_current.unsqueeze(1)
        freqmask_aug_x: torch.Tensor = torch.cat([freqmask_aug_voltage, freqmask_aug_current, capacity_records], dim=1)

        return cut_aug_x, freqmask_aug_x


class Cutout_jitter_aug(object):
    """
    Cutout augmentation with jitter for time series data.
    
    Randomly cuts out segments from the time series and replaces them with
    noised versions. This helps the model learn to handle missing or noisy data.
    
    Args:
        n_holes (int): Number of cutout patches to apply
        length (int): Length of each cutout patch
        mean (float): Mean of the Gaussian noise. Default: 0
        std (float): Standard deviation of the Gaussian noise. Default: 0.01
    """
    
    def __init__(self, n_holes: int, length: int, mean: float = 0, std: float = 0.01) -> None:
        self.n_holes = n_holes
        self.length = length
        self.mean = mean
        self.std = std

    def __call__(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Apply cutout-jitter augmentation to a sequence.
        
        Args:
            seq (Tensor): Tensor sequence of size (B, num_var, L)
            
        Returns:
            Tensor: Augmented sequence with n_holes of dimension length cut out
                   and replaced with noised versions
        """
        seq = seq.transpose(1, 2)  # [B, L, num_var]
        L = seq.size(1)

        mask = torch.ones_like(seq[0])  # [L, num_var]
        noise_ratio = torch.normal(
            self.mean, 
            self.std, 
            size=(seq.shape[0], seq.shape[1], seq.shape[2]), 
            device=seq.device
        )
        noise_ratio = torch.clip(
            noise_ratio, 
            min=self.mean - 3 * self.std, 
            max=self.mean + 3 * self.std
        )
        noisy_seq = seq + seq * noise_ratio
        
        for n in range(self.n_holes):
            y = np.random.randint(L)
            y1 = np.clip(y - self.length // 2, 0, L)
            y2 = np.clip(y + self.length // 2, 0, L)
            mask[y1:y2, :] = 0.

        mask = mask.expand_as(seq)
        seq = seq * mask + (1 - mask) * noisy_seq

        return seq.transpose(1, 2)


if __name__ == "__main__":
    # Test the augmentation
    aug_helper = BatchAugmentation_battery_revised()
    x = np.random.rand(2, 3, 100)
    original_x = copy.deepcopy(x)
    x_aug1, x_aug2 = aug_helper.batch_aug(x)
    
    if np.all(original_x == x):
        print('Original data unchanged: OK')
    if np.any(original_x != x_aug1.numpy()):
        print('Cutout-jitter augmentation applied: OK')
    if np.any(original_x != x_aug2.numpy()):
        print('Frequency masking augmentation applied: OK')
