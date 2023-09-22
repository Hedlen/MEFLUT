import os
import functools
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from PIL import Image
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
	"""Checks if a file is an allowed extension.
	Args:
		filename (string): path to a file
		extensions (iterable of strings): extensions to consider (lowercase)
	Returns:
		bool: True if the filename ends with one of given extensions
	"""
	filename_lower = filename.lower()
	return any(filename_lower.endswith(ext) for ext in extensions)


def image_seq_loader(img_seq_dir):
	img_seq_dir = os.path.expanduser(img_seq_dir)

	img_seq = []
	for root, _, fnames in sorted(os.walk(img_seq_dir)):
		for fname in sorted(fnames):
			if has_file_allowed_extension(fname, IMG_EXTENSIONS):
				image_name = os.path.join(root, fname)
				img_seq.append(Image.open(image_name))

	return img_seq

def get_default_img_seq_loader():
	return functools.partial(image_seq_loader)


class ImageSeqDataset(Dataset):
	def __init__(self, csv_file,
				 hr_img_seq_dir,
				 hr_transform=None,
				 lr_transform=None,
				 get_loader=get_default_img_seq_loader):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			hr_img_seq_dir (string): Directory with all the high resolution image sequences.
			transform (callable, optional): transform to be applied on a sample.
		"""
		self.seqs = pd.read_csv(csv_file, sep='\n', header=None)
		self.hr_root = hr_img_seq_dir
		self.hr_transform = hr_transform
		self.lr_transform = lr_transform
		self.loader = get_loader()

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			samples: a Tensor that represents a video segment.
		"""
		hr_seq_dir = os.path.join(self.hr_root, str(self.seqs.iloc[index, 0]))
		I = [Image.open(os.path.join(hr_seq_dir, filename)) for filename in sorted(os.listdir(hr_seq_dir), key=lambda x:int(x.rstrip('.jpg')))]#self.loader(hr_seq_dir)

		if self.hr_transform is not None:
			I_hr = self.hr_transform(I)
		if self.lr_transform is not None:
			I_lr = self.lr_transform(I)

		I_hr = torch.stack(I_hr, 0).contiguous()
		I_lr = torch.stack(I_lr, 0).contiguous()

		sample = {'I_hr': I_hr, 'I_lr': I_lr, 'case':str(self.seqs.iloc[index, 0])}
		return sample

	def __len__(self):
		return len(self.seqs)

	@staticmethod
	def _reorderBylum(seq):
		I = torch.sum(torch.sum(torch.sum(seq, 1), 1), 1)
		_, index = torch.sort(I)
		result = seq[index, :]
		return result
