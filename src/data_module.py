import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import json
import torch
from tqdm import tqdm

class URBE_Dataset(Dataset):
	def __init__(self, dataset_dir: str, data_type: str, annotations_file_path, hparams):
		self.data = list()
		self.data_type = data_type
		self.dataset_dir = os.path.join(dataset_dir, self.data_type)
		self.annotations = json.load(open(annotations_file_path, "r"))
		self.hparams = hparams
		self.transform = transforms.Compose([
			transforms.Resize((self.hparams.img_size, self.hparams.img_size)),
			# Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
			# to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
			transforms.ToTensor()
			# if I have to add a normalization factor I'll see later
		])
		if self.hparams.augmentation and self.data_type == "train": # a slightly image augmentation
				self.augmentation = transforms.Compose([transforms.RandomVerticalFlip(0.3), 
													 transforms.RandomHorizontalFlip(0.3)])	# da vedere cosa mettere!	
		self.make_data()
	
	def make_data(self):
		# this function read the fresh downloaded dataset and make it ready for the training
		print(f"Loading {self.data_type} dataset...")
		images_folder = [os.path.join(self.dataset_dir,e) for e in os.listdir(self.dataset_dir)]
		max_number = round(self.hparams.max_number_images/8) if (self.data_type == "val" or self.data_type == "test") else self.hparams.max_number_images
		for file_name in tqdm(images_folder[:max_number]):
			image_id = (file_name.split("_")[-1])[:-4]
			img = self.transform(Image.open(file_name).convert('RGB'))
			time = list(filter(lambda x: x["id"] == image_id, self.annotations["images"]))[0]["timeofday"]
			ann_list = list(filter(lambda x: x["image_id"] == image_id, self.annotations["annotations"]))
			labels = []
			for ann in ann_list:
				# since we Resize the image, we need to also change their bounding boxes...
				x1 = ann["bbox"][0] / (1280/self.hparams.img_size) # x1
				y1 = ann["bbox"][1] / (720/self.hparams.img_size) # x2
				x2 = x1 + ann["bbox"][2] / (1280/self.hparams.img_size)
				y2 = y1 + ann["bbox"][3] / (720/self.hparams.img_size)
				w = x2 - x1 # w
				h = y2 - y1 # h
				labels.append( [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2), ann["category_id"]] )
			self.data.append({"id" : image_id, "img" : img, "time" : time, "file_name" : file_name, "labels" : labels})
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		## AUGMENTATION ## 
  		# is performed only on the training set
		if self.hparams.augmentation and self.data_type == "train": # a slightly image augmentation
			data_tmp = self.data[idx]
			data_tmp["img"] = self.augmentation(data_tmp["img"])
			return data_tmp
		else:
			return self.data[idx]

class URBE_DataModule(pl.LightningDataModule):
	# static objs
	c2id = {'vehicle': 0, 'person': 1, 'motorbike': 2}
	id2c = {0: 'vehicle', 1: 'person', 2: 'motorbike'}
 
	def __init__(self, hparams: dict):
		super().__init__()
		self.save_hyperparameters(hparams, logger=False)

	def setup(self, stage=None):
		if not hasattr(self,"data_train"):
			# TRAIN
			self.data_train = URBE_Dataset(self.hparams.dataset_dir, "train", self.hparams.annotations_file_path, self.hparams)
			# VAL
			self.data_val = URBE_Dataset(self.hparams.dataset_dir, "val", self.hparams.annotations_file_path, self.hparams)
			# TEST
			self.data_test = URBE_Dataset(self.hparams.dataset_dir, "test", self.hparams.annotations_file_path, self.hparams)

	def train_dataloader(self):
		return DataLoader(
			self.data_train,
			batch_size=self.hparams.batch_size,
			shuffle=True,
			num_workers=self.hparams.n_cpu,
			collate_fn = self.collate,
			pin_memory=self.hparams.pin_memory,
			persistent_workers=True
		)

	def val_dataloader(self):
		return DataLoader(
			self.data_val,
			batch_size=self.hparams.batch_size,
			shuffle=False,
			num_workers=self.hparams.n_cpu,
			collate_fn = self.collate,
			pin_memory=self.hparams.pin_memory,
			persistent_workers=True
		)
  
	def test_dataloader(self):
		return DataLoader(
			self.data_test,
			batch_size=self.hparams.batch_size,
			shuffle=False,
			num_workers=self.hparams.n_cpu,
   			collate_fn = self.collate,
			pin_memory=self.hparams.pin_memory,
			persistent_workers=True
		)
  
	# we need a collate function because each image have a different number of bounding boxes
	def collate(self, batch):
		batch_out = dict()
		batch_out["id"] = [sample["id"] for sample in batch]
		batch_out["img"] = torch.stack([sample["img"] for sample in batch], dim=0)
		batch_out["time"] = [sample["time"] for sample in batch]
		batch_out["file_name"] = [sample["file_name"] for sample in batch]
		max_number_bbox = torch.tensor([len(sample["labels"]) for sample in batch]).max()
		batch_out["labels"] = []
		for sample in batch:
			batch_out["labels"].append( sample["labels"] + [ [] for _ in range(max_number_bbox - len(sample["labels"]))] )
		return batch_out
     
	
	#to invert the normalization of the compose transform. --> ci servirà?
	@staticmethod
	def denormalize(tensor):
		return tensor*0.5 + 0.5