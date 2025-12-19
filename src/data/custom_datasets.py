import os
import torch
import pandas as pd
import numpy as np
import logging
import random
from collections import defaultdict
from PIL import Image
import glob
import json
from pathlib import Path
from torch.utils.data import Dataset
from collections import Counter


logger = logging.getLogger(__name__)


class RawMicroscope(Dataset):
    def __init__(
        self,
        root_dir: Path,
        data,
        dataset_path: Path = None,
        train_type: str = "train",
        transform=None,
        num_imgs_per_slide: int = None,
        data_collection_method: str = "image",
        shuffle_labels: bool = False,
        use_monte_carlo: bool = False,
        use_hard_balancing: bool = False,
        num_classes: int = 3,
        reference_image_path: str = None,
        class_idx: list = None,  # List of class indices to include [0, 1] for binary
        merge_classes_for_binary: bool = False,  # Whether to merge classes
        class_merge_mapping: dict = None,  # Mapping for class merging

        embeddings_path: str = None,  # Path to precomputed embeddings
        aug_embeddings_path: str = None,  # Path to precomputed augmented embeddings,
        use_mmap: bool = False,  # Whether to use memory-mapped embeddings
    ):
        # random.seed(42)
        self.root_dir = root_dir
        self.dataset_path = dataset_path
        self.train_type = train_type
        self.transform = transform
        self.num_imgs_per_slide = num_imgs_per_slide
        self.data_collection_method = data_collection_method
        self.shuffle_labels = shuffle_labels and train_type == "train"
        self.use_monte_carlo = use_monte_carlo and train_type == "train"
        self.use_hard_balancing = use_hard_balancing
        self.num_classes = int(num_classes)

        # Class filtering and merging parameters
        self.class_idx = class_idx or list(range(self.num_classes))
        self.merge_classes_for_binary = merge_classes_for_binary
        self.class_merge_mapping = class_merge_mapping or {0: [0], 1: [1, 2]}

        self.index = []
        self.labels = []
        self.slide_ids = []
        self.slide_to_imgs = defaultdict(list)
        self.label_to_indices = defaultdict(list)

        # Path to precomputed embeddings
        self.embeddings_path = None
        self.aug_embeddings_path = None

        if embeddings_path is not None:
            self.embeddings_path = Path(embeddings_path)
            
        if aug_embeddings_path is not None:
            self.aug_embeddings_path = Path(aug_embeddings_path)
        
        # loading embeddings in memory to prevent costly I/O operations during training
        if self.aug_embeddings_path is not None and train_type == "train":
            packed_embeddings_path = self.aug_embeddings_path.parent / "packed" / f"{self.aug_embeddings_path.name}_embeddings.npy"
            if use_mmap:
                self.aug_embeddings = np.load(packed_embeddings_path, mmap_mode='r')
                logger.info("Using memory mapping")
            else:
                logger.info("Not using memory mapping")
                self.aug_embeddings = np.load(packed_embeddings_path)

            json_path = self.aug_embeddings_path.parent / "packed" / f"{self.aug_embeddings_path.name}_keys_to_index.json"
            with open(json_path, "r") as f:
                self.aug_key2idx = json.load(f)  # filename -> index
            
            logger.info(f"Loaded augmented embeddings from {packed_embeddings_path}")


        if self.embeddings_path is not None:
            packed_embeddings_path = self.embeddings_path.parent / "packed" / f"{self.embeddings_path.name}_embeddings.npy"
            if use_mmap:
                self.embeddings = np.load(packed_embeddings_path, mmap_mode='r')
                logger.info("Using memory mapping")
            else:
                self.embeddings = np.load(packed_embeddings_path)
                logger.info("Not using memory mapping")
            
            json_path = self.embeddings_path.parent / "packed" / f"{self.embeddings_path.name}_keys_to_index.json"
            with open(json_path, "r") as f:
                self.key2idx = json.load(f)  # filename -> index
            logger.info(f"Loaded embeddings from {packed_embeddings_path}")

    


        self._build_index(data)


        if isinstance(num_imgs_per_slide, int) and num_imgs_per_slide != 0:
            logger.info(f"Filtering slides to collect {num_imgs_per_slide} bags per slide ...")
            self._filter_slides(num_imgs_per_slide=num_imgs_per_slide)

        if data_collection_method == "slide":
            self._aggregate_by_slide_id()

        if self.use_monte_carlo:
            logger.warn("! MONTE CARLO SAMPLING ACTIVATED FOR TRAIN SET ! ")
            self._prepare_monte_carlo_sampling()
            self._print_monte_carlo_stats()

    def _build_index(self, data):
        """
        Get the paths of the files containing the data and optionally balance the bags across classes.
        """
        labels_to_bags = defaultdict(list)
        logger.info(f"Retrieving {self.train_type.upper()} files paths...")
        for entry in data:
            slide_id = entry["slide_id"]
            image_id = entry["image_id"]
            label = entry["label"]

            # Filter by class_idx if specified
            if label not in self.class_idx:
                continue 
           
            if self.dataset_path is not None:
                pattern = os.path.join(self.dataset_path, f"{slide_id}*{image_id}.tiff")
            else:
                pattern = os.path.join(self.root_dir, "dataset", f"{slide_id}*{image_id}.tiff")
            file_path = sorted(glob.glob(pattern))

            if len(file_path) != 1:
                logger.warning(f"Unexpected number of files ({len(file_path)}) for pattern: {pattern}")
                continue

            file_path = Path(file_path[0])
            labels_to_bags[label].append(
                (file_path, image_id, slide_id)
            )

        # Rest of the function remains the same until the loop that builds the index
        unique_labels = sorted(list(set(labels_to_bags.keys())))
        # unique_labels = sorted(labels_to_bags.keys())
        label_counts = {label: len(labels_to_bags[label]) for label in unique_labels}

        if self.shuffle_labels:
            logger.warn("! SHUFFLING LABELS ACTIVATED ! ")
            shuffled_labels = []
            for label in unique_labels:
                shuffled_labels.extend(random.sample([label] * label_counts[label], label_counts[label]))
            random.shuffle(shuffled_labels)

        for label, bags in labels_to_bags.items():
            if not self.use_monte_carlo:
                if self.use_hard_balancing is True:
                    num_bags = min(len(bags) for bags in labels_to_bags.values())
                else:
                    num_bags = max(len(bags) for bags in labels_to_bags.values())
                bags = bags[:num_bags]
            for file_path, image_id, slide_id in bags:
                self.index.append((file_path, image_id))
                assigned_label = shuffled_labels.pop(0) if self.shuffle_labels else label
                self.labels.append(assigned_label)
                self.slide_ids.append(slide_id)
                self.label_to_indices[assigned_label].append(len(self.index) - 1)

        logger.info(f"# of images: {len(self.index)} | # of slides: {len(set(self.slide_ids))}")
        
        # Remap labels to new indices if class filtering is applied
        if set(self.class_idx) != set(range(self.num_classes)):
            logger.info(f"Remapping labels from {self.class_idx} to [0, 1, ...]")
            label_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(self.class_idx)}
            self.labels = [label_mapping[label] for label in self.labels]
            
            # Update label_to_indices with new mapping
            new_label_to_indices = defaultdict(list)
            for idx, label in enumerate(self.labels):
                new_label_to_indices[label].append(idx)
            self.label_to_indices = new_label_to_indices
            
            logger.info(f"Label counts after remapping: {Counter(self.labels)}")
        
        logger.info(f"Final dataset: {len(self.index)} images | {len(set(self.slide_ids))} slides")

    def _filter_slides(self, num_imgs_per_slide: int = 10):
        """Filter slides to retain exactly num_imgs_per_slide images per slide."""

        filtered_indices = []
        slide_counts_post_filter = {}
        unique_slides = sorted(list(set(self.slide_ids)))
        # for slide_id in set(self.slide_ids):
        for slide_id in unique_slides:
            indices = [i for i, s_id in enumerate(self.slide_ids) if s_id == slide_id]
            if len(indices) == num_imgs_per_slide:
                filtered_indices.extend(indices)
                slide_counts_post_filter[slide_id] = len(indices)
            elif len(indices) > num_imgs_per_slide:
                sampled_indices = random.sample(indices, num_imgs_per_slide)
                filtered_indices.extend(sampled_indices)
                slide_counts_post_filter[slide_id] = len(sampled_indices)
            else:
                logger.info(f"Excluding Slide {slide_id} with {len(indices)} images (less than {num_imgs_per_slide}).")

        self.index = [self.index[i] for i in filtered_indices]
        self.labels = [self.labels[i] for i in filtered_indices]
        self.slide_ids = [self.slide_ids[i] for i in filtered_indices]
        

        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.labels):
            self.label_to_indices[label].append(i)

        logger.info(f"(FILTERED) # of bags : {len(self.index)} | # of slides: {len(set(self.slide_ids))}")
        # Verify that all slides now have exactly num_imgs_per_slide images
        assert all(count == num_imgs_per_slide for count in slide_counts_post_filter.values()), (
            "Some slides do not have the exact number of images after filtering."
        )

    

    def _prepare_monte_carlo_sampling(self):
        if self.data_collection_method == "slide":
            slide_labels = []
            for slide_id, imgs in self.slide_to_imgs.items():
                first_img_data = imgs[0]
                first_img_idx = self.index.index(first_img_data)
                slide_labels.append(self.labels[first_img_idx])

            self.label_to_slides = defaultdict(list)
            for idx, label in enumerate(slide_labels):
                self.label_to_slides[label].append(idx)
            self.min_class_size = min(len(slides) for slides in self.label_to_slides.values())
        else:
            self.min_class_size = min(len(indices) for indices in self.label_to_indices.values())

        logger.info(f"Minimum class size: {self.min_class_size}")
        self.monte_carlo_indices = []
        self._resample_monte_carlo_indices()

    def _print_monte_carlo_stats(self):
        logger.info("Monte Carlo Sampling Statistics:")
        logger.info(f"Minimum class size: {self.min_class_size}")

        if self.data_collection_method == "slide":
            logger.info(f"Number of classes: {len(self.label_to_slides)}")
            logger.info(f"Total slides per epoch: {self.min_class_size * len(self.label_to_slides)}")
            for label, slides in self.label_to_slides.items():
                logger.info(f"Class {label}: {len(slides)} slides, using {self.min_class_size}")
        else:
            logger.info(f"Number of classes: {len(self.label_to_indices)}")
            logger.info(f"Total samples per epoch: {self.min_class_size * len(self.label_to_indices)}")
            for label, indices in self.label_to_indices.items():
                logger.info(f"Class {label}: {len(indices)} samples, using {self.min_class_size}")

    def _resample_monte_carlo_indices(self):
        self.monte_carlo_indices = []
        if self.data_collection_method == "slide":
            for label, slides in self.label_to_slides.items():
                self.monte_carlo_indices.extend(random.sample(slides, self.min_class_size))
        else:
            for label, indices in self.label_to_indices.items():
                self.monte_carlo_indices.extend(random.sample(indices, self.min_class_size))
        random.shuffle(self.monte_carlo_indices)

    def on_epoch_end(self):
        if self.use_monte_carlo:
            self._resample_monte_carlo_indices()

    def _aggregate_by_slide_id(self):
        """Aggregate images by slide ID."""
        for i, (file_path, image_id) in enumerate(self.index):
            slide_data = (file_path, image_id)
            self.slide_to_imgs[self.slide_ids[i]].append(slide_data)

    def __len__(self):
        if self.use_monte_carlo:
            return self.min_class_size * (
                len(self.label_to_slides) if self.data_collection_method == "slide" else len(self.label_to_indices)
            )
        elif self.data_collection_method == "slide":
            return len(self.slide_to_imgs)
        else:
            return len(self.index)

    def _get_raw_single_image(self, idx):
        path, image_id = self.index[idx]
        try:
            img = Image.open(path)
        except Exception as e:
            logger.error(f"{path}: {e}")
            return None

        return img

    def _get_item_per_slide(self, idx):
        """Retrieve a batch of images corresponding to a slide."""
        slide_id = list(self.slide_to_imgs.keys())[idx]
        imgs_data = self.slide_to_imgs[slide_id]

        if self.embeddings_path is not None:
            bags_of_embeddings = []

        bags = []
        for file_path, image_id in imgs_data:
            try:
                if self.embeddings_path is not None:
                    img = torch.zeros((3, 224, 224))  # Dummy image when using embeddings

                    # --- build embedding path ---
                    if self.train_type == "train" and self.aug_embeddings_path is not None:
                        i = random.randrange(0, 10)  # random augmentation for train
                        fname = Path(file_path).stem + f"_aug{i}.npy"
                        # emb_path = Path(self.aug_embeddings_path) / fname
                        emb_idx = self.aug_key2idx.get(fname)
                        arr = self.aug_embeddings[emb_idx]

                    else: # for validation and test, use original image embeddings
                        fname = Path(file_path).stem + ".npy"
                        # emb_path = Path(self.embeddings_path) / fname
                        emb_idx = self.key2idx.get(fname)
                        arr = self.embeddings[emb_idx]

                    if emb_idx is None:
                        raise FileNotFoundError(f"Embedding {fname} not found in packed keys")

                    # load precomputed embeddings
                    embedding = torch.tensor(arr, dtype = torch.float32)

                    bags.append(img) # dummy bag of images lol
                    bags_of_embeddings.append(embedding)
  
                else:
                    with Image.open(file_path) as img:
                        if self.transform:
                            img = self.transform(img)
                        bags.append(img)
            except Exception as e:
                logger.error(f"{file_path}: {e}")
                continue

        if not bags:
            logger.warning(f"No valid images found for slide_id: {slide_id}. Skipping...")
            return None
        
        if self.embeddings_path is not None:
            if not bags_of_embeddings:
                logger.warning(f"No valid embeddings found for slide_id: {slide_id}. Skipping...")
                return None

        label_idx = self.index.index((imgs_data[0][0], imgs_data[0][1]))
        original_label = int(self.labels[label_idx])

        if self.num_classes == 2:
            label = 1 if original_label > 0 else 0
        else:
            label = original_label

        if self.embeddings_path is not None:
            return (
                torch.stack(bags),
                torch.tensor(label),
                slide_id,
                torch.tensor(original_label),
                torch.tensor(idx),
                
                torch.stack(bags_of_embeddings),  # (num_images, embedding_dim)
            )
            
        else:
            return (
                torch.stack(bags),
                torch.tensor(label),
                slide_id,
                torch.tensor(original_label),
                torch.tensor(idx),
            )

    def _get_item_per_image(self, idx):
        """Retrieve an image and its label."""
        path, image_id = self.index[idx]
        try:
            if self.embeddings_path is not None:
                pil_image = torch.zeros((3, 224, 224))  # Dummy image when using embeddings

                # --- build embedding path ---
                if self.train_type == "train" and self.aug_embeddings_path is not None:
                    i = random.randrange(0, 10)  # random augmentation for train
                    fname = Path(path).stem + f"_aug{i}.npy"
                    # emb_path = Path(self.aug_embeddings_path) / fname
                    emb_idx = self.aug_key2idx.get(fname)
                    arr = self.aug_embeddings[emb_idx]

                else: # for validation and test, use original image embeddings
                    fname = Path(path).stem + ".npy"
                    # emb_path = Path(self.embeddings_path) / fname
                    emb_idx = self.key2idx.get(fname)
                    arr = self.embeddings[emb_idx]

                if emb_idx is None:
                    raise FileNotFoundError(f"Embedding {fname} not found in packed keys")

                # load precomputed embeddings
                embedding = torch.tensor(arr, dtype = torch.float32)
                
            else:
                with Image.open(path) as pil_image:
                    if self.transform is not None:
                        pil_image = self.transform(pil_image)
        except Exception as e:
            logger.error(f"{path}: {e}")
            return None

        original_label = int(self.labels[idx])
        
            

        if self.num_classes == 2:
            label = 1 if original_label > 0 else 0
        else:
            label = original_label

        slide_id = self.slide_ids[idx]

        if self.embeddings_path is not None:
            return (
                pil_image,
                torch.tensor(label),
                slide_id,
                torch.tensor(original_label),
                torch.tensor(idx),

                embedding,
            )
        else:
            return (
                pil_image,
                torch.tensor(label),
                slide_id,
                torch.tensor(original_label),
                torch.tensor(idx),
            )

    def __getitem__(self, idx):
        if self.use_monte_carlo:
            original_idx = idx
            idx = self.monte_carlo_indices[idx]
            if (self.data_collection_method == "slide" and idx >= len(self.slide_to_imgs)) or (
                self.data_collection_method == "image" and idx >= len(self.index)
            ):
                logger.error(f"Monte Carlo index out of range: {idx} (original idx: {original_idx})")
                raise IndexError("Monte Carlo index out of range")

        if self.data_collection_method == "slide":
            result = self._get_item_per_slide(idx)
        elif self.data_collection_method == "image":
            result = self._get_item_per_image(idx)
        else:
            raise ValueError(f"Invalid data collection method: {self.data_collection_method}")

        if result is None:
            return self.__getitem__((idx + 1) % self.__len__())
        return result

