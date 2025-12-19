from typing import Tuple, List
import pandas as pd
import logging
from sklearn.model_selection import StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)


class MicroscopeKFoldSplitter:
    def __init__(self, csv_file_path, metadata_file_path, num_folds=5, random_state=42):
        self.csv_file_path = csv_file_path
        self.metadata_file_path = metadata_file_path
        self.num_folds = num_folds
        self.random_state = int(random_state)
        self.data = self._load_csv()
        self.folds = self._prepare_folds() if num_folds > 1 else self._prepare_single_split()

    def _load_csv(self) -> List[dict]:
        main_data = pd.read_csv(self.csv_file_path)
        metadata = pd.read_csv(self.metadata_file_path)

        # ensure similar data types for merging
        main_data["slide_id"] = main_data["slide_id"].astype(str)
        metadata["NUM DOSS"] = metadata["NUM DOSS"].astype(str)

        merged_data = main_data.merge(metadata, left_on="slide_id", right_on="NUM DOSS", how="left")

        return merged_data.to_dict(orient="records")

    def _prepare_folds(self) -> List[Tuple[list, list]]:
        logger.info(f"Preparing {self.num_folds} folds for stratified K-Fold cross-validation")

        for d in self.data:
            d["stratify_label"] = f"{d['label']}_{d['Etude']}"

        slide_labels = {d["slide_id"]: d["stratify_label"] for d in self.data}
        unique_slides = list(set(slide_labels.keys()))
        labels = [slide_labels[slide] for slide in unique_slides]

        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_state)
        folds = []

        for train_idx, val_idx in skf.split(unique_slides, labels):
            train_slides = set(unique_slides[idx] for idx in train_idx)
            val_slides = set(unique_slides[idx] for idx in val_idx)

            assert len(set(train_slides).intersection(set(val_slides))) == 0, "! DATA LEAKAGE ! "

            if len(set(train_slides).intersection(set(val_slides))) == 0:
                logger.info("No data leakage between train & val splits")

            train_data = [d for d in self.data if d["slide_id"] in train_slides]
            val_data = [d for d in self.data if d["slide_id"] in val_slides]

            folds.append((train_data, val_data))

        return folds

    def _prepare_single_split(self) -> List[Tuple[list, list]]:
        for d in self.data:
            d["stratify_label"] = f"{d['label']}_{d['Etude']}"

        slide_labels = {d["slide_id"]: d["stratify_label"] for d in self.data}
        unique_slides = list(set(slide_labels.keys()))
        labels = [slide_labels[slide] for slide in unique_slides]

        # Check class distribution
        label_counts = pd.Series(labels).value_counts()
        logger.info("Class distribution before split:")
        for label, count in label_counts.items():
            logger.info(f"Label: {label}, Count: {count}")

        # Identify classes with fewer than 2 samples
        low_count_labels = label_counts[label_counts < 2].index
        if not low_count_labels.empty:
            logger.warning(
                f"The following classes have fewer than 2 samples and will be excluded: {list(low_count_labels)}"
            )

        # Filter out slides with low count labels
        valid_slides = [slide for slide in unique_slides if slide_labels[slide] not in low_count_labels]
        valid_labels = [slide_labels[slide] for slide in valid_slides]

        # Proceed with the stratified split if there are valid slides
        if len(valid_slides) < 2:
            raise ValueError("Not enough valid slides for stratified split.")

        train_slides, val_slides = train_test_split(
            valid_slides,
            test_size=0.25,
            stratify=valid_labels,
            random_state=int(self.random_state),
        )

        assert len(set(train_slides).intersection(set(val_slides))) == 0, "! DATA LEAKAGE ! "
        if len(set(train_slides).intersection(set(val_slides))) == 0:
            logger.info("No data leakage between train & val splits")

        train_data = [d for d in self.data if d["slide_id"] in train_slides]
        val_data = [d for d in self.data if d["slide_id"] in val_slides]

        return [(train_data, val_data)]

    def get_fold(self, fold_index=0) -> Tuple[list, list]:
        """Returns the train and validation datasets for the specified fold or single split."""
        train_data, val_data = self.folds[fold_index]
        return train_data, val_data
