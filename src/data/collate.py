import torch

def collate_func(batch):
    """
    Custom collate function to handle batches
    """
    # Filter out None values (from failed loads)
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None

    images, labels, slide_ids, original_labels, indices = zip(*batch)

    if isinstance(images[0], torch.Tensor) and images[0].dim() > 3:
        # For bags of images (slide-level)
        images = torch.stack(images)
    else:
        # For single images
        images = torch.stack([img for img in images])

    return (
        images,
        torch.stack([label for label in labels]),
        slide_ids,
        torch.stack([label for label in original_labels]),
        torch.stack([index for index in indices]),
    )


def collate_func_embeddings(batch):
    """
    Custom collate function to handle batches with embeddings.


    This function:
    - filters out None samples (failed loads)
    - filters out samples whose embedding is None (failed embedding load)
    - stacks tensors appropriately and returns embeddings as the last element.
    """
    # filter out None samples
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None

    # filter out samples with missing embedding (embedding is last element)
    filtered = []
    for x in batch:
        if len(x) == 0:
            continue
        emb = x[-1]
        if emb is None:
            # log this worker-side (optional)
            # print(f"[COLLATE] skipping sample with missing embedding")
            continue
        filtered.append(x)
    batch = filtered
    if len(batch) == 0:
        return None

    # (image, label, slide_id, original_label, index, embedding)
    images, labels, slide_ids, original_labels, indices, embeddings = zip(*batch)

    if isinstance(images[0], torch.Tensor) and images[0].dim() > 3:
        images = torch.stack(images)
    else:
        images = torch.stack([img for img in images])

    labels = torch.stack([lbl for lbl in labels])
    original_labels = torch.stack([ol for ol in original_labels])
    indices = torch.stack([ix for ix in indices])

    embeddings = torch.stack([torch.as_tensor(e, dtype=torch.float32) if not isinstance(e, torch.Tensor) else e.float() for e in embeddings])

    return (
        images,
        labels,
        slide_ids,
        original_labels,
        indices,
        embeddings,
    )
