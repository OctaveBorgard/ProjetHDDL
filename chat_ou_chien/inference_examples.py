"""
Examples of how to properly handle lists of images for inference
"""
import torch
from models import Unet_Segmenterv2
from torch.nn.utils.rnn import pad_sequence


# ============================================================================
# EXAMPLE 1: Same-sized images (MOST EFFICIENT)
# ============================================================================
def example_1_same_size():
    """Process multiple images of the same size - fastest approach"""
    model = Unet_Segmenterv2(in_channels=3, num_classes=3)
    
    # Your list of images (same size)
    img_1 = torch.randn(3, 224, 224)
    img_2 = torch.randn(3, 224, 224)
    img_3 = torch.randn(3, 224, 224)
    img_list = [img_1, img_2, img_3]
    
    # Stack into a batch
    batch = torch.stack(img_list)  # Shape: (3, 3, 224, 224)
    
    # Single forward pass - processes all at once!
    outputs = model(batch)  # Shape: (3, 3, 224, 224)
    
    print("Example 1 - Same size:")
    print(f"  Input batch shape: {batch.shape}")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Processed 3 images in 1 forward pass\n")


# ============================================================================
# EXAMPLE 2: Different-sized images with padding (PRACTICAL)
# ============================================================================
def example_2_variable_size_with_padding():
    """Process images of different sizes by padding to max"""
    model = Unet_Segmenterv2(in_channels=3, num_classes=3)
    
    # Your list of images (different sizes)
    img_1 = torch.randn(3, 224, 224)
    img_2 = torch.randn(3, 256, 256)
    img_3 = torch.randn(3, 228, 228)
    img_list = [img_1, img_2, img_3]
    
    # Find max size
    max_h = max(img.shape[1] for img in img_list)
    max_w = max(img.shape[2] for img in img_list)
    print(f"Example 2 - Variable size (max: {max_h}x{max_w}):")
    
    # Pad all to max size
    padded = []
    for img in img_list:
        h_pad = max_h - img.shape[1]
        w_pad = max_w - img.shape[2]
        padded_img = torch.nn.functional.pad(img, (0, w_pad, 0, h_pad))
        padded.append(padded_img)
    
    # Stack into batch
    batch = torch.stack(padded)  # Shape: (3, 3, 256, 256)
    
    # Single forward pass
    outputs = model(batch)  # Shape: (3, 3, 256, 256)
    
    # Remove padding from outputs
    results = []
    for i, img in enumerate(img_list):
        h, w = img.shape[1], img.shape[2]
        output = outputs[i, :, :h, :w]  # Crop back to original size
        results.append(output)
    
    print(f"  Input shapes: {[img.shape for img in img_list]}")
    print(f"  Padded batch shape: {batch.shape}")
    print(f"  Output shapes after cropping: {[r.shape for r in results]}")
    print(f"  Processed 3 images (different sizes) in 1 forward pass\n")


# ============================================================================
# EXAMPLE 3: Custom collate function for DataLoader
# ============================================================================
def collate_variable_size(batch):
    """
    Custom collate function for DataLoader to handle variable-sized images
    """
    # batch is a list of (image, label) tuples
    images, labels = zip(*batch)
    
    # Find max dimensions
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    # Pad images to max size
    padded_images = []
    for img in images:
        h_pad = max_h - img.shape[1]
        w_pad = max_w - img.shape[2]
        padded_img = torch.nn.functional.pad(img, (0, w_pad, 0, h_pad))
        padded_images.append(padded_img)
    
    # Stack and return
    images_batch = torch.stack(padded_images)
    labels_batch = torch.stack(labels)
    
    return images_batch, labels_batch


def example_3_dataloader():
    """Using custom collate with DataLoader"""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dummy dataset with variable-sized images
    # In real scenario, you'd load actual images
    images = [
        torch.randn(3, 224, 224),
        torch.randn(3, 256, 256),
        torch.randn(3, 228, 228),
    ]
    labels = torch.tensor([0, 1, 0])
    
    dataset = TensorDataset(torch.stack(images) if all(
        img.shape == images[0].shape for img in images
    ) else torch.tensor(images, dtype=torch.float32), labels)
    
    # This won't work well with TensorDataset for variable sizes, 
    # so here's a better approach:
    
    class VariableSizeDataset(torch.utils.data.Dataset):
        def __init__(self, image_list, label_list):
            self.images = image_list
            self.labels = label_list
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]
    
    dataset = VariableSizeDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_variable_size)
    
    model = Unet_Segmenterv2(in_channels=3, num_classes=3)
    
    print("Example 3 - DataLoader with variable sizes:")
    for batch_images, batch_labels in dataloader:
        print(f"  Batch images shape: {batch_images.shape}")
        print(f"  Batch labels shape: {batch_labels.shape}")
        outputs = model(batch_images)
        print(f"  Model outputs shape: {outputs.shape}")
        break  # Just show first batch
    print()


# ============================================================================
# EXAMPLE 4: Batch processing with size preservation
# ============================================================================
def example_4_with_size_tracking():
    """Keep track of original sizes for proper output cropping"""
    model = Unet_Segmenterv2(in_channels=3, num_classes=3)
    
    img_1 = torch.randn(3, 224, 224)
    img_2 = torch.randn(3, 256, 256)
    img_3 = torch.randn(3, 228, 228)
    img_list = [img_1, img_2, img_3]
    
    # Store original shapes
    original_shapes = [(img.shape[1], img.shape[2]) for img in img_list]
    
    # Pad to max
    max_h = max(h for h, w in original_shapes)
    max_w = max(w for h, w in original_shapes)
    
    padded = []
    for img in img_list:
        h_pad = max_h - img.shape[1]
        w_pad = max_w - img.shape[2]
        padded_img = torch.nn.functional.pad(img, (0, w_pad, 0, h_pad))
        padded.append(padded_img)
    
    batch = torch.stack(padded)
    outputs = model(batch)
    
    # Extract outputs with original sizes
    results = {}
    for i, (h, w) in enumerate(original_shapes):
        results[f"image_{i}"] = outputs[i, :, :h, :w]
    
    print("Example 4 - Size tracking:")
    print(f"  Original shapes: {original_shapes}")
    print(f"  Output shapes (restored): {[(v.shape[1], v.shape[2]) for v in results.values()]}")
    print()


# ============================================================================
# Run examples
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("INFERENCE EXAMPLES FOR MULTIPLE IMAGES")
    print("=" * 70 + "\n")
    
    example_1_same_size()
    example_2_variable_size_with_padding()
    example_3_dataloader()
    example_4_with_size_tracking()
    
    print("=" * 70)
    print("RECOMMENDATION:")
    print("  - Same size → Example 1 (fastest)")
    print("  - Variable size → Example 2 or 4 (balanced)")
    print("  - Training pipeline → Example 3 (DataLoader)")
    print("=" * 70)
