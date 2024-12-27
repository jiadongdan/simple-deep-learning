import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 2: Define the Vision Transformer (ViT) model
class SimpleViT(nn.Module):
    def __init__(self, image_size=28, patch_size=7, num_classes=10, dim=64, depth=6, heads=8, mlp_dim=128):
        super(SimpleViT, self).__init__()
        
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size
        
        # Embedding layer
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, dim))  # Randomly initialized positional embedding; this is a learnable parameter that encodes the position of each patch
        
        # Transformer blocks
        self.transformer = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True) for _ in range(depth)]
        )
        
        # Classification head
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        # Other settings
        self.patch_size = patch_size
        self.dim = dim

    def forward(self, x):
        # Input shape: (B, C, H, W), where B=batch size, C=channels, H=height, W=width

        # Step 1: Extract patches using unfold
        # Unfold slices the input tensor into patches of size (patch_size, patch_size) along height (H) and width (W)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # Shape after unfold: (B, C, num_patches_H, num_patches_W, patch_size, patch_size), where num_patches_H = H/patch_size

        # Step 2: Rearrange dimensions
        x = x.permute(0, 2, 3, 1, 4, 5)  # Rearrange to: (B, num_patches_H, num_patches_W, C, patch_size, patch_size)
        
        # Step 3: Flatten each patch
        # Reshape to: (B, num_patches, patch_dim), where num_patches = num_patches_H * num_patches_W, patch_dim = patch_size * patch_size
        x = x.reshape(x.shape[0], -1, self.patch_size * self.patch_size)

        # Step 4: Apply linear patch embedding and add positional encoding
        x = self.patch_embedding(x) + self.positional_embedding

        # Step 5: Pass through transformer blocks
        x = self.transformer(x)

        # Step 6: Aggregate patch embeddings (mean pooling)
        x = x.mean(dim=1)
        
        # Step 7: Pass through classification head
        x = self.mlp_head(x)
        return x

# Step 3: Train the model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleViT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Step 4: Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

