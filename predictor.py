import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import Counter
import warnings

# --- 1. CONFIGURATION ---

# Hardcode the model path here
MODEL_PATH = 'late_fusion_model.pt'

# Suppress a specific deprecation warning from PyTorch
warnings.filterwarnings("ignore", category=UserWarning, message=".*ResNet50_Weights.IMAGENET1K_V1.*")

# --- 2. REQUIRED CLASS DEFINITIONS (from your training script) ---
# These classes MUST be defined to load the pickled/saved model.

class SimpleTokenizer:
    """Simple tokenizer for text data. Must match the one used in training."""

    def __init__(self, max_tokens=5000, max_length=256):
        self.max_tokens = max_tokens
        self.max_length = max_length
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2

    def fit(self, texts):
        """Build vocabulary from texts."""
        word_freq = Counter()
        for text in texts:
            words = str(text).lower().split()
            word_freq.update(words)
        
        most_common = word_freq.most_common(self.max_tokens - 2)

        for word, _ in most_common:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    def encode(self, text):
        """Convert text to sequence of indices."""
        words = str(text).lower().split()
        indices = [self.word2idx.get(word, 1) for word in words]  # 1 is <UNK>

        # Pad or truncate
        if len(indices) < self.max_length:
            indices += [0] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]

        return indices

class LateFusionModel(nn.Module):
    """Late Fusion model. Must match the one used in training."""

    def __init__(self, vocab_size, embed_dim=128, lstm_hidden=256):
        super(LateFusionModel, self).__init__()

        # Image Branch
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = False

        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])

        self.image_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        # Text Branch
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True)

        self.text_predictor = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, images, captions):
        # Image branch
        image_features = self.image_encoder(images)
        image_pred = self.image_predictor(image_features)

        # Text branch
        embedded = self.embedding(captions)
        lstm_out, (hidden, _) = self.lstm(embedded)
        text_features = hidden[-1]
        text_pred = self.text_predictor(text_features)

        # Late fusion - combine predictions
        combined = torch.cat([image_pred, text_pred], dim=1)
        output = self.fusion(combined)

        return output

# --- 3. PREDICTOR CLASS ---

class PostPredictor:
    """
    A class to load the trained LateFusionModel (from the hardcoded
    MODEL_PATH) and provide a simple prediction interface.
    """
    def __init__(self, device=None):
        """
        Initializes the predictor.
        
        Args:
            device (torch.device, optional): Device to run on. 
                                            Autodetects if None.
        """
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model on device: {self.device}")

        # Load the checkpoint from the global MODEL_PATH
        try:
            if not os.path.exists(MODEL_PATH):
                 raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
                 
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Error loading model checkpoint from {MODEL_PATH}: {e}")
            raise

        # Load tokenizer
        if 'tokenizer' not in checkpoint:
            raise ValueError("Tokenizer not found in checkpoint.")
        self.tokenizer = checkpoint['tokenizer']

        # Load model configuration
        if 'vocab_size' not in checkpoint:
            raise ValueError("vocab_size not found in checkpoint.")
        vocab_size = checkpoint['vocab_size']

        # Initialize model and load weights
        self.model = LateFusionModel(vocab_size=vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode (IMPORTANT)

        # Define the image transforms (must match training/validation)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _preprocess_image(self, image_input):
        """Preprocesses the image (path or PIL) into a tensor."""
        if isinstance(image_input, str):
            try:
                image = Image.open(image_input).convert('RGB')
            except FileNotFoundError:
                print(f"Error: Image file not found at {image_input}")
                raise
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise TypeError("image_input must be a file path (str) or PIL Image")
        
        # Apply transforms and add batch dimension
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor

    def _preprocess_text(self, caption):
        """Preprocesses the text caption into a tensor."""
        indices = self.tokenizer.encode(caption)
        caption_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
        return caption_tensor

    def predict(self, image_input, caption):
        """
        Makes a performance prediction for an image and caption.
        
        Args:
            image_input (str or PIL.Image): Path to the image or a PIL Image object.
            caption (str): The text caption for the post.
            
        Returns:
            dict: A dictionary containing 'score_normalized' (0-1) and
                  'score_percentage' (0-100).
        """
        # Preprocess inputs
        image_tensor = self._preprocess_image(image_input).to(self.device)
        caption_tensor = self._preprocess_text(caption).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor, caption_tensor)
        
        # Get the raw score (model predicts 0-1)
        score_normalized = output.item()
        
        # Convert to percentage and clip between 0 and 100
        score_percentage = score_normalized * 100.0
        score_percentage_clipped = max(0.0, min(100.0, score_percentage))

        return {
            'score_normalized': score_normalized,
            'score_percentage': score_percentage_clipped
        }

# --- 4. EXAMPLE USAGE (when run as a script) ---

if __name__ == "__main__":
    
    # This block only runs when you execute `python predictor.py` directly
    
    if not os.path.exists(MODEL_PATH):
        print("="*50)
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        print("Please make sure 'late_fusion_model.pt' is in the same directory.")
        print("="*50)
    else:
        try:
            # 1. Initialize the predictor (no path needed)
            print("Initializing predictor...")
            predictor = PostPredictor()
            print("Predictor initialized.")
            
            # 2. Create a dummy image for testing
            dummy_image = Image.new('RGB', (224, 224), color='blue')
            dummy_caption = "This is a test caption for my post!"

            # 3. Make a prediction
            print("Running test prediction...")
            result = predictor.predict(dummy_image, dummy_caption)
            
            print("\n--- Prediction Result (Dummy Image) ---")
            print(f"Caption: '{dummy_caption}'")
            print(f"Normalized Score: {result['score_normalized']:.4f}")
            print(f"Percentage Score: {result['score_percentage']:.2f}%")
            
        except Exception as e:
            print(f"\nAn error occurred during the test: {e}")