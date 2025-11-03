from flask import Flask, jsonify, render_template, request
from predictor import PostPredictor
from collections import Counter
from PIL import Image  # <-- ADD THIS
import io               # <-- ADD THIS
import traceback        # <-- ADD THIS
import warnings

# --- This class must be here for the model to load ---
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
# --- End of required class ---


app = Flask(__name__)

# Initialize the predictor
try:
    predictor = PostPredictor()
    print("✅ Predictor loaded successfully.")
except Exception as e:
    print(f"❌ Error loading predictor: {e}")
    print(traceback.format_exc())
    predictor = None

@app.route("/")
def root():
    return render_template("index.html")

# --- THIS IS THE CORRECTED ROUTE ---
@app.route("/predict", methods=["POST"])
def predict():
    if predictor is None:
        return jsonify({
            'success': False, 
            'error': 'Model is not loaded. Check server logs.'
        }), 500

    # Check if the correct form parts are present
    if 'image' not in request.files or 'caption' not in request.form:
        return jsonify({
            'success': False, 
            'error': 'Missing image or caption in form data'
        }), 400

    try:
        # Read the image file from request.files
        image_file = request.files['image']
        
        # Read the caption text from request.form
        caption = request.form['caption']

        # Convert the file stream to a PIL Image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Get the prediction
        result = predictor.predict(image, caption)

        print(f"Prediction result: {result}")
        
        # Return the actual result from the predictor
        return jsonify({
            'success': True,
            'score_normalized': result['score_normalized'],
            'score_percentage': result['score_percentage']
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False, 
            'error': f'Prediction failed: {str(e)}'
        }), 500
# --- END OF CORRECTION ---


if __name__ == "__main__":
    app.run(debug=True, port=5000)