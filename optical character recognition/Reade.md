OCR System
This project implements an Optical Character Recognition system that performs character enrollment, detection, and recognition using SIFT features and connected component labeling.

Features
Enrollment:

Padded and resized character images are processed using SIFT.

Features are stored in features.json.

Detection:

Converts a test image to binary.

Uses BFS-based connected component labeling to extract bounding boxes for each character.

Recognition:

Crops detected characters from a processed test image.

Matches extracted SIFT descriptors against enrolled character features.

Labels characters based on matching thresholds, with "UNKNOWN" for unmatched characters.

Project Structure
bash
Copy
.
├── data
│   ├── characters           # Folder with character images.
│   └── test_img.jpg         # Test image for OCR.
├── features.json            # Generated SIFT descriptors.
├── results.json             # Output with character bounding boxes and names.
└── ocr.py                   # Main code.
Requirements
Python 3.x

OpenCV (with contrib modules for SIFT)

NumPy

Install dependencies with:

bash
Copy
pip install opencv-python opencv-contrib-python numpy
Usage
Run the project with:

bash
Copy
python ocr.py --test_img "./data/test_img.jpg" --character_folder_path "./data/characters" --result_saving_directory "./"
The script processes the test image and saves the results to results.json.
