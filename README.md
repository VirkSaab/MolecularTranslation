# Bristol-Myers Squibb – Molecular Translation Kaggle Competition

*Bristol-Myers Squibb is a global biopharmaceutical company working to transform patients' lives through science. Their mission is to discover, develop, and deliver innovative medicines that help patients prevail over serious diseases.*

**InChI**: [International Chemical Identifier](https://en.wikipedia.org/wiki/International_Chemical_Identifier) - machine-readable chemical descriptions

**Goal**: convert images back to the underlying chemical structure annotated as InChI text.

**Evaluation**: Submissions are evaluated on the mean [Levenshtein distance](http://en.wikipedia.org/wiki/Levenshtein_distance) (*basically, a string metric for measuring the difference between two sequences*) between the InChi strings you submit and the ground truth InChi values.

**Note**:
* The images provided (both in the training data as well as the test data) may be rotated to different angles, be at various resolutions, and have different noise levels.
* There are about 4M total images in this dataset. Unzipping the downloaded data will take a non-trivial amount of time.


**data info**:
* `train/` - the training images, arranged in a 3-level (as InChI strings) folder structure by `image_id`
* `test/` - the test images, arranged in the same folder structure as` train/`
* `train_labels.csv` - ground truth InChi labels for the training images
* `sample_submission.csv` - a sample submission file in the correct format


## My approach and Pipeline:
Step 1: Preprocess image to remove noise and thicker lines
Step 2: Use RDKit to create images from InChI strings to train a autoencoder which can reconstruct/refine molecule structures quality.
Step 3: Use reconstructed images to train ImageTranslator to inchi strings using transformers.

Step 2 resources:
1. trying this architecture to implement autoencoder - https://paperswithcode.com/paper/deep-convolutional-autoencoders-for