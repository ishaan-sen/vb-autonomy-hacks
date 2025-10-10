## Introduction

This project is a prototype for autonomous convoy navigation in rapidly changing environments. The goal is to enable safe and efficient routing of convoys during emergency and military operations. Key contributions include:

- **Three‑modality threat detection** (vision, thermal and sound). Instead of relying on a single sensor, the system fuses information from aerial images, thermal cameras and audio signals to detect threats such as fires, floods, traffic incidents and gunshots.
- **Two‑stage swarm‑guided navigation** consisting of static environment scanning and dynamic route planning. Drone swarms first map the environment and detect threats; then a route planner adapts the convoy path in real time based on the detected hazards.
- **Human‑centered decision making** where a human operator remains “in the loop” and can select among alternate routes.

The system was tested on the Nashville, Tennessee road network (approximately 34 000 nodes and 113 000 edges) and uses a combination of synthetic and real data: roughly 5 500 aerial images and ~100 threat polygons. AWS services such as IoT Core/Greengrass, S3, SageMaker, Bedrock, Rekognition, Lambda and DynamoDB provide the cloud infrastructure.

## Data Acquisition and Generation

### Map and Threat Data

The project uses OpenStreetMap (OSM) data to build a road network for Nashville. The Streamlit UI loads an OSM XML map file (`data/map`) using OSMnx and converts it into a network graph. Polygons representing path obstructions and threats are loaded either from an S3 bucket or from a local CSV file (`data/threats.csv`). Each row of the CSV contains a bounding box (`lon_min`, `lat_min`, `lon_max`, `lat_max`) along with a `stem` name, `class` and `class_id`.

Hostile areas beyond explicit threats are specified in code. For example, the UI defines polygons for an enemy patrol area in the northwest, an adversary stronghold, a sniper overwatch position, a hostile checkpoint and a high‑risk corridor. These areas incur extra penalty during route planning.

### Synthetic Audio Dataset

The `generator/audio_generator/generate_audio_dataset.py` script synthesizes an audio dataset for anomaly detection. Default parameters define a 16 kHz sampling rate, 20 s clip length and the number of streets/clips per split. Each street profile has a base tone frequency, ambient pink noise power and frequency modulation depth. A Poisson process determines how many “non‑anomalous” events (e.g., car passing, dog bark, horn) occur in a clip. If the street is marked anomalous, additional gunshot or impact events are injected at random start times and signal‑to‑noise ratios. Clips are sliced into 1.5 s segments with 0.5 s hop; each segment is labelled as anomalous if it overlaps any injected event. Metadata, events and segment labels are written to CSV files when running `generate_audio_dataset.py`.

### Synthetic Image and Thermal Data

The `generator/image_generator` directory contains scripts to augment image data:

- **Satellite image retrieval:** `query_satellite_map.py` uses Google Earth Engine to download high‑resolution NAIP or Sentinel‑2 imagery for a specific bounding box. The script clips the image to the exact region.
- **Tile splitting:** `split_statellite_map.py` splits a large TIFF map into fixed‑size tiles (100×100 px by default) without georeferencing. This produces manageable patches for data augmentation or inference.
- **Incident image generation:** `generate_incident.py` sends each input image to Google’s Gemini vision model and requests prompts such as “Add flooding to this area” or “Create traffic incident in this region”. Generated images are saved with descriptive filenames. The script checks whether an image/prompt combination has already been generated before calling the API.
- **Thermal and fire simulation:** `generate_thermal_image.py` selects a subset of images and applies thermal colorization (inferno, jet, turbo, etc.) and optionally adds a Gaussian hotspot overlay to simulate fire. The script uses either OpenCV or Matplotlib colormaps. A thermal overlay is blended with the original image, and a Gaussian hotspot is drawn with random center and sigma. The user can specify how many images to convert to thermal only and how many to thermal+fire; the script writes them into separate output folders.

### Threat Dataset

A `data/threats.csv` file lists numerous obstructions and threats in the navigation region. Each row records the bounding box and class (e.g., `enemy_presence`, `road destruction`) for a threat. These polygons are used by the route planner to remove blocked edges and compute risk penalties.

## Threat Detection Models

### Audio Event Detection

The `detector/audio_synth/audio_event` module implements a convolutional recurrent neural network (CRNN) for audio anomaly detection:

1. **Configuration:** `config.py` defines the sample rate, number of mel filters, log‑mel spectrogram parameters, augmentation settings and threshold hyper‑parameters for hysteresis and smoothing.
2. **Dataset loading:** `dataset.py` reads the segments CSV and returns audio segments as tensors. Augmentations include random gain and additive noise. Log‑mel spectrograms are computed using `torchaudio` transforms. The collate function pads variable‑length spectrograms for batching.
3. **Model:** `models/crnn.py` defines a CRNN architecture. A sequence of convolutional blocks with batch normalization and pooling feeds into a bidirectional GRU and a linear head. The final output is a sigmoid probability representing the presence of an event.
4. **Post‑processing:** `postprocess.py` smooths probability sequences using a moving average, applies hysteresis thresholding, merges close events and removes events shorter than a minimum duration. The thresholds are configurable via `config.py`.
5. **Inference and evaluation:** `test_event_detector.py` loads the trained checkpoint, passes audio segments through the model, writes per‑segment probabilities to a CSV and applies post‑processing to produce event intervals. The script writes event summaries per clip and optionally per street. Evaluation metrics such as PR‑AUC, ROC‑AUC and best F1 are provided in `metrics.py`.

### Vision Anomaly Detection

The primary vision classifier is based on AWS Rekognition's Custom Labels model for fast, accurate classification. In scenarios where cloud connectivity is disrupted, a local secondary vision classifier is implemented as redundancy in `detector/image_classifier`:

1. **Configuration:** `config.py` defines paths, hyper‑parameters and a class list. The model uses ConvNeXt‑Tiny by default and trains for 200 epochs with a learning rate of 5×10⁻⁵ and batch size 256.
2. **Dataset:** The `datasets/pickle_dataset.py` module loads images from pickled files. Each pickle contains RGB images and labels; the `make_splits` function stratifies them into train/val/test sets. Training transforms include random resized crops, flips, rotations, colour jitter and Gaussian blur. Evaluation uses a simple resize and tensor conversion.
3. **Model builder:** `models/classifier.py` wraps several pre‑trained models. For ConvNeXt‑Tiny, the final linear layer is replaced to match the number of classes. The script also supports ViT or timm models.
4. **Training and evaluation:** `sageMaker/vision.py` implements the training loop used on SageMaker. It supports focal loss, class weighting, exponential moving average (EMA) and data augmentations such as MixUp and CutMix. A logit bias can be tuned to improve recall for a specific class (e.g., traffic incidents) by adding a scalar to the logits and selecting the bias that maximizes macro‑F1 on the validation set. The script computes metrics for each epoch and uses early stopping.
5. **Inference:** `detector/image_classifier/eval.py` loads the best checkpoint and runs inference on the test split, printing classification metrics. `finetune_generated_images.py` demonstrates how to fine‑tune the pre‑trained model on a smaller generated dataset: it loads the 5‑class classifier, slices its logits to keep only the three classes of interest, stratifies the generated images into train/val/test splits and trains for 20 epochs. The script selects the indices of the ‘normal’, ‘flooded_areas’ and ‘traffic_incident’ classes in the original classifier, then fine‑tunes all model parameters with a low learning rate.

### Thermal Anomaly Detection

`detector/thermal_classifier/thermal_classifier.py` trains a model to classify thermal images as “normal” or “fire”. The script reads images from a `generated_thermal_images` directory; uses a simple resize and normalization transform; and splits the dataset into train/val/test using stratified sampling. A custom convolutional neural network (`DeeperCNN`) contains six convolution layers with batch normalization and max pooling followed by fully connected layers. The model is trained with cross‑entropy loss and the best model (highest validation accuracy) is saved. At the end of training, the script prints a classification report and confusion matrix and saves the final weights.

### Performance Metrics

Summary metrics for each modality:

|Modality|Dataset size|Classes|F1|Precision|Recall|Comments|
|---|---|---|---|---|---|---|
|**Vision**|5 138 images with four anomaly classes (collapsed building, fire, flooding, traffic incident)|5|≥ 0.94 across classes|0.95–1.00|0.93–1.00|Model trained with AWS Rekognition and the AIDER dataset; high performance achieved on all obstruction types.|
|**Sound**|2 208 audio segments with two classes (fire, normal)|2|1.0|1.0|1.0|Synthetic dataset and CRNN model yield perfect scores on the test set.|
|**Thermal**|1 000 thermal images with two classes (fire vs normal)|2|0.990 (Fire) / 0.9901 (Normal)|1.0 (Fire) / 0.9804 (Normal)|0.98 (Fire) / 1.0 (Normal)|DeeperCNN classifier achieves very high accuracy on simulated thermal imagery.|