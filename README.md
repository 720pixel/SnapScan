
# SnapScan: Advanced Face Detection and Recognition Tool

This repository provides a comprehensive toolkit for advanced face detection and duplicate face detection using multiple models. The toolkit consists of two main scripts: `facedetector.py` and `dupedetector.py`, which provide a variety of functionalities for processing images and videos to detect faces, enhance images, and remove duplicate faces.

## Features

- **Multiple Face Detection Models**: Supports OpenCV Haar Cascade, YuNet, and RetinaFace models for detecting faces in images and videos.
- **Image Enhancement**: Enhances detected faces using image sharpening techniques for better clarity.
- **Duplicate Face Detection**: Identifies and removes duplicate faces from a set of detected faces.
- **Face Matching**: Matches a given face image against a database of faces to find similar faces.
- **Flexible Input Handling**: Processes both images and videos, with support for specifying time ranges for video processing.
- **Detailed Output**: Saves both the detected face images and the corresponding frames with visual annotations.

## Installation

To get started, clone the repository and install the required dependencies:

```sh
git clone https://github.com/yourusername/snapscan.git
cd snapscan
pip3 install -r requirements.txt
```

## Libraries Used

The toolkit uses the following libraries:

- `opencv-python`: For image and video processing.
- `numpy`: For numerical operations.
- `pillow`: For image handling.
- `tqdm`: For displaying progress bars.
- `retinaface`: For advanced face detection.

These libraries are listed in the `requirements.txt` file and can be installed using pip:

```sh
pip3 install -r requirements.txt
```

## Usage

### Face Detection and Processing

You can use the `facedetector.py` script to detect faces in an image or a video file. It supports various options for customization:

```sh
python3 facedetector.py -i <input_file> -o <output_folder> [options]
```

#### Options:

- `-i, --input`: Path to the input file (image or video).
- `-o, --output`: Path to the output directory (default: `output`).
- `-r, --range`: Time range for video processing (format: `MM.SS-MM.SS` or `SS-SS`).
- `-n, --number`: Maximum number of face outputs (default: unlimited).
- `--debug`: Enable debug mode for detailed process information.
- `-nf, --no-folder`: Save all outputs in the main output directory without creating separate folders for each face.
- `-m, --match`: Match the input image to the database.
- `-dupes, --detect-duplicates`: Detect and remove duplicate faces after processing.
- `-mode`: Mode for face detection, duplicate detection, and matching. Choices: `1` (OpenCV), `2` (YuNet), `3` (RetinaFace). Default: `1`.

### Example Commands:

#### Detect Faces in an Image:

```sh
python3 facedetector.py -i input.jpg -o output_folder -mode 1
```

#### Detect Faces in a Video:

```sh
python3 facedetector.py -i input.mp4 -o output_folder -r 00.30-01.30 -mode 2
```

#### Detect and Remove Duplicate Faces:

```sh
python3 facedetector.py -i input.mp4 -o output_folder -dupes -mode 3
```

### Duplicate Face Detection

The `dupedetector.py` script focuses on detecting and removing duplicate faces from a set of detected faces:

```sh
python3 dupedetector.py -d <output_folder> -mode <mode>
```

#### Options:

- `-d, --detect`: Path to the output folder for duplicate detection.
- `-m, --match`: Path to the image for face matching.
- `-mode`: Mode for face detection and matching. Choices: `1` (OpenCV), `2` (YuNet), `3` (RetinaFace). Default: `1`.

### Example Commands:

#### Detect and Remove Duplicates:

```sh
python3 dupedetector.py -d output_folder -mode 3
```

#### Match Face Against Database:

```sh
python3 dupedetector.py -m query_image.jpg -mode 2
```

## GUi (Beta)
```sh
python3 SnapScan.py or Double Click and open with python
```
![image](https://github.com/user-attachments/assets/a99bbbca-97cf-4bfe-9c7e-09e6a073763a)


## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue to discuss new features or report bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the developers of the OpenCV, RetinaFace, and YuNet libraries for their excellent tools and resources.
