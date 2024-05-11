# Yolo v4 with Custom Loss Functions

## Project Description

This repository contains modifications to the Yolo v4 object detection model, where the loss function has been changed to use Normalized Wasserstein Distance and Bhattacharyya Distance. The goal of this project is to evaluate the performance of these distance metrics in improving the accuracy and efficiency of the model.

## Folder Structure

- `cfg/`: Configuration files for model parameters.
- `data/`: Dataset used for training and testing the model.
- `images/`: Miscellaneous images related to the project.
- `models/`: Model definitions and training specifications.
- `utils/`: Utility scripts for data preprocessing, performance evaluation etc.
- `weights/`: Pre-trained weights and final model weights after training.
- `detect.py`: Script for performing object detection with the trained model.
- `requirements.txt`: Python dependencies required for the project.
- `test.py`: Script for testing the model performance.
- `train.py`: Script for training the model.

## Installation

To set up the project environment:

```bash
git clone https://github.com/yourusername/your-repository-name.git
cd your-repository-name
pip install -r requirements.txt
```

## Training

```
python train.py --device 0 --batch-size 16 --img 640 640 --data coco.yaml --cfg cfg/yolov4-pacsp.cfg --weights '' --name yolov4-pacsp
```

## Testing

```
python test.py --img 640 --conf 0.001 --batch 8 --device 0 --data coco.yaml --cfg cfg/yolov4-pacsp.cfg --weights weights/yolov4-pacsp.pt
```

## Citation

```
@article{bochkovskiy2020yolov4,
  title={{YOLOv4}: Optimal Speed and Accuracy of Object Detection},
  author={Bochkovskiy, Alexey and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2004.10934},
  year={2020}
}
```

```
@inproceedings{wang2020cspnet,
  title={{CSPNet}: A New Backbone That Can Enhance Learning Capability of {CNN}},
  author={Wang, Chien-Yao and Mark Liao, Hong-Yuan and Wu, Yueh-Hua and Chen, Ping-Yang and Hsieh, Jun-Wei and Yeh, I-Hau},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={390--391},
  year={2020}
}
```

## Acknowledgements

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

## Contributing
Contributions to this project are welcome! You can contribute in the following ways:

- Submitting a pull request with improvements to code or documentation.
- Reporting issues or suggesting new features.
- Improving the existing models or training with different datasets.

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

```vbnet
This template provides a comprehensive overview of what should be included in the README for clarity and effectiveness in communicating the project's purpose and usage. Adjust the links and repository details according to your actual GitHub repository setup.
```


