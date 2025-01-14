Pixel Play’25 Challenge
Overview
This project focuses on classifying animal images into 50 distinct classes using ResNet101, a deep convolutional neural network renowned for its robust feature extraction capabilities. The goal was to achieve high accuracy while minimizing computational costs by leveraging pretrained weights and employing advanced training strategies such as freezing and unfreezing certain layers.

The implementation of this project was carefully structured to ensure optimal performance and computational efficiency. Below is a breakdown of the vital aspects of the code and their roles:
1.	Data Loading and Preprocessing:
o	Custom Dataset Creation: Images were loaded into a custom dataset that returned data in NumPy format. This dataset enabled random access during every epoch and was loaded in batches of size 16.
o	Data Splitting: 15% of the data was reserved for evaluation, and the remaining was used for training. Test data was handled separately with randomized transformations.
o	Image Transformations:
	Random cropping between 70%-100% and resizing to 300x300.
	Random horizontal flipping and rotation between ±30 degrees to improve generalization.
	Pixel values were normalized using the mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225].
2.	Model Development:
o	Architecture:
	ResNet101 was chosen for its advanced residual connections that mitigate the vanishing gradient problem in deep networks.
	The network includes initial convolutional and max-pooling layers, followed by residual blocks and a fully connected layer for classification.
o	Pretrained Weights: ImageNet-pretrained weights were used to initialize the network, leveraging its prior knowledge for feature extraction.
3.	Training Strategy:
o	Loss Function: Cross-entropy loss was employed to measure classification accuracy.
o	Optimizer: Stochastic Gradient Descent (SGD) was used for weight updates.
o	Learning Rate Scheduler:
	A MultiStepLR scheduler was implemented to dynamically adjust the learning rate.
	The learning rate was reduced by a factor of 0.9 after the 16th and 21st epochs based on loss graph observations.
o	Freezing Layers:
	All layers except the last convolutional block and fully connected head were frozen during initial training.
	This allowed the network to retain learned features while adapting higher layers to the specific dataset.
4.	Explainability and Robustness:
o	Random transformations ensured that the model learned robust features by exposing it to varied orientations and perspectives.
o	A voting mechanism was used during testing to enhance prediction reliability by averaging multiple predictions per image.
5.	Output Generation:
o	Final predictions were saved in a CSV format for evaluation and submission.
6.	Hardware:
o	The training process utilized Kaggle’s P100 GPU, which provided sufficient computational power to train the model efficiently despite occasional kernel crashes.
Vital Implementation Highlights:
•	Freezing and Unfreezing Layers: This technique was pivotal in balancing computational efficiency and performance, enabling faster convergence without sacrificing accuracy.
•	Learning Rate Scheduling: The adaptive adjustment of the learning rate ensured stable and effective training over 25 epochs.
•	Data Augmentation: Random cropping, flipping, and rotation significantly improved the generalization capability of the model.
•	Prediction Voting Mechanism: This approach minimized the impact of outlier predictions and boosted the reliability of the final results.

Results
•	Seen Dataset (15% evaluation split): Achieved 91-93% accuracy.
•	Unseen Dataset (test): Achieved 80% accuracy.

Challenges and Solutions
1.	Zero-Shot Learning:
o	Difficulty in finding reliable resources.
o	Focused on transfer learning techniques instead.
2.	Computation Power:
o	Training on Kaggle GPU P100 took 2-3 hours, causing delays.
o	Optimized model configurations to reduce runtime.
3.	Imbalanced Dataset:
o	Addressed misclassification (e.g., cows as Dalmatians) using data augmentation techniques.
4.	Accuracy Fluctuations:
o	Debugged carefully after updates to maintain performance.
5.	Batch Size Optimization:
o	Balanced accuracy and computation speed through experiments.

Learning Outcomes
•	Gained hands-on experience with Kaggle's computational resources.
•	Mastered fine-tuning of pretrained models for specific tasks.
•	Developed expertise in data augmentation and learning rate scheduling.
•	Enhanced debugging skills and understanding of model behavior.
•	Applied advanced OOP concepts to create modular and reusable code structures.

Future Work
•	Explore zero-shot learning techniques.
•	Experiment with other architectures like EfficientNet.
•	Address dataset imbalance with advanced augmentation strategies.
•	Optimize computational efficiency further for faster training.

