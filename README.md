
# Housing Price Prediction Project

## Overview
This project aims to predict housing prices based on various features such as average area income, house age, number of rooms, number of bedrooms, and area population. We employ a neural network model built with TensorFlow 2.0 and Keras.

## Theory Topics Covered

### Perceptron Model to Neural Networks
- Understanding the basic concepts of neural networks from single perceptrons to multi-layer perceptrons.
- Deep learning neural network concepts.

### Activation Functions
- Sigmoid, Hyperbolic Tangent (tanh), and Rectified Linear Unit (ReLU).

### Cost Functions
- Quadratic cost function, Cross-entropy loss function for classification.

### Feed Forward Networks
- Building and understanding the forward pass of a neural network.

### BackPropagation
- Gradients, gradient descent, and optimization techniques (e.g., Adam).

### Multi-Class Classification
- Softmax function for multi-class problems.

## Coding Topics Covered

### TensorFlow 2.0 Keras Syntax
- Basic syntax for building and training models.
- Using TensorFlow 2.0 and Keras for model construction.

### ANN with Keras
- Building artificial neural networks with Keras.

### Regression and Classification
- Implementing regression models using neural networks.
- Using neural networks for binary and multi-class classification tasks.

### Exercises for Keras ANN
- Hands-on exercises to reinforce concepts.

### Tensorboard Visualizations
- Visualizing model training progress and performance metrics using TensorBoard.

## Data Preparation

### Data Source
- USA Housing dataset from Kaggle.

### Data Preprocessing
- Loading the dataset.
- Splitting the data into training and testing sets.
- Scaling features using Min-Max normalization.

## Model Construction and Training

### Model Architecture
- A sequential neural network model with multiple hidden layers.
- Activation functions: ReLU for hidden layers and linear output layer for regression.

### Compilation
- Using RMSprop optimizer and Mean Squared Error (MSE) loss function.

### Training
- Training the model for 250 epochs.
- Monitoring training progress with loss visualizations.

## Model Evaluation

### Evaluation
- Evaluating the model on both training and test sets.
- Metrics used: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

### Results Visualization
- Scatter plot comparing true values and model predictions.

## Saving and Loading the Model

### Saving the model
- Save the trained model using Keras native format (.keras).

### Loading the model
- Load the saved model for later use or predictions.

## Directory Structure
```
kotlin
├── data/
│   └── USA_Housing.csv
├── models/
│   └── housing_prediction.keras
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_building.ipynb
│   ├── training_and_evaluation.ipynb
│   └── predictions.ipynb
├── images/
│   ├── loss_curve.png
│   ├── predictions_scatter.png
│   └── model_architecture.png
├── README.md
├── requirements.txt
└── LICENSE
```

## Prerequisites
- Python (>= 3.7)
- TensorFlow 2.0+
- pandas, numpy, scikit-learn, seaborn

## Usage

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Prepare the data:
```bash
python data_preprocessing.py
```

### Train the model:
```bash
python model_building.py
```

### Evaluate the model:
```bash
python training_and_evaluation.py
```

### View predictions:
```bash
python predictions.py
```

## Results
- The model is trained to predict housing prices accurately.
- Evaluation metrics such as MAE and RMSE provide insights into model performance.
- Visualizations (scatter plot) help in understanding the relationship between predicted and actual prices.

## Future Work
- Exploring different model architectures (e.g., deep neural networks with more hidden layers).
- Experimenting with different activation functions and optimizers.
- Implementing more advanced regularization techniques to prevent overfitting.
- Incorporating advanced data preprocessing methods (e.g., feature engineering).

## License
This project is licensed under the MIT License.

## Acknowledgments
- The original dataset was sourced from Kaggle.
- The implementation utilizes TensorFlow 2.0 and Keras.
```

Feel free to ask if you need any more help or have any questions about your project!
