# Adversarial AI Attacks and Defenses with Randomized Smoothing

This project is a Streamlit application that demonstrates adversarial attacks and defenses in machine learning, specifically using the Projected Gradient Descent (PGD) attack and Randomized Smoothing as a defense mechanism. The application is designed to be educational, showcasing how adversarial examples can fool neural networks and how defenses like Randomized Smoothing can improve robustness.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Application Workflow](#application-workflow)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
  - [Adversarial Attack (PGD)](#adversarial-attack-pgd)
  - [Defense Mechanism (Randomized Smoothing)](#defense-mechanism-randomized-smoothing)
  - [Robustness Certification](#robustness-certification)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

## Overview

Adversarial attacks pose a significant threat to machine learning models, especially in critical applications like image recognition, autonomous driving, and cybersecurity. This project illustrates how adversarial examples can deceive a neural network trained on the MNIST dataset and how Randomized Smoothing can be used to defend against such attacks, providing robustness guarantees.

## Features

- **Train a Neural Network**: Train a convolutional neural network (CNN) on the MNIST dataset.
- **Generate Adversarial Examples**: Use the Projected Gradient Descent (PGD) attack to create adversarial images that can fool the neural network.
- **Randomized Smoothing Defense**: Implement Randomized Smoothing to enhance the model's robustness against adversarial attacks.
- **Robustness Certification**: Compute a certified robustness radius within which the model's predictions are guaranteed to remain unchanged.
- **Interactive Interface**: Utilize Streamlit for an interactive user interface, allowing users to experiment with different parameters.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/Pzalms/Adverserial-ai-attacks-defenses.git
cd Adversarial-AI-Attacks-Defenses
```

### Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:

  ```bash
  venv\Scripts\activate
  ```

- On Unix or MacOS:

  ```bash
  source venv/bin/activate
  ```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The `requirements.txt` file should contain the following packages:

```
streamlit
tensorflow
numpy
scipy
```

If the `requirements.txt` file is not present, you can install the packages manually:

```bash
pip install streamlit tensorflow numpy scipy
```

## Usage

### Running the Application

Start the Streamlit application by running:

```bash
streamlit run app.py
```

This will open the application in your default web browser at `http://localhost:8501/`.

### Application Workflow

1. **Train the Model**:
   - Click on the **'Train Model'** button to train the neural network on the MNIST dataset.
   - Training progress and success messages will be displayed.

2. **Generate an Adversarial Example**:
   - Navigate to the **'Adversarial Attack'** section.
   - Use the **'Sample Index'** slider to select an image from the test dataset.
   - Adjust the **'Epsilon'** (maximum perturbation), **'Alpha'** (step size), and **'Number of Iterations'** for the PGD attack.
   - Click on **'Generate Adversarial Example'** to create and display the adversarial image.
   - The model's prediction on the adversarial image will be shown below the image.

3. **Predict with Randomized Smoothing**:
   - Navigate to the **'Randomized Smoothing Defense'** section.
   - Set the **'Number of Samples'** and **'Noise Standard Deviation (Sigma)'**.
   - Click on **'Predict with Randomized Smoothing'** to obtain the smoothed prediction and the certification radius.
   - The certification radius indicates the robustness of the prediction against adversarial perturbations within that radius.

## Project Structure

```
├── app.py
├── attacks.py
├── defense.py
├── model.py
├── requirements.txt
├── README.md
```

- **app.py**: The main Streamlit application file.
- **attacks.py**: Contains the implementation of the PGD adversarial attack.
- **defense.py**: Implements the Randomized Smoothing defense and robustness certification.
- **model.py**: Defines the neural network architecture and compilation.
- **requirements.txt**: Lists the required Python packages.
- **README.md**: Project documentation (this file).

## Technical Details

### Adversarial Attack (PGD)

The Projected Gradient Descent (PGD) attack is an iterative method that generates adversarial examples by maximizing the loss of the model with respect to the input image, while keeping the perturbation within a specified bound (epsilon).

**Parameters**:

- **Epsilon**: The maximum allowed perturbation for each pixel.
- **Alpha**: The step size in each iteration.
- **Number of Iterations**: Total number of iterations to perform the attack.

**Implementation**:

```python
def pgd_attack(model, images, labels, epsilon, alpha, num_iter):
    # Attack implementation
```

### Defense Mechanism (Randomized Smoothing)

Randomized Smoothing is a certified defense that constructs a smoothed classifier by averaging the predictions of a base classifier over Gaussian noise added to the input. This method can provide probabilistic guarantees about the classifier's robustness.

**Parameters**:

- **Number of Samples**: The number of noisy samples to average over.
- **Sigma**: The standard deviation of the Gaussian noise added to the inputs.

**Implementation**:

```python
def randomized_smoothing_predict(model, x, num_samples, sigma):
    # Prediction using randomized smoothing
```

### Robustness Certification

The robustness certification computes a radius within which the model's prediction is guaranteed to remain unchanged with a certain confidence level. This is based on statistical methods and provides a formal guarantee of robustness.

**Implementation**:

```python
def certify_robustness(model, x, num_samples, sigma, alpha=0.001):
    # Compute certified robustness radius
```

## Results

After training the model and performing adversarial attacks and defenses, you can observe:

- **Adversarial Examples**: Images that are visually similar to the original but cause the model to make incorrect predictions.
- **Model Predictions**: How the model's predictions change before and after applying the adversarial attack.
- **Randomized Smoothing Effectiveness**: The smoothed classifier often correctly predicts the label of adversarial examples.
- **Certification Radius**: Indicates the extent to which the model is robust against adversarial perturbations.

## Future Work

- **Extend to Other Datasets**: Apply the methods to more complex datasets like CIFAR-10 or ImageNet.
- **Implement Additional Attacks**: Incorporate attacks like the Carlini & Wagner (C&W) attack or DeepFool.
- **Advanced Defense Mechanisms**: Explore other defense strategies like adversarial training with stronger attacks or feature denoising.
- **Performance Optimization**: Improve the efficiency of randomized smoothing by using optimized sampling techniques.
- **Visualization**: Enhance the application with more visualizations, such as confidence heatmaps or adversarial perturbation maps.

## References

- **Adversarial Attacks**:
  - [Ian J. Goodfellow et al., "Explaining and Harnessing Adversarial Examples"](https://arxiv.org/abs/1412.6572)
  - [Aleksander Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks"](https://arxiv.org/abs/1706.06083)

- **Randomized Smoothing**:
  - [Jeremy M. Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing"](https://arxiv.org/abs/1902.02918)
  - [Microsoft Research Blog, "Randomized Smoothing"](https://www.microsoft.com/en-us/research/blog/randomized-smoothing/)

- **Streamlit Documentation**:
  - [Streamlit Official Documentation](https://docs.streamlit.io/)

- **TensorFlow**:
  - [TensorFlow Official Website](https://www.tensorflow.org/)
