import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from model import create_model
from attacks import pgd_attack
from defense import randomized_smoothing_predict, certify_robustness
import numpy as np
import gc

def main():
    st.sidebar.title("Adversarial AI Attacks and Defenses")

    # Sidebar instructions
    st.sidebar.write("""
    **How to Use the App:**

    1. **Train the Model:**
       - Click the 'Train Model' button to train the base classifier on the MNIST dataset.

    2. **Test Adversarial Attack:**
       - Adjust the 'Epsilon for Attack' slider to set the maximum perturbation for generating adversarial images.
       - Set 'Alpha' (step size) and 'Number of Iterations' for the PGD attack.
       - Use the 'Sample Index' slider to select an image from the test set.
       - Click the 'Generate Adversarial Example' button to create and display an adversarial image.
       - The model's prediction for the adversarial image will be shown below the image.

    3. **Randomized Smoothing Prediction:**
       - Set the 'Number of Samples' and 'Noise Standard Deviation' for randomized smoothing.
       - Click the 'Predict with Randomized Smoothing' button to get the smoothed prediction and robustness certification.

    **How It Works:**

    - **Adversarial Attack:** Uses the Projected Gradient Descent (PGD) attack to generate adversarial examples.
    - **Randomized Smoothing:** Improves model robustness by averaging predictions over noisy inputs.
    - **Certification:** Provides a robustness radius within which the prediction is provably unchanged.

    **Epsilon:** Represents the maximum perturbation allowed for the images. Higher values lead to stronger attacks.
    """)

    st.title('Adversarial Attack and Randomized Smoothing Defense System')

    # Load and preprocess data
    @st.cache_data
    def load_data():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
        return (x_train, y_train), (x_test, y_test)

    (x_train, y_train), (x_test, y_test) = load_data()

    # Initialize model
    @st.cache_resource
    def initialize_model():
        model = create_model()
        return model

    model = initialize_model()

    # Train the model
    if st.button('Train Model'):
        with st.spinner('Training model...'):
            model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
        st.success('Model trained successfully.')

    # Generate adversarial example
    st.header('Adversarial Attack')
    sample_index = st.slider('Sample Index', 0, len(x_test)-1, 0)
    epsilon_attack = st.slider('Epsilon (Maximum Perturbation)', 0.0, 0.3, 0.1)
    alpha_attack = st.slider('Alpha (Step Size)', 0.001, 0.1, 0.01)
    num_iter_attack = st.slider('Number of Iterations', 1, 50, 10)
    if st.button('Generate Adversarial Example'):
        image = x_test[sample_index:sample_index+1]
        label = y_test[sample_index:sample_index+1]
        adversarial_image = pgd_attack(model, image, label, epsilon_attack, alpha_attack, num_iter_attack)
        st.image(adversarial_image.numpy().squeeze(), caption='Adversarial Image', use_column_width=True)
        prediction = model.predict(adversarial_image)
        predicted_label = tf.argmax(prediction, axis=1).numpy()[0]
        st.write(f'Predicted Label: {predicted_label}')

    # Randomized smoothing prediction
    st.header('Randomized Smoothing Defense')
    num_samples = st.number_input('Number of Samples', min_value=100, max_value=10000, value=1000, step=100)
    sigma = st.slider('Noise Standard Deviation (Sigma)', 0.01, 1.0, 0.25)
    if st.button('Predict with Randomized Smoothing'):
        image = x_test[sample_index:sample_index+1]
        smoothed_pred = randomized_smoothing_predict(model, image, num_samples, sigma)
        st.write(f'Smoothed Prediction: {smoothed_pred}')

        # Certification
        radius = certify_robustness(model, image, num_samples, sigma)
        st.write(f'Certification Radius: {radius}')

if __name__ == '__main__':
    main()
