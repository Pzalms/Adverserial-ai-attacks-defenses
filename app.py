import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from model import create_model
from attacks import fgsm_attack
from defense import adversarial_training
import numpy as np

def main():
    st.sidebar.title("Adversarial AI Attacks and Defenses")
    
    # Sidebar instructions
    st.sidebar.write("""
    **How to Use the App:**

    1. **Train with Adversarial Defense:**
       - Adjust the 'Epsilon' slider to set the perturbation strength for adversarial examples.
       - Click the 'Train with Adversarial Defense' button to train the model with adversarial examples.

    2. **Test Adversarial Attack:**
       - Adjust the 'Epsilon for Attack' slider to set the perturbation strength for generating adversarial images.
       - Use the 'Sample Index' slider to select an image from the test set.
       - Click the 'Test Adversarial Attack' button to generate and display an adversarial image.
       - The model's prediction for the adversarial image will be shown below the image.

    **How It Works:**

    - **Adversarial Attack:** The app uses the Fast Gradient Sign Method (FGSM) to generate adversarial examples by adding perturbations to the input images.
    - **Adversarial Defense:** The app trains a neural network on a combination of original and adversarial examples to make it more robust against attacks.
    - **Model:** A simple neural network is used for demonstration, and it is trained on the MNIST dataset.

    **Epsilon:** Represents the magnitude of perturbations added to the images. Higher values lead to stronger attacks.
    """)
    
    st.title('Adversarial Attack and Defense System')

    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train[..., tf.newaxis], x_test[..., tf.newaxis]

    # Initialize model
    model = create_model()

    # Store session state
    if 'training' not in st.session_state:
        st.session_state.training = False
    if 'epsilon' not in st.session_state:
        st.session_state.epsilon = 0.1
    if 'sample_index' not in st.session_state:
        st.session_state.sample_index = 0
    if 'show_train_slider' not in st.session_state:
        st.session_state.show_train_slider = False
    if 'show_attack_slider' not in st.session_state:
        st.session_state.show_attack_slider = False

    # Training
    if st.button('Train with Adversarial Defense'):
        st.session_state.show_train_slider = True
        st.session_state.show_attack_slider = False
        st.session_state.training = True
        
    if st.session_state.show_train_slider:
        epsilon = st.slider('Epsilon for Training', 0.0, 1.0, st.session_state.epsilon, key='train_epsilon_slider')
        st.session_state.epsilon = epsilon

        if st.session_state.training:
            with st.spinner('Training model...'):
                adversarial_training(model, x_train, y_train, x_test, y_test, epsilon)
            st.write("Model trained with adversarial defense.")
            st.session_state.training = False

    if st.session_state.training:
        st.text('Training in progress...')
    
    # Test attack
    if st.button('Test Adversarial Attack'):
        st.session_state.show_train_slider = False
        st.session_state.show_attack_slider = True

    if st.session_state.show_attack_slider:
        epsilon = st.slider('Epsilon for Attack', 0.0, 1.0, st.session_state.epsilon, key='attack_epsilon_slider')
        sample_index = st.slider('Sample Index', 0, len(x_test)-1, st.session_state.sample_index, key='sample_index_slider')
        st.session_state.epsilon = epsilon
        st.session_state.sample_index = sample_index
        
        image = x_test[sample_index:sample_index+1]
        label = y_test[sample_index:sample_index+1]
        
        adversarial_image = fgsm_attack(model, image, label, epsilon)
        
        # Convert tensor to numpy array
        adversarial_image_np = adversarial_image.numpy()[0, ..., 0]
        
        # Display the image using Streamlit
        st.image(adversarial_image_np, caption='Adversarial Image', use_column_width=True)
        
        prediction = model.predict(adversarial_image)
        st.write(f'Prediction: {tf.argmax(prediction, axis=1).numpy()[0]}')

if __name__ == '__main__':
    main()
