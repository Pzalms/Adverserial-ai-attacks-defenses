import tensorflow as tf
from attacks import fgsm_attack

def adversarial_training(model, x_train, y_train, x_val, y_val, epsilon, epochs=5):
    for epoch in range(epochs):
        # Generate adversarial examples
        x_train_adv = fgsm_attack(model, x_train, y_train, epsilon)
        
        # Combine original and adversarial examples
        x_combined = tf.concat([x_train, x_train_adv], axis=0)
        y_combined = tf.concat([y_train, y_train], axis=0)
        
        # Shuffle combined data
        dataset = tf.data.Dataset.from_tensor_slices((x_combined, y_combined))
        dataset = dataset.shuffle(buffer_size=1024).batch(32)  # Shuffle and batch
        
        # Train the model
        model.fit(dataset, epochs=1, validation_data=(x_val, y_val))
        
        print(f'Epoch {epoch+1}/{epochs} completed.')

    print('Adversarial training completed.')
