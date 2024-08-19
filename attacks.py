import tensorflow as tf

def fgsm_attack(model, images, labels, epsilon):
    images = tf.convert_to_tensor(images)
    labels = tf.convert_to_tensor(labels)
    
    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
    
    gradients = tape.gradient(loss, images)
    
    # Debugging statements
    print("Loss:", loss.numpy())
    print("Gradients min:", tf.reduce_min(gradients).numpy())
    print("Gradients max:", tf.reduce_max(gradients).numpy())
    
    perturbations = epsilon * tf.sign(gradients)
    adversarial_images = images + perturbations
    
    # Debugging statements
    print("Perturbations min:", tf.reduce_min(perturbations).numpy())
    print("Perturbations max:", tf.reduce_max(perturbations).numpy())
    print("Adversarial Images min:", tf.reduce_min(adversarial_images).numpy())
    print("Adversarial Images max:", tf.reduce_max(adversarial_images).numpy())
    
    return tf.clip_by_value(adversarial_images, 0, 1)
