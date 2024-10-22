import tensorflow as tf

def pgd_attack(model, images, labels, epsilon, alpha, num_iter):
    """
    Projected Gradient Descent Attack.

    Args:
        model: The target model.
        images: Original images.
        labels: True labels.
        epsilon: Maximum perturbation.
        alpha: Step size.
        num_iter: Number of iterations.

    Returns:
        Adversarial examples.
    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    adv_images = tf.identity(images)

    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv_images)
            predictions = model(adv_images, training=False)
            loss = loss_object(labels, predictions)

        # Compute gradients
        gradients = tape.gradient(loss, adv_images)
        # Update adversarial images
        adv_images = adv_images + alpha * tf.sign(gradients)
        # Project back into epsilon-ball
        adv_images = tf.clip_by_value(adv_images, images - epsilon, images + epsilon)
        # Ensure values are within [0,1]
        adv_images = tf.clip_by_value(adv_images, 0.0, 1.0)

    return adv_images
