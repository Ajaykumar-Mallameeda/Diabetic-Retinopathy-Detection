"""Deep learning models for diabetic retinopathy detection"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from ..utils.config import Config


class CNNModel(tf.keras.Model):
    """Custom CNN model for diabetic retinopathy classification"""

    def __init__(self, hidden_dim: int = None, num_classes: int = None,
                 reg: tf.keras.regularizers = None):
        """
        Initialize custom CNN model.

        Args:
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            reg: Regularizer for layers
        """
        super(CNNModel, self).__init__()

        self.hidden_dim = hidden_dim or Config.MODEL_PARAMS['cnn_hidden_dim']
        self.num_classes = num_classes or Config.MODEL_PARAMS['cnn_num_classes']
        self.reg = reg or tf.keras.regularizers.l2(0.0001)

        # Convolutional layers
        self.cnn1 = tf.keras.layers.Conv2D(
            32, 4, 4, activation='relu',
            kernel_regularizer=self.reg,
            bias_regularizer=self.reg,
            name='conv1'
        )

        self.cnn2 = tf.keras.layers.Conv2D(
            64, 3, 2, activation='relu',
            kernel_regularizer=self.reg,
            bias_regularizer=self.reg,
            name='conv2'
        )

        self.cnn3 = tf.keras.layers.Conv2D(
            128, 3, 2, activation='relu',
            kernel_regularizer=self.reg,
            bias_regularizer=self.reg,
            name='conv3'
        )

        # Dense layers
        self.l1 = tf.keras.layers.Dense(
            self.hidden_dim, activation='relu',
            kernel_regularizer=self.reg,
            bias_regularizer=self.reg,
            name='dense1'
        )

        self.l2 = tf.keras.layers.Dense(
            self.hidden_dim, activation='relu',
            kernel_regularizer=self.reg,
            bias_regularizer=self.reg,
            name='dense2'
        )

        self.l3 = tf.keras.layers.Dense(
            self.hidden_dim, activation='relu',
            kernel_regularizer=self.reg,
            bias_regularizer=self.reg,
            name='dense3'
        )

        # Output layer
        self.out = tf.keras.layers.Dense(
            self.num_classes,
            kernel_regularizer=self.reg,
            activation='softmax',
            bias_regularizer=self.reg,
            name='output'
        )

    def call(self, obs, training=None):
        """Forward pass"""
        x = self.cnn1(obs)
        x = self.cnn2(x)
        fin_cnn_layer = self.cnn3(x)
        x = tf.keras.layers.Flatten()(fin_cnn_layer)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return self.out(x), fin_cnn_layer


class MobileNetV2Transfer(tf.keras.Model):
    """Transfer learning model using MobileNetV2"""

    def __init__(self, input_shape: Tuple[int, int, int],
                 hidden_dim: int = None, num_classes: int = None,
                 fine_tune_at: int = 100):
        """
        Initialize MobileNetV2 transfer learning model.

        Args:
            input_shape: Input image shape
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            fine_tune_at: Layer index from which to fine-tune
        """
        super(MobileNetV2Transfer, self).__init__()

        self.hidden_dim = hidden_dim or Config.MODEL_PARAMS['cnn_hidden_dim']
        self.num_classes = num_classes or Config.MODEL_PARAMS['cnn_num_classes']
        self.reg = tf.keras.regularizers.l2(0.0001)

        # Load pretrained MobileNetV2
        self.pretrained = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )

        # Freeze early layers
        for layer in self.pretrained.layers[:fine_tune_at]:
            layer.trainable = False

        # Dense layers
        self.l1 = tf.keras.layers.Dense(
            self.hidden_dim, activation='relu',
            kernel_regularizer=self.reg,
            bias_regularizer=self.reg
        )

        self.l2 = tf.keras.layers.Dense(
            self.hidden_dim, activation='relu',
            kernel_regularizer=self.reg,
            bias_regularizer=self.reg
        )

        self.l3 = tf.keras.layers.Dense(
            self.hidden_dim, activation='relu',
            kernel_regularizer=self.reg,
            bias_regularizer=self.reg
        )

        # Output layer
        self.out = tf.keras.layers.Dense(
            self.num_classes,
            kernel_regularizer=self.reg,
            activation='softmax',
            bias_regularizer=self.reg
        )

    def call(self, obs, training=None):
        """Forward pass"""
        fin_cnn_layer = self.pretrained(obs)
        x = tf.keras.layers.Flatten()(fin_cnn_layer)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return self.out(x), fin_cnn_layer


class GradCAM:
    """Grad-CAM implementation for model interpretability"""

    def __init__(self, model: tf.keras.Model):
        """
        Initialize Grad-CAM.

        Args:
            model: Trained model
        """
        self.model = model

    def __call__(self, x_value: tf.Tensor,
                 should_resize: bool = True,
                 three_dims: bool = False) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            x_value: Input image tensor
            should_resize: Whether to resize heatmap to input size
            three_dims: Whether to return 3D heatmap

        Returns:
            Grad-CAM heatmap
        """
        with tf.GradientTape() as tape:
            out, c_layer = self.model(x_value, training=False)
            y = tf.reduce_max(out, 1)

        # Get gradients
        grad = tape.gradient(y, c_layer)[0]
        output = c_layer[0]

        # Global average pooling
        weights = np.mean(grad, axis=(0, 1))
        grad_cam = np.zeros(output.shape[0:2], dtype=np.float32)

        # Weighted combination
        for i, w in enumerate(weights):
            grad_cam += w * output[:, :, i]

        # ReLU
        grad_cam = np.maximum(grad_cam, 0)

        # Resize if needed
        if should_resize:
            grad_cam = grad_cam / np.max(grad_cam)  # Normalize to [0, 1]
            grad_cam = np.squeeze(tf.image.resize(
                np.expand_dims(np.expand_dims(grad_cam, 0), 3),
                x_value.shape[1:3],
                preserve_aspect_ratio=False,
                antialias=False
            ))

        # Convert to 3D if requested
        if three_dims:
            grad_cam = np.expand_dims(grad_cam, axis=2)
            grad_cam = np.tile(grad_cam, [1, 1, 3])

        return grad_cam


class Trainer:
    """Training utilities for deep learning models"""

    def __init__(self, model: tf.keras.Model,
                 learning_rate: float = None):
        """
        Initialize trainer.

        Args:
            model: Model to train
            learning_rate: Learning rate
        """
        self.model = model
        self.learning_rate = learning_rate or Config.MODEL_PARAMS['cnn_learning_rate']

        # Loss and optimizer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy'
        )
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy'
        )

    @tf.function
    def train_step(self, images: tf.Tensor, labels: tf.Tensor) -> None:
        """Perform one training step"""
        with tf.GradientTape() as tape:
            predictions, _ = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images: tf.Tensor, labels: tf.Tensor) -> None:
        """Perform one validation step"""
        predictions, _ = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def train(self, train_data, val_data, epochs: int,
              log_dir: str = None) -> dict:
        """
        Train the model.

        Args:
            train_data: Training data generator
            val_data: Validation data generator
            epochs: Number of epochs
            log_dir: Directory for TensorBoard logs

        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': []
        }

        # Setup TensorBoard if requested
        if log_dir:
            train_summary_writer = tf.summary.create_file_writer(
                f'{log_dir}/train'
            )
            test_summary_writer = tf.summary.create_file_writer(
                f'{log_dir}/test'
            )

        for epoch in range(epochs):
            # Reset metrics
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            # Train on one batch
            images, labels = next(train_data)
            images = tf.convert_to_tensor(images, dtype=tf.float32)
            labels = tf.convert_to_tensor(labels, dtype=tf.int32)
            self.train_step(images, labels)

            # Validate on one batch
            test_images, test_labels = next(val_data)
            test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)
            test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int32)
            self.test_step(test_images, test_labels)

            # Record metrics
            history['train_loss'].append(self.train_loss.result().numpy())
            history['train_accuracy'].append(
                self.train_accuracy.result().numpy() * 100
            )
            history['test_loss'].append(self.test_loss.result().numpy())
            history['test_accuracy'].append(
                self.test_accuracy.result().numpy() * 100
            )

            # Log to TensorBoard
            if log_dir:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                    tf.summary.scalar(
                        'accuracy', self.train_accuracy.result(), step=epoch
                    )

                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', self.test_loss.result(), step=epoch)
                    tf.summary.scalar(
                        'accuracy', self.test_accuracy.result(), step=epoch
                    )

            # Print progress
            template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}%, ' \
                      'Test Loss: {:.4f}, Test Accuracy: {:.2f}%'
            print(template.format(
                epoch + 1,
                self.train_loss.result(),
                self.train_accuracy.result() * 100,
                self.test_loss.result(),
                self.test_accuracy.result() * 100
            ))

        return history

    def plot_training_history(self, history: dict, save_path: str = None) -> None:
        """
        Plot training history.

        Args:
            history: Training history dictionary
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['test_loss'], label='Test Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Accuracy plot
        ax2.plot(history['train_accuracy'], label='Train Accuracy')
        ax2.plot(history['test_accuracy'], label='Test Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()