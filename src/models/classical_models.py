"""Classical machine learning models for diabetic retinopathy detection"""

import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from typing import Tuple, List, Dict, Any
from ..utils.config import Config


class SVMClassifier:
    """SVM classifier with RBF kernel for diabetic retinopathy detection"""

    def __init__(self, C: float = None, kernel: str = None, random_state: int = None):
        """
        Initialize SVM classifier.

        Args:
            C: Regularization parameter
            kernel: Kernel type
            random_state: Random seed
        """
        self.C = C or Config.MODEL_PARAMS['svm_c']
        self.kernel = kernel or Config.MODEL_PARAMS['svm_kernel']
        self.random_state = random_state or Config.TRAINING['random_state']
        self.model = SVC(C=self.C, kernel=self.kernel, random_state=self.random_state)
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the SVM model.

        Args:
            X: Feature vectors
            y: Labels

        Returns:
            Training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config.TRAINING['test_size'],
            random_state=self.random_state,
            stratify=y if Config.TRAINING['stratify'] else None
        )

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model.

        Args:
            X: Feature vectors
            y: True labels

        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        y_pred = self.predict(X)

        return {
            'accuracy': accuracy_score(y, y_pred),
            'classification_report': classification_report(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }

    def save_model(self, filepath: str) -> None:
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        joblib.dump(self.model, filepath)

    def load_model(self, filepath: str) -> None:
        """Load trained model from file"""
        self.model = joblib.load(filepath)
        self.is_trained = True


class KNNClassifier:
    """K-Nearest Neighbors classifier for diabetic retinopathy detection"""

    def __init__(self, n_neighbors: int = None, random_state: int = None):
        """
        Initialize KNN classifier.

        Args:
            n_neighbors: Number of neighbors to consider
            random_state: Random seed
        """
        self.n_neighbors = n_neighbors or Config.MODEL_PARAMS['knn_neighbors']
        self.random_state = random_state or Config.TRAINING['random_state']
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the KNN model.

        Args:
            X: Feature vectors
            y: Labels

        Returns:
            Training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config.TRAINING['test_size'],
            random_state=self.random_state,
            stratify=y if Config.TRAINING['stratify'] else None
        )

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model.

        Args:
            X: Feature vectors
            y: True labels

        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        y_pred = self.predict(X)

        return {
            'accuracy': accuracy_score(y, y_pred),
            'classification_report': classification_report(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }

    def save_model(self, filepath: str) -> None:
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        joblib.dump(self.model, filepath)

    def load_model(self, filepath: str) -> None:
        """Load trained model from file"""
        self.model = joblib.load(filepath)
        self.is_trained = True


class ModelComparator:
    """Utility class to compare multiple models"""

    @staticmethod
    def compare_models(X: np.ndarray, y: np.ndarray,
                      models: List[Tuple[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on the same dataset.

        Args:
            X: Feature vectors
            y: Labels
            models: List of (name, model) tuples

        Returns:
            Comparison results
        """
        results = {}

        # Split data once for fair comparison
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config.TRAINING['test_size'],
            random_state=Config.TRAINING['random_state'],
            stratify=y if Config.TRAINING['stratify'] else None
        )

        for name, model in models:
            # Train model
            model.train(X_train, y_train)

            # Evaluate
            metrics = model.evaluate(X_test, y_test)
            results[name] = {
                'accuracy': metrics['accuracy'],
                'model_type': type(model).__name__
            }

        return results

    @staticmethod
    def print_comparison(results: Dict[str, Dict[str, float]]) -> None:
        """Print comparison results in a formatted way"""
        print("\nModel Comparison Results:")
        print("-" * 50)
        for name, metrics in results.items():
            print(f"{name}:")
            print(f"  Type: {metrics['model_type']}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print("-" * 50)