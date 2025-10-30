"""
Neural Networks Visualization Script

This script generates plots for activation functions and loss functions
used in neural networks. The plots are saved to the res/neural-networks/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def create_output_directory():
    """Create output directory for plots if it doesn't exist."""
    output_dir = "../../res/neural-networks"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_activation_functions(output_dir):
    """Generate separate plots for each activation function."""

    # Activation Functions
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def tanh(x):
        return np.tanh(x)

    def relu(x):
        return np.maximum(0, x)

    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def swish(x):
        return x * sigmoid(x)

    def linear(x):
        return x

    def softmax_single(x):
        # For single input, softmax approaches sigmoid
        return sigmoid(x)

    # Plot activation functions - each in separate plot
    x = np.linspace(-5, 5, 1000)

    activations = [
        ("Sigmoid", sigmoid, "#2563eb"),  # Blue
        ("Tanh", tanh, "#2563eb"),  # Blue
        ("ReLU", relu, "#2563eb"),  # Blue
        ("Leaky ReLU", leaky_relu, "#2563eb"),  # Blue
        ("ELU", elu, "#2563eb"),  # Blue
        ("Swish", swish, "#2563eb"),  # Blue
        ("Linear", linear, "#2563eb"),  # Blue
        ("Softmax (single)", softmax_single, "#2563eb"),  # Blue
    ]

    for name, func, color in activations:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        y = func(x)
        ax.plot(x, y, linewidth=2, color=color)
        ax.set_title(f"{name} Activation Function", fontsize=14, fontweight="bold")
        ax.set_xlabel("Input (x)")
        ax.set_ylabel("Output f(x)")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)

        # Set y-limits for better visualization
        if name == "Linear":
            ax.set_ylim(-5, 5)
        elif name in ["Sigmoid", "Tanh", "Softmax (single)"]:
            ax.set_ylim(-0.2, 1.2)
        else:
            ax.set_ylim(-2, 5)

        plt.tight_layout()
        # Save with filename based on function name
        filename = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("__", "_")
        )
        plt.savefig(
            f"{output_dir}/activation_{filename}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"✓ {name} activation plot saved successfully!")


def plot_regression_losses(output_dir):
    """Generate plots for regression loss functions (MSE and MAE)."""

    def mse_loss(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean((y_true - y_pred) ** 2)

    def mae_loss(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred))

    # For MSE and MAE, plot vs prediction error
    error_range = np.linspace(-2, 2, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    mse_vals = [mse_loss([0], [e]) for e in error_range]
    ax1.plot(error_range, mse_vals, linewidth=2, color="#2563eb", label="MSE")
    ax1.set_title("Mean Squared Error (MSE)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Prediction Error")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    mae_vals = [mae_loss([0], [e]) for e in error_range]
    ax2.plot(error_range, mae_vals, linewidth=2, color="#2563eb", label="MAE")
    ax2.set_title("Mean Absolute Error (MAE)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Prediction Error")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/regression_losses.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Regression losses plot saved successfully!")


def plot_binary_cross_entropy(output_dir):
    """Generate plots for binary cross entropy loss."""

    def bce_loss(y_true, y_pred, eps=1e-15):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    prob_range = np.linspace(0.001, 0.999, 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # BCE for y_true = 1
    bce_1 = [bce_loss([1], [p]) for p in prob_range]
    ax1.plot(prob_range, bce_1, linewidth=2, color="#2563eb", label="y_true = 1")
    ax1.set_title("Binary Cross Entropy (y_true = 1)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Predicted Probability")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # BCE for y_true = 0
    bce_0 = [bce_loss([0], [p]) for p in prob_range]
    ax2.plot(prob_range, bce_0, linewidth=2, color="#2563eb", label="y_true = 0")
    ax2.set_title("Binary Cross Entropy (y_true = 0)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/binary_cross_entropy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Binary cross entropy plot saved successfully!")


def plot_categorical_cross_entropy(output_dir):
    """Generate plots for categorical cross entropy loss."""

    def categorical_ce_loss(y_true, y_pred, eps=1e-15):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred))

    # Categorical Cross Entropy (simplified for 3 classes)
    classes = 3
    prob_range = np.linspace(0.001, 0.999, 100)

    fig, axes = plt.subplots(1, classes, figsize=(18, 6))

    for true_class in range(classes):
        losses = []
        for p in prob_range:
            # Create one-hot true label
            y_true = np.zeros(classes)
            y_true[true_class] = 1

            # Create prediction with p probability for true class, rest distributed
            y_pred = np.full(classes, (1 - p) / (classes - 1))
            y_pred[true_class] = p

            loss = categorical_ce_loss(y_true, y_pred)
            losses.append(loss)

        axes[true_class].plot(prob_range, losses, linewidth=2, color="#2563eb")
        axes[true_class].set_title(
            f"Categorical CE (True Class = {true_class})",
            fontsize=14,
            fontweight="bold",
        )
        axes[true_class].set_xlabel("Predicted Probability for True Class")
        axes[true_class].set_ylabel("Loss")
        axes[true_class].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/categorical_cross_entropy.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("✓ Categorical cross entropy plot saved successfully!")


def main():
    """Main function to generate all plots."""
    print("Generating Neural Network Visualization Plots...")
    print("=" * 50)

    # Set matplotlib style
    plt.style.use("default")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12

    # Create output directory
    output_dir = create_output_directory()

    # Generate all plots
    plot_activation_functions(output_dir)
    plot_regression_losses(output_dir)
    plot_binary_cross_entropy(output_dir)
    plot_categorical_cross_entropy(output_dir)

    print("=" * 50)
    print(f"Plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()

