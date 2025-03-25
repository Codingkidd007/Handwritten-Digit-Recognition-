import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load MNIST dataset from Scikit-learn
digits = datasets.load_digits()

# Features (flattened 8x8 images) and labels
X = digits.images.reshape((len(digits.images), -1))  # Flatten the 2D images to 1D
y = digits.target

# Split dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (mean = 0, variance = 1) to improve performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an SVM classifier
model = SVC(kernel='linear', C=1)  # You can use 'rbf' or 'poly' kernel for better performance
print("Training the model...")
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Function to visualize predictions
def plot_sample(index):
    plt.imshow(X_test[index].reshape(8, 8), cmap='gray')
    plt.title(f"Predicted: {y_pred[index]}, Actual: {y_test[index]}")
    plt.axis('off')
    plt.show()

# Show 5 random test samples with predictions
for i in np.random.choice(range(len(X_test)), 5, replace=False):
    plot_sample(i)
