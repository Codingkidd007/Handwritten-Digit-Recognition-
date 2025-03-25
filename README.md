---

## **🖊 Handwritten Digit Recognition (MNIST) using SVM**  

This project implements a **Handwritten Digit Recognition System** using the **MNIST dataset** and a **Support Vector Machine (SVM)** classifier. Unlike deep learning models, this approach uses **traditional machine learning** techniques to achieve high accuracy with lower computational cost.  

---

## **📌 Features**
✅ Load and preprocess the **MNIST dataset** (handwritten digits from 0-9).  
✅ **Flatten and normalize** the image data for better model performance.  
✅ Train an **SVM classifier** (Support Vector Machine) for digit recognition.  
✅ Evaluate the model using **accuracy score**.  
✅ **Visualize predictions** on sample images.  

---

## **🚀 Technologies Used**
- Python 🐍  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## **📂 Dataset**
The **MNIST dataset** is a collection of **8×8 grayscale images** of handwritten digits (0–9). Each image is **flattened into a 64-pixel feature vector** for training.  

---

## **📜 Installation & Usage**
### **1️⃣ Install Dependencies**
```bash
pip install numpy matplotlib scikit-learn
```

### **2️⃣ Run the Code**
```bash
python mnist_svm.py
```

### **3️⃣ Sample Output**
- The model **trains on MNIST data**.  
- It prints the **test accuracy**.  
- It displays **predictions on sample images**.  

---

## **📌 Code Overview**
```python
# Load dataset
digits = datasets.load_digits()
X = digits.images.reshape((len(digits.images), -1))  # Flatten images
y = digits.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM classifier
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
```

---

## **📊 Results**
✅ **High Accuracy**: The SVM classifier achieves **~98% accuracy** on the MNIST dataset.  
✅ **Efficient Computation**: No need for deep learning libraries like TensorFlow.  
✅ **Clear Visualization**: Displays sample predictions on test images.  

---

## **🎯 Future Enhancements**
🚀 Improve accuracy using **different SVM kernels** (e.g., 'rbf', 'poly').  
🚀 Apply **PCA (Principal Component Analysis)** for dimensionality reduction.  
🚀 Deploy the model as a **web app** using Flask or Streamlit.  

---

## **📜 License**
This project is open-source under the **MIT License**.  

---

