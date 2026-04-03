# 🛒 BigMart Sales Prediction

## 📌 Project Overview

This project aims to predict the sales of products in BigMart stores using machine learning techniques.
It helps businesses understand how different factors like product type, outlet type, and visibility affect sales.

---

## 🚀 Features

* Predicts product sales based on input features
* Interactive web app using Streamlit
* Clean and simple UI
* Handles categorical and numerical data using a pipeline

---

## 🧠 Machine Learning Approach

* Model Used: **Ridge Regression**
* Preprocessing:

  * OneHot Encoding for categorical features
  * Numerical features passed directly
* Pipeline used to combine preprocessing and model

---

## 📂 Project Structure

```
Bigmart_Project/
│
├── main.py                     # Streamlit app
├── train_model.py              # Model training script
├── cleaned_data.csv           # Processed dataset
├── Ridge_Regression_best_model.pkl  # Trained model
├── requirements.txt           # Required libraries
└── README.md                  # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone <your-repo-link>
cd Bigmart_Project
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the application

```
streamlit run main.py
```

---

## 🔁 Train the Model (Optional)

If you face any compatibility issues with the `.pkl` file:

```
python train_model.py
```

This will regenerate the trained model.

---

## 📊 Dataset

The dataset contains information about:

* Item Type
* Outlet Type
* Item Visibility
* Item Weight
* Outlet Establishment Year
* And more...

---

## 💡 Key Learnings

* Built an end-to-end ML pipeline
* Solved model serialization issues
* Learned Streamlit for deployment
* Understood real-world data preprocessing

---


## ⭐ Conclusion

This project demonstrates how machine learning can be applied to real-world retail problems to improve decision-making and sales forecasting.
