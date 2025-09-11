# 🏡 House Price Prediction

This project predicts **California house prices** using machine learning.  
It includes:
- Data preprocessing pipeline (with missing values handling, scaling, encoding)
- Model selection (Linear Regression, Decision Tree, Random Forest)
- Final trained Random Forest model
- A **Streamlit app** for user-friendly predictions

---

## 📂 Files in this project
- `app.py` → Streamlit app (manual user input for predictions)  
- `House_Price.py` → Builds pipeline, trains Random Forest model, saves `model.pkl` & `pipeline.pkl`  
- `Selecting_model.py` → Compares Linear Regression, Decision Tree, and Random Forest using cross-validation  
- `housing.csv` → Dataset used for training and testing  
- `model.pkl` → Trained Random Forest model  
- `pipeline.pkl` → Saved preprocessing pipeline  
- `requirements.txt` → Dependencies  
- `README.md` → Documentation  

---

## ⚙️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/libta018/House-Price-Prediction.git
cd House-Price-Prediction
