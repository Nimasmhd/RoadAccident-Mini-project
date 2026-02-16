# 🚗 Road Accident Prediction System

This project is a **Machine Learning-based web application** that predicts the probability of road accidents using historical data. The system is built with **Python**, **Random Forest algorithm**, and deployed using **Streamlit**.

This project is part of my research:  
**"Predicting Road Accident Probabilities in Vavuniya Town, Sri Lanka Using Machine Learning Techniques"**

---

## 📌 Features

- Predict road accident probability based on input factors  
- User-friendly web interface using Streamlit  
- Machine Learning model trained using Random Forest  
- Real-time prediction results  
- Preprocessing using scaler and label encoder  

---

## 🛠️ Technologies Used

- Python  
- Streamlit  
- Scikit-learn  
- Pandas  
- NumPy  
- Pickle  

---

## 📂 Project Structure

```
Road_Accident_Prediction/
│
├── app.py                          # Main Streamlit application
├── road_accident_prediction.pkl   # Trained Random Forest model
├── scaler.pkl                     # Saved scaler object
├── label_encoder.pkl              # Saved label encoder
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## ⚙️ Dependencies

Install the required libraries using:

```
pip install -r requirements.txt
```

Or install manually:

```
pip install streamlit pandas numpy scikit-learn
```

---

## 🧠 Machine Learning Model

- Algorithm: **Random Forest Classifier**
- Model file: `road_accident_prediction.pkl`
- Preprocessing:
  - `scaler.pkl` → for feature scaling  
  - `label_encoder.pkl` → for encoding categorical variables  

---

## 🚀 How to Run the Application

Follow these steps:

### Step 1: Clone the Repository

```
git clone https://github.com/your-username/road-accident-prediction.git
cd road-accident-prediction
```

---

### Step 2: Install Dependencies

```
pip install -r requirements.txt
```

---

### Step 3: Make Sure Model Files Exist

Ensure the following files are in the project folder:

- `road_accident_prediction.pkl`  
- `scaler.pkl`  
- `label_encoder.pkl`  

---

### Step 4: Run the Streamlit App

```
streamlit run app.py
```

---

### Step 5: Open in Browser

Streamlit will automatically open:

```
http://localhost:8501
```

---

## 📊 How It Works

1. User inputs accident-related data (e.g., weather, road condition, time, etc.)
2. Data is preprocessed using:
   - Label Encoder
   - Scaler
3. Processed data is sent to the trained model
4. Model predicts accident probability
5. Result is displayed on the web interface

---

## 📁 Example Code (Model Loading in app.py)

```python
import pickle

# Load model
model = pickle.load(open("road_accident_prediction.pkl", "rb"))

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load label encoder
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
```

---

## 📌 Future Improvements

- Add real-time data integration  
- Improve model accuracy with more data  
- Deploy to cloud (Streamlit Cloud / AWS)  
- Add visualization dashboard  

---

## 👨‍💻 Author

**Nimas Rfk**  
Undergraduate - University of Vavuniya  

---

## 📄 License

This project is for academic and research purposes.
