# 🍽️ Swiggy Restaurant Recommendation System

A machine learning-powered restaurant recommendation system based on user preferences (city, cuisine, rating, cost) built with Python and Streamlit.

---

## 📌 Problem Statement

The objective is to build a recommendation system based on restaurant data provided in a CSV file (contained inside the `swiggy.rar` archive).  
The system should recommend restaurants based on user input features such as:

- City
- Cuisine
- Minimum Rating
- Maximum Cost

The application uses **KMeans clustering** and **similarity-based filtering** to generate the best restaurant recommendations. Results are displayed in a clean, easy-to-use **Streamlit** web application.

---

## 📊 Dataset

- Archive File: `swiggy.rar`
- Inside archive:
  - `swiggy.csv` - Original restaurant dataset
- Columns in CSV:
  - `id`, `name`, `city`, `rating`, `rating_count`, `cost`, `cuisine`, `lic_no`, `link`, `address`, `menu`
- Categorical Features: `name`, `city`, `cuisine`
- Numerical Features: `rating`, `rating_count`, `cost`

---

## 📂 Project Structure

| File | Description |
|:-----|:------------|
| `swiggy.py` | Main Streamlit application and recommendation logic |
| `swiggy.rar` | Compressed archive containing `swiggy.csv` dataset |
| `clean_data.csv` | Cleaned data (created after cleaning) |
| `encode_data.csv` | Encoded data with features |
| `encoder.pkl` | Saved OneHotEncoder and MultiLabelBinarizer |

---

## ⚙️ Execution Flow

1. Load and clean the dataset (`swiggy.csv` extracted from `swiggy.rar`).
2. Display input widgets for user preferences (city, cuisine, rating, cost).
3. After collecting inputs, perform:
   - Feature encoding
   - Feature scaling
   - KMeans clustering
4. Recommend top restaurants matching user preferences.
5. Display recommendations in a neat table.

---

## 🛠️ Python Packages Used

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)

---

## 📦 Installation Instructions

1. **Clone this repository**
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

2. **Install required Python packages**
```bash
pip install pandas numpy streamlit scikit-learn
```

3. **Check your environment**
- Python 3.8+ recommended

4. **Extract swiggy.rar**
Make sure to extract `swiggy.csv` from `swiggy.rar` archive before running the project.

---

## 🚀 How to Run the Project

In your project directory, run:

```bash
streamlit run swiggy.py
```

This will open the Streamlit app in your web browser at `http://localhost:8501`.

---

## 📝 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute freely!

---

## 📚 Additional Notes

- The first time you run the app, it will create `clean_data.csv`, `encode_data.csv`, and `encoder.pkl`.
- The cuisines shown to users are **original names** from the dataset (not encoded labels).
- Machine learning (KMeans clustering) is applied **after** user inputs to optimize performance.
- The app gracefully handles missing or incorrect data.
- Fully designed for educational and demonstration purposes.

---

## ✨ Future Enhancements

- Add cosine similarity-based recommendation
- Add restaurant images and details in output
- Add filtering based on offers or delivery time

---
