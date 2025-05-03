# 🍲 Dishcovery - AI-Powered Image-to-Recipe Recommendation System

Dishcovery is a smart machine learning project that detects ingredients from food images and recommends Indian recipes using a trained image classifier and a logistic regression-based recommender. Built for our Machine Learning class at WPI, it features a sleek Streamlit interface and a modular backend.

---

## 🌟 Features

- 🖼️ Upload an image → detect ingredients (via CNN model)
- ✍️ Type ingredients manually (optional)
- 🔄 Combine both for better accuracy
- 🍛 Get Indian recipes instantly using logistic regression on a custom dataset
- 💻 Easy-to-use Streamlit interface (no CLI needed)

---

## 🚀 How to Run Locally

### ✅ 1. Clone this repository

```bash

run :
git clone https://github.com/abhyudaitheakur/Dishcovery.git
cd Dishcovery


### ✅ 2. Install dependencies
Make sure you have Python 3.7+ and pip installed. Then:

run :
pip install -r requirements.txt

✅ 3. Launch the App

run :
streamlit run app.py


Your browser will open at http://localhost:8501.


📁 Project Structure

Dishcovery/
├── app.py                      # Streamlit app
├── cleaned_indian_recipes.csv # Recipe dataset
├── trained_model.h5           # Food image detection model
├── labels.txt                 # Ingredient labels
├── recommender.py             # Core recipe recommendation logic
├── indian_recipe_recommender.py
├── requirements.txt
├── README.md


📦 Dependencies
The app requires the following Python packages (auto-installed from requirements.txt):


streamlit
pandas
numpy
scikit-learn
tensorflow
Pillow
