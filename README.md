AIML Tourism Experience Analytics

This project analyzes tourism experience data and predicts user experience using Machine Learning.

Files:

train_models.py  
Trains ML models:
- Regression model to predict Rating
- Classification model to predict Visit Mode

app.py  
Streamlit application that allows users to:
- Select Country, City, Attraction, Year, Month
- Predict Rating
- Predict Visit Mode
- View Recommended Attractions

Pipeline:

CSV → ML Models → Streamlit

How to Run:

python train_models.py  
python -m streamlit run app.py  

Models Used:

- Linear Regression (Rating Prediction)
- Random Forest Classifier (Visit Mode Prediction)

Output:

Predicted Rating  
Predicted Visit Mode  
Recommended Attractions
