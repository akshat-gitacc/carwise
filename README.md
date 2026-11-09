# CarWise

CarWise estimates the price of a used/unused car. You enter the car's details, and get a predicted price, the probability behind it, and an explanation of which factors pushed the price up or down.

## What it does

- Predicts the price of a car based on inputed features corresponding to its quality. The car can be used or unused. A generated explanation of why the price is predicted as such would also be provided in brief.
- Uses an XGBoost model trained on CarDekho India v3 Dataset.
- Uses SHAP (TreeSHAP) to show which factors helped or hurt the rating.

## Project layout

```
carwise/
├── backend/          # Data, model, explanations,
├── frontend/         # Built using Streamlit
├── data/             # CarDekho Dataset
├── models/           # Trained model 
└── requirements.txt
```



## Credits

Built with XGBoost, SHAP, and FASTAPI.
