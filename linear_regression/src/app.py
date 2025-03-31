import streamlit as st
import pandas as pd
import joblib

def load_model(model_name):
    return joblib.load(f'models/{model_name}.pkl')

def predict(model, input_data):
    return model.predict(input_data)

def main():
    st.title("Energy Efficiency Predictor")
    
    # Model selection
    model_name = st.selectbox(
        'Select Model',
        ['linearregression', 'ridge', 'lasso']
    )
    
    model = load_model(model_name)
    
    # Input form
    st.sidebar.header("Input Parameters")
    input_data = {}
    for i in range(1, 9):
        input_data[f'X{i}'] = st.sidebar.number_input(f'X{i}', value=0.0)
    
    if st.sidebar.button('Predict'):
        input_df = pd.DataFrame([input_data])
        prediction = predict(model, input_df)
        
        st.subheader("Prediction Results")
        for i, target in enumerate(['Y1','Y2']):
            st.write(f"{target}: {prediction[0][i]:.2f}")

if __name__ == '__main__':
    main()