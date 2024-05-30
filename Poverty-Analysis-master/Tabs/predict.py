"""This modules contains data about prediction page"""

# Import necessary modules
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
# Import necessary functions from web_functions
from web_functions import predict, train_model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def app(df, X, y):
    """This function create the prediction page"""

    # Add title to the page
    st.title("Prediction Page")

    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                This app uses <b style="color:green">Random Forest Classifier</b> for the Poverty Analysis.
            </p>
        """, unsafe_allow_html=True)
    
    # Take feature input from the user
    # Add a subheader
    st.subheader("Select Values:")

    # Take input of features from the user.
    
    Population = st.slider("Population", float(df["Population"].min()), float(df["Population"].max()))
    Literacy_Index = st.slider("Literacy Rate", float(df["Literacy_Index"].min()), float(df["Literacy_Index"].max()))
    Poverty_Index = st.slider("Poverty Index", float(df["Poverty_Index"].min()), float(df["Poverty_Index"].max()))
    Standard_of_Life = st.slider("Standard of Living", float(df["Standard_of_Life"].min()), float(df["Standard_of_Life"].max()))
    Hunger_Index = st.slider("Hunger Index", float(df["Hunger_Index"].min()), float(df["Hunger_Index"].max()))
    Satisfaction_Level = st.slider("Governance Satisfaction", float(df["Satisfaction_Level"].min()), float(df["Satisfaction_Level"].max()))
    Healthcare = st.slider("Healthcare Satisfaction", float(df["Healthcare"].min()), float(df["Healthcare"].max()))
    Basic_Needs = st.slider("Basic Needs", float(df["Basic_Needs"].min()), float(df["Basic_Needs"].max()))
     
    

    # Create a list to store all the features
    features = [Population,Literacy_Index,Poverty_Index,Standard_of_Life,Hunger_Index,Satisfaction_Level,Healthcare,Basic_Needs]

    # Create a button to predict
    if st.button("Detect Class"):
        # Get prediction and model score
        prediction, score = predict(X, y, features)
        score = score
        prediction, score = predict(X, y, features)
        
        # After taking user input and before predicting the class
# Create a DataFrame for user input
        user_input_df = pd.DataFrame({
    "Feature": [ "Literacy Rate", "Poverty Index", "Standard of Living", "Hunger Index", 
                "Governance Satisfaction", "Healthcare Satisfaction", "Basic Needs"],
    "Value": [ Literacy_Index, Poverty_Index, Standard_of_Life, Hunger_Index, 
              Satisfaction_Level, Healthcare, Basic_Needs]
})

        st.info("User Input:")
        # st.bar_chart(user_input_df.set_index("Feature"))
        st.write(user_input_df)
        # st.set_option('deprecation.showPyplotGlobalUse', False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Value", y="Feature", data=user_input_df, palette="viridis")
        plt.title("User Input Impact on Poverty")
        plt.xlabel("Value")
        plt.ylabel("Feature")
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
    # Plotting distribution of features
        # st.write("Distribution of Features")
        # plt.figure(figsize=(10, 6))
        # sns.histplot(df["Population"], bins=20, kde=True)
        # st.pyplot()
        # st.set_option('deprecation.showPyplotGlobalUse', False)

    # Comparing with dataset
        st.info("Comparing with Dataset")
        comparison_df = pd.DataFrame({"Actual Score": y.values, "Predicted Score": prediction}, index=df.index)
        st.write(comparison_df)

    # # Visualizing comparison
    #     st.info("Visualization of Comparison")
    #     plt.figure(figsize=(10, 6))
    #     sns.countplot(x="Actual Score", data=comparison_df, palette="viridis")
    #     sns.countplot(x="Predicted Score", data=comparison_df, palette="plasma")
    #     plt.legend(["Actual", "Predicted"])
    #     st.pyplot()

        

    # Display a brief summary
        

        feature_values = {
    "Population": Population,
    "Literacy Rate": Literacy_Index,
    "Poverty Index": Poverty_Index,
    "Standard of Living": Standard_of_Life,
    "Hunger Index": Hunger_Index,
    "Governance Satisfaction": Satisfaction_Level,
    "Healthcare Satisfaction": Healthcare,
    "Basic Needs": Basic_Needs
} 
                
        if (prediction == 2):
            st.error("Extremely poor! On brink of poverty")
            
        elif (prediction == 3):
            st.error("Very Poor! Need financial aids")
          
        elif (prediction == 4):
            st.error("Poor! Need financial aids")
     
        elif (prediction == 5):
            st.warning("Average. Need good governance")
      
        elif (prediction == 6):
            st.warning("Good! Well to do")
    
        elif (prediction == 7):
            st.warning("Very good! Significantly good.")

        elif (prediction == 8):
            st.success("Prosperous State! Doing good.")
    
        elif (prediction == 9):
            st.success("Magnificient! Surplus aids present")
  
        elif (prediction == 10):
            st.success("Extremely Good and Rich State")
 
        
        # Prfloat teh score of the model 
        st.sidebar.write("The model used is trusted by beaurocratists and has an accuracy of ", round((score*100),2),"%")
