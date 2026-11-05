"""SymbioteScan - MMORPG Bot Detection Application."""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="SymbioteScan",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 SymbioteScan")
st.subheader("MMORPG Game Bot Detection System")

st.write(
    "Analyze player behavioral characteristics "
    "to identify potential game bots."
)

MODEL_PATH = "symbiotescan_model.pkl"

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    ["Home", "Bot Detection", "About"]
)

if page == "Home":

    st.header("Welcome to SymbioteScan")

    st.write(
        """
        SymbioteScan uses machine learning to distinguish
        between human players and automated game bots.

        **Main technologies:**
        - Python
        - Pandas
        - Scikit-learn
        - Random Forest
        - Streamlit
        """
    )

elif page == "Bot Detection":

    st.header("Player Bot Detection")

    uploaded_file = st.file_uploader(
        "Upload player behavioral data (CSV)",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            data = pd.read_csv(uploaded_file)

            if "Type" in data.columns:
                st.error(
                    "Remove the Type column before prediction."
                )

            else:
                saved = joblib.load(MODEL_PATH)

                model = saved["model"]
                feature_columns = saved["feature_columns"]

                forbidden = [
                    col for col in data.columns
                    if col.lower() in ["actor", "a_acc"]
                    or "ip" in col.lower()
                ]

                data = data.drop(
                    columns=forbidden,
                    errors="ignore"
                )

                features = data.reindex(
                    columns=feature_columns
                )

                if features.empty:
                    st.error("No player data found.")

                else:
                    predictions = model.predict(features)

                    probabilities = model.predict_proba(
                        features
                    )

                    st.success("Analysis completed!")

                    results = pd.DataFrame({
                        "Prediction": predictions,
                        "Confidence (%)": [
                            round(float(max(p)) * 100, 2)
                            for p in probabilities
                        ]
                    })

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    st.download_button(
                        "Download prediction results",
                        results.to_csv(index=False),
                        file_name="symbiotescan_results.csv",
                        mime="text/csv"
                    )

        except FileNotFoundError:
            st.error(
                "Model not found. Train the model first."
            )

        except Exception as error:
            st.error(f"Error: {error}")

elif page == "About":

    st.header("About SymbioteScan")

    st.write(
        """
        SymbioteScan is a machine-learning project
        designed to detect MMORPG game bots through
        player behavioral characteristics.

        The project is based on research using
        Aion MMORPG game data.
        """
    )

    st.write("Classification: Human vs Bot")
    st.write("Algorithm: Random Forest")
