# frontend/app.py

import streamlit as st
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")
if not BACKEND_URL:
    st.error("Backend URL not configured. Please set the BACKEND_URL environment variable.")

# Streamlit Layout
st.title("TrackMate - MLOps Assistant")

menu = ["Create Experiment", "View Experiments", "Manage Runs", "Ask Assistant"]
choice = st.sidebar.selectbox("Menu", menu)

@st.cache_data()
def get_experiments():
    try:
        response = requests.get(f"{BACKEND_URL}/experiments/")
        response.raise_for_status()
        return response.json()["experiments"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching experiments: {str(e)}")
        return []

if choice == "Create Experiment":
    st.subheader("Create a New Experiment")

    with st.form("create_experiment_form"):
        exp_name = st.text_input("Experiment Name")
        exp_desc = st.text_area("Description")
        submit = st.form_submit_button("Create Experiment")

        if submit:
            if exp_name:
                payload = {
                    "name": exp_name,
                    "description": exp_desc
                }
                try:
                    response = requests.post(f"{BACKEND_URL}/experiments/", json=payload)
                    response.raise_for_status()
                    res = response.json()
                    st.success(f"Experiment '{res['name']}' created with ID: {res['experiment_id']}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.error("Experiment name cannot be empty.")

elif choice == "View Experiments":
    st.subheader("List of Experiments")

    experiments = get_experiments()
    if experiments:
        exp_df = pd.DataFrame(experiments)
        st.dataframe(exp_df)
    else:
        st.info("No experiments found.")

elif choice == "Manage Runs":
    st.subheader("Manage Runs")

    experiments = get_experiments()
    if experiments:
        exp_names = [exp["name"] for exp in experiments]
        selected_exp = st.selectbox("Select Experiment", exp_names)
        selected_exp_id = next(exp["experiment_id"] for exp in experiments if exp["name"] == selected_exp)

        # Start a new run
        with st.expander("Start a New Run"):
            run_name = st.text_input("Run Name (optional)")
            if st.button("Start Run"):
                payload = {"run_name": run_name}
                try:
                    run_response = requests.post(f"{BACKEND_URL}/experiments/{selected_exp_id}/runs/", json=payload)
                    run_response.raise_for_status()
                    run_res = run_response.json()
                    st.success(f"Run started with ID: {run_res['run_id']} and Status: {run_res['status']}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error starting run: {str(e)}")

        # Log parameters and metrics
        st.subheader("Log Parameters and Metrics")

        run_id = st.selectbox("Select Run ID", [None] + [run.get("run_id") for run in experiments if run.get("run_id")])
        if run_id:
            with st.form("log_form"):
                param_key = st.text_input("Parameter Key")
                param_value = st.text_input("Parameter Value")
                metric_key = st.text_input("Metric Key")
                metric_value = st.number_input("Metric Value", min_value=0.0)
                submit = st.form_submit_button("Log Parameter and Metric")

                if submit:
                    try:
                        if param_key and param_value:
                            param_payload = {"key": param_key, "value": param_value}
                            param_response = requests.post(f"{BACKEND_URL}/runs/{run_id}/params/", json=param_payload)
                            param_response.raise_for_status()
                            st.success(param_response.json()["message"])

                        if metric_key:
                            metric_payload = {"key": metric_key, "value": metric_value}
                            metric_response = requests.post(f"{BACKEND_URL}/runs/{run_id}/metrics/", json=metric_payload)
                            metric_response.raise_for_status()
                            st.success(metric_response.json()["message"])
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error logging data: {str(e)}")
        else:
            st.info("Please select a run ID.")

        # View run details
        with st.expander("View Run Details"):
            if st.button("Get Run Details") and run_id:
                try:
                    run_details_response = requests.get(f"{BACKEND_URL}/experiments/{selected_exp_id}/runs/{run_id}/")
                    run_details_response.raise_for_status()
                    run_data = run_details_response.json()["run"]
                    st.write(run_data)

                    # Plot metrics if available
                    if run_data["metrics"]:
                        metrics_df = pd.DataFrame(list(run_data["metrics"].items()), columns=["Metric", "Value"])
                        st.bar_chart(metrics_df.set_index("Metric"))
                except requests.exceptions.RequestException as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("No experiments found.")

elif choice == "Ask Assistant":
    st.subheader("Ask the Assistant")

    st.markdown("**Examples of questions you can ask:**")
    st.markdown("""
        - How can I improve the accuracy of my model?
        - Why did run `run_id` perform worse than run `run_id`?
        - What are the most significant parameters affecting my model's performance?
    """)

    user_question = st.text_area("Enter your question about your experiments or runs:")
    if st.button("Get Answer"):
        if user_question:
            payload = {"prompt": user_question}
            try:
                response = requests.post(f"{BACKEND_URL}/assistant/query", json=payload)
                response.raise_for_status()
                assistant_response = response.json()["response"]
                st.markdown("**Assistant:**")
                st.write(assistant_response)
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {str(e)}")
        else:
            st.error("Please enter a question.")
