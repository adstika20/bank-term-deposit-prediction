import gradio as gr
import pandas as pd
import joblib

# =========================
# LOAD ARTIFACTS
# =========================
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

categorical_cols = ['job', 'marital', 'education', 'default',
                    'housing', 'loan', 'contact', 'month', 'poutcome']

# =========================
# PREDICT FUNCTION
# =========================
def predict(job, marital, education, default, housing, loan, contact,
            month, poutcome, age, balance, duration, campaign, pdays, previous):

    # buat dict
    input_dict = {
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "poutcome": poutcome,
        "age": age,
        "balance": balance,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous
    }

    df = pd.DataFrame([input_dict])

    # encode kategorikal
    for col in categorical_cols:
        le = label_encoders[col]
        df[col] = le.transform(df[col])

    # susun kolom sesuai feature_columns
    df = df[feature_columns]

    # scaling
    X_scaled = scaler.transform(df)

    # predict
    pred = model.predict(X_scaled)

    # decode label target
    target_le = label_encoders["y"]
    final_label = target_le.inverse_transform(pred)[0]  # “yes” or “no”

    return f"Prediction: {final_label.upper()}"


# =========================
# GRADIO APP
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# Bank Campaign Prediction (SMOTE-trained Model)")

    with gr.Row():
        with gr.Column():
            job = gr.Dropdown(["admin.","technician","services","management",
                               "retired","blue-collar","entrepreneur","housemaid",
                               "self-employed","unemployed","student","unknown"],
                              label="Job")

            marital = gr.Dropdown(["married","single","divorced"], label="Marital Status")
            education = gr.Dropdown(["primary","secondary","tertiary","unknown"], label="Education")
            default = gr.Dropdown(["yes","no"], label="Default Credit?")
            housing = gr.Dropdown(["yes","no"], label="Housing Loan?")
            loan = gr.Dropdown(["yes","no"], label="Personal Loan?")
            contact = gr.Dropdown(["cellular","telephone","unknown"], label="Contact Type")
            month = gr.Dropdown(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], label="Month")
            poutcome = gr.Dropdown(["success","failure","other","unknown"], label="Previous Outcome")

        with gr.Column():
            age = gr.Number(label="Age")
            balance = gr.Number(label="Balance")
            duration = gr.Number(label="Duration")
            campaign = gr.Number(label="Campaign")
            pdays = gr.Number(label="pdays")
            previous = gr.Number(label="previous")

            btn = gr.Button("Predict Term Deposit Subscription")
            output = gr.Textbox(label="Result")

    btn.click(
        predict,
        inputs=[job, marital, education, default, housing, loan, contact,
                month, poutcome, age, balance, duration, campaign, pdays, previous],
        outputs=output
    )

demo.launch()
