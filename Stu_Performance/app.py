# app.py
import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import io

st.set_page_config(page_title="Single Student Predictor (No CSV)", layout="centered")
st.title("🎯 Single Student Placement Predictor — Text Inputs & Dropdowns")
st.write("Upload a trained model (.pkl) trained on 1–10 inputs. Then enter one student's details manually and predict.")

# -------------------------
# Upload trained model (.pkl)
# -------------------------
uploaded_model = st.file_uploader("📁 Upload trained model (.pkl)", type=["pkl"])

model = None
scaler = None
columns = None
le_target = None
accuracy = None

if uploaded_model is not None:
    try:
        model_data = pickle.load(uploaded_model)
        model = model_data.get("model")
        scaler = model_data.get("scaler", None)        # may be None if retrained for 1-10
        columns = model_data.get("columns", None)
        le_target = model_data.get("label_encoder_target", None)
        accuracy = model_data.get("accuracy", None)

        if model is None:
            st.error("❌ Invalid .pkl: missing key 'model'. Provide a compatible pickle.")
            model = None
        else:
            st.success("✅ Model loaded.")
            if accuracy is not None:
                st.info(f"Saved model accuracy: {accuracy*100:.2f}%")
            if columns is not None:
                st.write("Saved feature order detected.")
            if scaler is not None:
                st.warning("Note: this model has a saved scaler. If you trained on raw 1-10 values, scaler should be None.")
    except Exception as e:
        st.error(f"Failed to load .pkl: {e}")
        model = None

st.markdown("---")
st.header("Enter single student details")

# -------------------------
# Input form (number inputs + dropdown)
# -------------------------
if model is None:
    st.info("Upload a trained model (.pkl) to enable single prediction (no CSV required).")
else:
    # optional student name/id
    student_id = st.text_input("Student ID / Name (optional)", value="")

    # Numeric inputs (use number_input for typed values)
    iq = st.number_input("IQ (1–10)", min_value=1.0, max_value=10.0, value=5.0, step=0.1, format="%.1f",
                         help="Enter IQ mapped to 1–10 scale (e.g., 9.0).")
    prev_sem = st.number_input("Previous Semester Result (1–10)", min_value=1.0, max_value=10.0, value=5.0, step=0.1, format="%.1f",
                               help="Previous semester aggregate mapped to 1–10.")
    cgpa = st.number_input("CGPA (1–10)", min_value=1.0, max_value=10.0, value=5.0, step=0.1, format="%.1f",
                           help="CGPA mapped to 1–10.")
    academic_perf = st.number_input("Academic Performance (1–10)", min_value=1.0, max_value=10.0, value=5.0, step=0.1, format="%.1f",
                                    help="Overall academic performance rating (1–10).")
    internship = st.selectbox("Internship Experience", options=["No", "Yes"], index=0,
                              help="Has the student completed any internship? Select Yes or No.")
    extra_curr = st.number_input("Extra Curricular Score (1–10)", min_value=1.0, max_value=10.0, value=5.0, step=0.1, format="%.1f",
                                help="Extracurricular activities rating (1–10).")
    comm_skill = st.number_input("Communication Skills (1–10)", min_value=1.0, max_value=10.0, value=5.0, step=0.1, format="%.1f",
                                 help="Communication skills rating (1–10).")
    projects = st.number_input("Projects Completed (1–10)", min_value=1.0, max_value=10.0, value=5.0, step=0.1, format="%.1f",
                               help="Number/quality of projects mapped to 1–10.")

    # Validate the inputs (defensive)
    input_dict = {
        "IQ": float(iq),
        "Prev_Sem_Result": float(prev_sem),
        "CGPA": float(cgpa),
        "Academic_Performance": float(academic_perf),
        "Internship_Experience": 1 if internship == "Yes" else 0,
        "Extra_Curricular_Score": float(extra_curr),
        "Communication_Skills": float(comm_skill),
        "Projects_Completed": float(projects)
    }

    def validate_inputs(d):
        errs = []
        for k, v in d.items():
            if k == "Internship_Experience":
                continue
            try:
                fv = float(v)
            except Exception:
                errs.append(f"{k} must be a number.")
                continue
            if not (1.0 <= fv <= 10.0):
                errs.append(f"{k} must be between 1 and 10.")
        return errs

    errors = validate_inputs(input_dict)
    if errors:
        for e in errors:
            st.error(e)

    # Predict button
    if st.button("🔮 Predict Placement"):
        if errors:
            st.error("Fix validation errors before predicting.")
        else:
            # Build DataFrame using proper column order
            input_df = pd.DataFrame([input_dict])

            # Reorder to saved columns if available (keep overlapping columns)
            if columns is not None:
                try:
                    cols_needed = [c for c in columns if c in input_df.columns]
                    input_df = input_df[cols_needed]
                except Exception:
                    st.warning("Could not reorder input to match saved columns; proceeding with default order.")

            # Apply scaler only IF a scaler exists in pickle.
            # IMPORTANT: if your model was retrained on 1-10 values, scaler will be None and we will not scale.
            if scaler is not None:
                try:
                    input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
                except Exception as e:
                    st.warning(f"Scaler.transform failed: {e} — continuing without scaling.")

            # Run prediction
            try:
                pred_enc = model.predict(input_df)[0]
                proba_vec = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
                prob_pos = proba_vec[1] if proba_vec is not None else None

                # Decode label if label encoder exists
                if le_target is not None:
                    try:
                        label = le_target.inverse_transform([pred_enc])[0]
                    except Exception:
                        label = str(pred_enc)
                else:
                    label = "Yes" if int(pred_enc) == 1 else "No"

                # Display results
                if prob_pos is not None:
                    if int(pred_enc) == 1:
                        st.success(f"✅ Predicted: {label} — Confidence: {prob_pos*100:.2f}%")
                    else:
                        st.error(f"❌ Predicted: {label} — Confidence: {(1-prob_pos)*100:.2f}%")
                else:
                    st.info(f"Predicted: {label} (probability unavailable)")

                # Save to session log
                log_row = {
                    "student_id": student_id,
                    **input_dict,
                    "predicted_label": label,
                    "predicted_encoded": int(pred_enc),
                    "probability": float(prob_pos) if prob_pos is not None else None,
                    "timestamp": datetime.utcnow().isoformat()
                }

                if "predictions_log" not in st.session_state:
                    st.session_state["predictions_log"] = pd.DataFrame([log_row])
                else:
                    st.session_state["predictions_log"] = pd.concat([st.session_state["predictions_log"], pd.DataFrame([log_row])], ignore_index=True)

                st.success("✅ Prediction saved to session log.")
                st.subheader("🗂️ Predictions log (this session)")
                st.dataframe(st.session_state["predictions_log"].sort_values(by="timestamp", ascending=False))

                # Download CSV button
                csv_bytes = st.session_state["predictions_log"].to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download predictions CSV", data=csv_bytes, file_name="predictions_log.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
