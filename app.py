import streamlit as st
import numpy as np
import pickle

# Load trained model and preprocessing objects
model = pickle.load(open("thyroid_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))

# Streamlit UI setup
st.set_page_config(page_title="Hypothyroidism Predictor", page_icon="🩺")
st.title("🩺 Hypothyroidism Disease Predictor")
st.markdown("Please enter the following medical values to get a prediction.")

# Final selected features used in the model
features = [
    "age", "sex", "on_thyroxine", "on_antithyroid_medication",
    "thyroid_surgery", "i131_treatment", "lithium", "goitre", "tumor",
    "tsh", "t3", "tt4", "t4u", "fti"
]

# Binary features (excluding 'sex')
binary_features = [
    "on_thyroxine", "on_antithyroid_medication",
    "thyroid_surgery", "i131_treatment", "lithium", "goitre", "tumor"
]

input_data = []

# Input loop
for feature in features:
    label = feature.replace("_", " ").title()

    if feature == "sex":
        gender = st.selectbox("Sex", ["Female", "Male"])
        val = 1 if gender == "Male" else 0

    elif feature in binary_features:
        bin_choice = st.selectbox(f"{label} (Yes / No)", ["No", "Yes"])
        val = 1 if bin_choice == "Yes" else 0

    elif feature == "age":
        val = st.slider("Age (years)", min_value=0, max_value=120, value=40, step=1)
        if val < 10:
            st.warning("👶 Age below 10 is rare in hypothyroid diagnosis.")
        elif val > 85:
            st.warning("🧓 Higher age may increase risk. Please ensure data is accurate.")

    else:
        val = st.number_input(f"{label}", step=0.01)

    input_data.append(val)

# Predict button
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)

    # Preprocessing
    input_scaled = scaler.transform(input_array)
    input_pca = pca.transform(input_scaled)
    prediction = model.predict(input_pca)[0]

    # Output result
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("⚠️ The person is likely to have **Hypothyroidism**.")

        st.markdown("---")
        st.subheader("🔍 Suggested Follow-up:")
        st.markdown("""
- 📌 **Recommended Lab Tests:**
  - TSH (Thyroid Stimulating Hormone)
  - Free T3, Free T4
  - Anti-TPO antibodies (for autoimmune check)
  - CBC, Vitamin D, and Lipid Profile (optional)

- 💊 **Lifestyle Tips:**
  - Take thyroid medication regularly if prescribed.
  - Avoid high-soy and high-iodine foods unless advised.
  - Maintain regular light exercise and good sleep hygiene.

- 🩺 **Next Step:**
  - Please consult an **Endocrinologist** for confirmation and treatment.
        """)

    else:
        st.success("✅ The person is **Not Hypothyroid**.")
        st.markdown("🩺 Keep monitoring health parameters annually and maintain a balanced lifestyle.")

