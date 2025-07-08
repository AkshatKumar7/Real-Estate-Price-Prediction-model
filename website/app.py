from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import numpy as np

app = Flask(__name__)

# Path to models directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# Individual city models
individual_city_models = {
    'ahmedabad': 'ahmedabad_model.pkl',
    'bangalore': 'bangalore_model.pkl',
    'chennai': 'chennai_model.pkl',
    'faridabad': 'faridabad_model.pkl',
    'greaternoida': 'greaternoida_model.pkl',
    'gurgaon': 'gurgaon_model.pkl',
    'hyderabad': 'hyderabad_model.pkl',
    'jaipur': 'jaipur_model.pkl',
    'kolkata': 'kolkata_model.pkl',
    'newdelhi': 'newdelhi_model.pkl',
    'raipur': 'raipur_model.pkl',
    'ranchi': 'ranchi_model.pkl'
}

# Group city models
group_city_models = {
    "group_0_99_model.pkl": ['ahmadnagar', 'belgaum', 'bhiwandi', 'jodhpur', 'kozhikode',
                             'madurai', 'navsari', 'nellore', 'palakkad', 'pondicherry',
                             'rajahmundry', 'satara', 'shimla', 'solapur', 'tirupati', 'udupi', 'vrindavan'],
    "group_100_199_model.pkl": ['indore', 'allahabad', 'bhopal', 'durgapur', 'ernakulam',
                                'gwalior', 'haridwar', 'jabalpur', 'ludhiana', 'mysore', 'trichy',
                                'udaipur', 'vapi'],
    "group_200_499_model.pkl": ['agra', 'aurangabad', 'badlapur', 'bhubaneswar', 'guntur',
                                'mangalore', 'nashik', 'panchkula', 'siliguri', 'thrissur',
                                'trivandrum', 'varanasi'],
    "group_500_999_model.pkl": ['navi-mumbai', 'nagpur', 'lucknow', 'coimbatore', 'dehradun',
                                'ghaziabad', 'guwahati', 'jamshedpur', 'kalyan', 'kanpur',
                                'palghar', 'patna', 'sonipat', 'vijayawada'],
    "group_1000_1999_model.pkl": ['thane', 'mumbai', 'noida', 'bhiwadi', 'chandigarh', 'goa',
                                  'kochi', 'mohali', 'visakhapatnam', 'zirakpur'],
    "group_2000_2499_model.pkl": ['pune', 'surat', 'vadodara']
}

def get_model_filename(city):
    city = city.lower()
    if city in individual_city_models:
        return individual_city_models[city]
    for model_file, cities in group_city_models.items():
        if city in cities:
            return model_file
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    # Default form values
    form_values = {
        "selected_city": "",
        "floor_ratio": "",
        "bathroom": "",
        "balcony": "",
        "bhk": "",
        "carpet_area": "",
        "furnishing": "Unfurnished",
        "facing": 90,
        "ownership": 0,
        "car_parking": 0,
        "transaction": 0,
        "garden_park": 0,
        "main_road": 0,
        "pool": 0,
        "not_available": 0
    }

    if request.method == "POST":
        try:
            city = request.form.get("city", "").strip().lower()
            form_values["selected_city"] = city

            def parse_float(field): return float(request.form.get(field, 0) or 0)
            def parse_int(field): return int(request.form.get(field, 0) or 0)
            def get_value(field): return request.form.get(field, "")

            input_data = {
                "Floor (Ratio)": parse_float("floor_ratio"),
                "Bathroom": parse_int("bathroom"),
                "Balcony": parse_int("balcony"),
                "Car Parking": parse_int("car_parking"),
                "BHK": parse_int("bhk"),
                "Final Carpet Area (in sqft)": parse_float("carpet_area"),
                "Furnishing": get_value("furnishing"),
                "facing": parse_int("facing"),
                "Ownership": parse_int("ownership"),
                "Transaction": parse_int("transaction"),
                "Garden/Park": parse_int("garden_park"),
                "Main Road": parse_int("main_road"),
                "Pool": parse_int("pool"),
                "Not Available": parse_int("not_available")
            }

            form_values.update({
                "floor_ratio": input_data["Floor (Ratio)"],
                "bathroom": input_data["Bathroom"],
                "balcony": input_data["Balcony"],
                "bhk": input_data["BHK"],
                "carpet_area": input_data["Final Carpet Area (in sqft)"],
                "furnishing": input_data["Furnishing"],
                "facing": input_data["facing"],
                "ownership": input_data["Ownership"],
                "car_parking": input_data["Car Parking"],
                "transaction": input_data["Transaction"],
                "garden_park": input_data["Garden/Park"],
                "main_road": input_data["Main Road"],
                "pool": input_data["Pool"],
                "not_available": input_data["Not Available"]
            })

            model_file = get_model_filename(city)
            if not model_file:
                error = f"No model available for '{city.title()}'"
            else:
                model_path = os.path.join(MODEL_DIR, model_file)
                if not os.path.exists(model_path):
                    error = f"Model file not found at: {model_path}"
                else:
                    model = joblib.load(model_path)

                    # Handle model with or without named_steps
                    if hasattr(model, 'named_steps'):
                        preprocessor = model.named_steps["preprocessor"]
                    else:
                        error = "Invalid model format: missing 'named_steps'"
                        return render_template("index.html", prediction=prediction, error=error,
                                               cities=sorted_cities, **form_values)

                    try:
                        used_columns = preprocessor.transformers_[0][2] + preprocessor.transformers_[1][2]
                    except Exception:
                        used_columns = list(input_data.keys())

                    df = pd.DataFrame([input_data])
                    missing_cols = [col for col in used_columns if col not in df.columns]

                    if missing_cols:
                        error = f"Missing columns in input: {missing_cols}"
                    else:
                        df_filtered = df[used_columns]
                        log_price = model.predict(df_filtered)[0]
                        predicted_price = np.expm1(log_price)
                        prediction = f"Predicted Price: ₹{predicted_price:.2f} lakhs"

        except Exception as e:
            error = f"Prediction failed: {str(e)}"

    sorted_cities = sorted(
        set(individual_city_models.keys())
        | {city for cities in group_city_models.values() for city in cities}
    )

    return render_template("index.html", prediction=prediction, error=error, cities=sorted_cities, **form_values)

if __name__ == "__main__":
    print("▶ Running from:", os.path.abspath(os.getcwd()))
    app.run(debug=True)
