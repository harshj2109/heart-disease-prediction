
from flask import Flask, request, render_template
from src.Heart.pipeline.Prediction_pipeline import (
    CustomData,
    PredictPipeline
)

app = Flask(__name__)


# Health check route
@app.route("/health")
def health():
    return "<h1>Flask is working successfully!</h1>"


# Home and prediction route
@app.route("/", methods=["GET", "POST"])
def home():

    print("Home route accessed", flush=True)

    # Display homepage
    if request.method == "GET":
        return render_template("index.html")

    # Handle prediction form
    if request.method == "POST":

        try:
            print("Prediction request received", flush=True)

            # Collect form data
            required_fields = [
                "age", "sex", "cp", "trestbps", "chol",
                "fbs", "restecg", "thalach", "exang",
                "oldpeak", "slope", "ca", "thal"
            ]

            for field in required_fields:
                if not request.form.get(field):
                    raise ValueError(
                        f"Missing input value: {field}"
                    )

            # Convert inputs to numeric values
            data = CustomData(
                age=int(request.form["age"]),
                sex=int(request.form["sex"]),
                cp=int(request.form["cp"]),
                trestbps=int(request.form["trestbps"]),
                chol=int(request.form["chol"]),
                fbs=int(request.form["fbs"]),
                restecg=int(request.form["restecg"]),
                thalach=int(request.form["thalach"]),
                exang=int(request.form["exang"]),
                oldpeak=float(request.form["oldpeak"]),
                slope=int(request.form["slope"]),
                ca=int(request.form["ca"]),
                thal=int(request.form["thal"])
            )

            # Convert inputs into DataFrame
            final_data = data.get_data_as_dataframe()

            print("Input Data:", flush=True)
            print(final_data, flush=True)

            # Load model and make prediction
            print("Loading prediction pipeline...", flush=True)

            predict_pipeline = PredictPipeline()

            print("Generating prediction...", flush=True)

            prediction = predict_pipeline.predict(final_data)

            result = int(prediction[0])

            print("Prediction successful:", result, flush=True)

            # Display result
            return render_template(
                "result.html",
                final_result=result
            )

        except Exception as e:

            app.logger.exception("Prediction failed")

            # Return visible error instead of a blank page
            return (
                f"<h1>Prediction Error</h1>"
                f"<pre>{str(e)}</pre>"
                f'<a href="/">Go Back</a>',
                500
            )


# Start Flask application
if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=8080,
        debug=True,
        use_reloader=False
    )
