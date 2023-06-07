from flask import Flask, render_template, request, redirect, url_for
from src.pipelines.prediction_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)
app = application

categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC',
                    'CALC', 'MTRANS']
numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

columns = categorical_cols + numerical_cols

gender_categories = ['Female', 'Male']
family_history_with_overweight_categories = ['no', 'yes']
favc_categories = ['no', 'yes']
caec_categories = ['Always', 'Frequently', 'Sometimes', 'no']
smoke_categories = ['no', 'yes']
scc_categories = ['no', 'yes']
calc_categories = ['Always', 'Frequently', 'Sometimes', 'no']
mtrans_categories = ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking']

drop_down_order = [gender_categories,
                   family_history_with_overweight_categories,
                   favc_categories,
                   caec_categories,
                   smoke_categories,
                   scc_categories,
                   calc_categories,
                   mtrans_categories]


@app.route("/")
def index():
    # Get the value of the training_complete parameter from the URL
    training_complete = request.args.get('training_complete', False)
    return render_template("index.html", col=columns, drop_down_order=drop_down_order,
                           categorical_cols=categorical_cols, enumerate=enumerate)


@app.route("/predict", methods=["GET", "POST"])
def new_prediction():
    Gender = request.form['Gender']
    family_history_with_overweight = request.form['family_history_with_overweight']
    FAVC = request.form['FAVC']
    CAEC = request.form['CAEC']
    SMOKE = request.form['SMOKE']
    SCC = request.form['SCC']
    CALC = request.form['CALC']
    MTRANS = request.form['MTRANS']
    Age = request.form['Age']
    Height = request.form['Height']
    Weight = request.form['Weight']
    FCVC = request.form['FCVC']
    NCP = request.form['NCP']
    CH2O = request.form['CH2O']
    FAF = request.form['FAF']
    TUE = request.form['TUE']

    data = CustomData(Gender, family_history_with_overweight, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS, Age, Height,
                      Weight, FCVC, NCP, CH2O, FAF, TUE)
    df = data.get_data_as_dataframe()
    model = PredictionPipeline()
    prediction = model.predict(df)
    prediction = int(prediction[0])
    if prediction == 1:
        prediction = 'Obesity_Type_I'
    elif prediction == 2:
        prediction = 'Obesity_Type_III'
    elif prediction == 3:
        prediction = 'Obesity_Type_II'
    elif prediction == 4:
        prediction = 'Overweight_Level_I'
    elif prediction == 5:
        prediction = 'Overweight_Level_II'
    elif prediction == 6:
        prediction = 'Normal_Weight'
    elif prediction == 7:
        prediction = 'Insufficient_Weight'
    return render_template("result.html", predict=prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
