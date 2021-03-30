from flask import Flask
from flask import request
from tensorflow import keras
import numpy as np

app = Flask(__name__)
# Load the NN model
model = keras.models.load_model('NN_addition_operation.h5')

@app.route("/")
def index():
    input0 = request.args.get("input0", "")
    input1 = request.args.get("input1", "")
    input2 = request.args.get("input2", "")
    input3 = request.args.get("input3", "")
    input4 = request.args.get("input4", "")
    all_inputs = [input0, input1, input2, input3, input4]

    if not not any(all_inputs):
        all_inputs = [float(i) for i in all_inputs] # convert to float
        all_inputs = np.array([all_inputs])
        output = NeuralNetwork(all_inputs)
    else:
        output = ""
    return (
        " Insert inputs to predict the addition output:"
        + """<form action="" method="get">
                Input0: <input type="text" name="input0">
                Input1: <input type="text" name="input1">
                Input2: <input type="text" name="input2">
                Input3: <input type="text" name="input3">
                Input4: <input type="text" name="input4">
                <input type="submit" value="Send">
            </form>"""
        + "Output: "
        + output
    )

def NeuralNetwork(all_inputs):
    """Script"""
    try:

        prediction = model.predict(all_inputs)
        return str(prediction[0][0])  
    except ValueError:
        return "invalid input"



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)