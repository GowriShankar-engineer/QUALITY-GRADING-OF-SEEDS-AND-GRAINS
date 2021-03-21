#importing the modules here
from flask import Flask,render_template
import os
from flask import request
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
#importing the ends modules here

#flask initalize starts here
app = Flask(__name__, template_folder="template")
UPLOAD_FOLDER = "C:/Users/Shankar Dinesh/Desktop/webapp/TESTING APP"
#flask initailize ends here 


#initalize the globally for model starts here
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('model.h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
size = ()
#initalize the globally for model ends here


#predict function starts here
def predict(image_path, model,size,data):
    #image setup startshere
    #image = image_path
    image = Image.open(image_path)
    #size = (224, 224)
    image = ImageOps.fit(image, (224,224), Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    #prediction starts here
    prediction = model.predict(data)
    #predtion ends here
    
    label = ['Rice Grade 1', 'Rice Grade 2', 'Rice Grade 3', 'Green Gram 1','Green Gram 2']
    count = -1
    index = 0
    prediction = prediction[0];
    max = prediction.max();
    for n, i in enumerate(prediction):
        if i == max:
            index = n  

    return label[index]
#predict function ends here

#setting up the upload controls starts here
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict(image_location, model,size,data)
            pred = pred.upper()
            return render_template('index.html', prediction=pred)
    return render_template("index.html", prediction="")
#setting up the upload controls starts here

#Starts Renders
if __name__ == "__main__":
    app.run(debug=True)
#Ends Renders