from flask import Flask, render_template, request, session
import os
from werkzeug.utils import secure_filename
from cell_segmentation_model_inference import mask_image_cell_segments

# *** Backend operation

# WSGI Application
# Defining upload folder path
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
# # Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name for template path
# The default folder name for static files should be "static" else need to mention custom folder for static path
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'


@app.route('/')
def index():
    return render_template('index_upload_and_show_data.html')


@app.route('/', methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # Upload file flask
        uploaded_img = request.files['uploaded-file']
        # Extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)
        # Upload file to database (defined uploaded folder in static path)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)

        return render_template('index_upload_and_show_data_page2.html')


@app.route('/show_image')
def displayImage():
    # Retrieving uploaded file path from session
    img_file_path = session.get('uploaded_img_file_path', None)
    # Display image in Flask application web page
    return render_template('show_image.html', user_image=img_file_path)


@app.route('/cell_segmentation')
def segmentCells():
    # Retrieving uploaded file path from session
    img_file_path = session.get('uploaded_img_file_path', None)
    image_orig, image_orig_masked = mask_image_cell_segments(image_path=img_file_path,
                                                             model_chkpt_path='MaskRCNN_FineTuned/pytorch_model-e26.bin')
    full_masked_image_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'masked_image.png')
    image_orig_masked.save(full_masked_image_filepath, 'PNG')

    # Display image in Flask application web page
    return render_template('show_masked_image.html', user_image=full_masked_image_filepath)


if __name__ == '__main__':
    app.run(debug=True)
