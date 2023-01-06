from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import subprocess

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dfsfdafssdffds'
app.config['UPLOAD_FOLDER'] = 'templates/images'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

#@app.route('/', methods=['GET',"POST"])

@app.route('/home', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename('1.tif'))) # Then save the file
        #exec(open('predict.py').read())
        form2 = UploadFileForm()
        ans= subprocess.getoutput('python tools/predict.py --cfg experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --testModel output/imagenet/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100/model_best.pth.tar')
        return render_template('index.html', form=form ,ans=ans)
    
        #subprocess.getoutput('python predict.py --cfg cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --testModel model_best.pth.tar')
        
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)