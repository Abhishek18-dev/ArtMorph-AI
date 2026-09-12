import os
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
import io

from pathlib import Path

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std


app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField('Content Path')
    style_path = HiddenField('Style Path')
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    try:
        import torch_directml
        device = torch_directml.device()
    except:
        device = torch.device("cpu")

print(f"Using device: {device}")



# TEMPORARILY DISABLED — only testing Flask UI
# encoder = None
# decoder = None


# encoder = VGGEncoder('vgg_normalised.pth').to(device)
encoder = VGGEncoder(str(BASE_DIR / 'vgg_normalised.pth')).to(device)
decoder = Decoder().to(device)
# decoder.load_state_dict(torch.load(r'C:\Users\Abhishek\NST_AdaIN_Main\NST_AdaIN\experiment\test\checkpoint_epoch_2.pth'))
# decoder.load_state_dict(torch.load(r'C:\Users\Abhishek\Downloads\checkpoint_epoch_19.pth'))



checkpoint = torch.load(
    BASE_DIR /  'decoder_final.pth',
    map_location='cpu'
)

decoder.load_state_dict(checkpoint['decoder'])

decoder = decoder.to(device)



encoder.eval()
decoder.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # for render
    # content_transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor()
    # ])

    style_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    #for render
    # style_transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor()
    # ])

    content_image = content_transform(content_image).unsqueeze(0).to(device)  # unsqueeze used to do the 0 because the model expects a batch dimension so earlier we added a batch dimension to the content image using unsqueeze(0) and now why we are doing unsqueeze(0) to the style image is because the model expects a batch dimension for both content and style images. The model is designed to process batches of images, even if we are only passing a single image. By adding a batch dimension, we ensure that the input shape matches what the model expects, which is typically (batch_size, channels, height, width). In this case, unsqueeze(0) adds a new dimension at the 0th index, effectively creating a batch of size 1 for both content and style images.
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image , is_test=True)
        style_feats = encoder(style_image , is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats , style_feats)
        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats  # why this because we want to control the degree of stylization. The alpha parameter allows us to blend the stylized features with the original content features. When alpha is 1, we get full stylization, and when alpha is 0, we retain the original content features. By adjusting alpha, we can achieve a balance between the content and style in the final output. if alpha is 0.5, we get an equal blend of content and style features, resulting in a stylized image that retains some content information while incorporating style characteristics.

        stylized_image = decoder(stylized_feats)

    return stylized_image




def save_image(image, path):
    # here we also say image is a tensor and it is cleaner way to say it tensor
    # output stylized images are currently tensors from the GPU
    image = image.cpu().clone().squeeze(0)  # remove the batch dimension
    # image = transforms.ToPILImage()(image)
    image = image.clamp(0, 1)  # ensure the pixel values are in the range [0, 1]
    image = transforms.ToPILImage()(image)
    image.save(path)








@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    result_filename = None # i created
    error= None

    if form.validate_on_submit():
        print("✅ FORM VALIDATED")
        # print("CONTENT:", form.content.data)
        # print("CONTENT FILENAME:", form.content.data.filename if form.content.data else None)
        # print("STYLE:", form.style.data)
        # print("STYLE FILENAME:", form.style.data.filename if form.style.data else None)

        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(
                    os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
                )
                form.content_path.data = content_filename
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(
                    os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
                )
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data

        if content_filename and style_filename:
            print("✅ BOTH FILES SAVED")
            # print("Content path:", content_filename)
            # print("Style path:", style_filename)

            content_path = os.path.join(
                app.config['UPLOAD_FOLDER'], content_filename
            )
            style_path = os.path.join(
                app.config['UPLOAD_FOLDER'], style_filename
            )

            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')

                alpha = float(form.alpha.data)

                print("🚀 STARTING STYLE TRANSFER")

                stylized_image = style_transfer( content_image, style_image, encoder, decoder, alpha, device
                )

                print("✅ STYLE TRANSFER COMPLETED")

                result_filename = f'stylized_{content_filename}'
                result_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], result_filename
                )

                save_image(stylized_image, result_path)

                print("✅ RESULT SAVED:", result_path)

                result_image = result_filename

            except Exception as e:
                print("❌ STYLE TRANSFER ERROR:", repr(e))
                error = f"An error occurred during style transfer: {str(e)}"

    
    else:
        print("❌ FORM VALIDATION FAILED")
        print("ERRORS:", form.errors)
        if request.method == 'POST':
            if not content_filename:
                error = "Please upload a content image."
            elif not style_filename:
                error = "Please upload a style image."



    return render_template('index.html', form=form, result_image=result_image, content_filename=content_filename, style_filename=style_filename, error=error)



@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('static/examples', filename)




if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)