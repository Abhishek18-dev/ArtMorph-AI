from pathlib import Path

import streamlit as st
import torch
from PIL import Image

# Reuse only the model classes and utility functions.
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
EXAMPLES_FOLDER = BASE_DIR / "static" / "examples"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


# Load models once and keep them cached across Streamlit reruns.
@st.cache_resource
def load_models():

    # Same device selection as the Flask application.
    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        try:
            import torch_directml
            device = torch_directml.device()
        except Exception:
            device = torch.device("cpu")

    # Load encoder.
    encoder = VGGEncoder(
        str(BASE_DIR / "vgg_normalised.pth")
    ).to(device)

    # Load decoder.
    decoder = Decoder().to(device)

    checkpoint = torch.load(
        BASE_DIR / "decoder_final.pth",
        map_location="cpu"
    )

    decoder.load_state_dict(checkpoint["decoder"])
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    print(f"Using device: {device}")

    return encoder, decoder, device


# Models are loaded only once.
encoder, decoder, device = load_models()


# Same allowed file logic as the Flask application.
ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "gif",
    "bmp",
    "tiff",
    "webp",
}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# Same style-transfer logic.
# 256x256 preprocessing and AdaIN flow are unchanged.
def style_transfer(
    content_image,
    style_image,
    encoder,
    decoder,
    alpha,
    device
):
    # for cpu
    # content_image = content_image.resize((256, 256))
    # style_image = style_image.resize((256, 256))

    # for GPU
    content_image = content_image.resize((512, 512))
    style_image = style_image.resize((512, 512))

    transform = __import__(
        "torchvision.transforms",
        fromlist=["transforms"]
    ).ToTensor()

    content_image = transform(content_image).unsqueeze(0).to(device)
    style_image = transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(
            content_feats,
            style_feats
        )

        stylized_feats = (
            alpha * stylized_feats
            + (1 - alpha) * content_feats
        )

        stylized_image = decoder(stylized_feats)

    return stylized_image


# Same image-saving logic.
def save_image(tensor, path):
    image = tensor.detach().cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)

    image = image.permute(1, 2, 0).numpy()

    image = Image.fromarray(
        (image * 255).astype("uint8")
    )

    image.save(path)


# Small amount of CSS to keep the Streamlit UI close to the original UI.
st.markdown("""
<style>
    .stApp {
        background: #0f172a;
        color: #e2e8f0;
    }

    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 2.5rem;
    }

    .section-title {
        text-align: center;
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .result-title {
        text-align: center;
        color: #34d399;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 1rem 0;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 50px;
        border: none;
        padding: 0.7rem 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white;
    }

    div.stDownloadButton > button {
        width: 100%;
        border-radius: 50px;
    }
</style>
""", unsafe_allow_html=True)


# Title.
st.markdown(
    '<div class="main-title">ArtMorph AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Redefine Reality with AI-Powered Artistry'
    '</div>',
    unsafe_allow_html=True
)


# Upload content image.
content_file = st.file_uploader(
    "Content Source",
    type=[
        "png",
        "jpg",
        "jpeg",
        "gif",
        "bmp",
        "tiff",
        "webp"
    ],
    key="content"
)

# Upload style image.
style_file = st.file_uploader(
    "Style Reference",
    type=[
        "png",
        "jpg",
        "jpeg",
        "gif",
        "bmp",
        "tiff",
        "webp"
    ],
    key="style"
)


# Preview selected images.
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        '<div class="section-title">Content Source</div>',
        unsafe_allow_html=True
    )

    if content_file:
        st.image(content_file, width="stretch")
    else:
        st.info("Select Content Image")


with col2:
    st.markdown(
        '<div class="section-title">Style Reference</div>',
        unsafe_allow_html=True
    )

    if style_file:
        st.image(style_file, width="stretch")
    else:
        st.info("Select Style Image")


# Style strength.
st.markdown("### Style Strength")

alpha = st.slider(
    "Style Strength",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.1,
    label_visibility="collapsed"
)


# Style transfer.
if st.button("Transfer Style"):

    if not content_file or not style_file:
        st.error(
            "Please upload both a content image and a style image."
        )

    elif (
        not allowed_file(content_file.name)
        or not allowed_file(style_file.name)
    ):
        st.error("Unsupported image format.")

    else:

        # Keep the same filenames and upload location.
        content_filename = Path(content_file.name).name
        style_filename = Path(style_file.name).name

        content_path = UPLOAD_FOLDER / content_filename
        style_path = UPLOAD_FOLDER / style_filename

        # Save uploaded files.
        with open(content_path, "wb") as f:
            f.write(content_file.getbuffer())

        with open(style_path, "wb") as f:
            f.write(style_file.getbuffer())

        try:
            # Open images as RGB.
            content_image = Image.open(content_path).convert("RGB")
            style_image = Image.open(style_path).convert("RGB")

            # Run AdaIN.
            with st.spinner("Neural Network is dreaming..."):

                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    encoder,
                    decoder,
                    float(alpha),
                    device
                )

            # Same output naming convention.
            result_filename = f"stylized_{content_filename}"
            result_path = UPLOAD_FOLDER / result_filename

            # Save output.
            save_image(
                stylized_image,
                result_path
            )

            # Show result.
            st.markdown(
                '<div class="result-title">Stylized Result</div>',
                unsafe_allow_html=True
            )

            st.image(
                result_path,
                width="stretch"
            )

            # Download result.
            with open(result_path, "rb") as f:
                st.download_button(
                    "Download Result",
                    data=f,
                    file_name=result_filename,
                    mime="image/png"
                )

        except Exception as e:
            st.error(
                f"An error occurred during style transfer: {str(e)}"
            )


# Examples.
st.markdown("---")
st.markdown("## Examples")

examples = [
    (
        "salman-khan.webp",
        "style2.jpg",
        "stylized_salman-khan-2.webp"
    ),
    (
        "salman-khan.webp",
        "style1.jpg",
        "stylized_salman-khan-1.webp"
    ),
]

for content_name, style_name, output_name in examples:

    content_path = EXAMPLES_FOLDER / content_name
    style_path = EXAMPLES_FOLDER / style_name
    output_path = EXAMPLES_FOLDER / output_name

    if (
        content_path.exists()
        and style_path.exists()
        and output_path.exists()
    ):

        col1, col2, col3 = st.columns([1, 0.15, 1])

        with col1:
            st.caption("CONTENT + STYLE")
            st.image(
                [content_path, style_path],
                width="stretch"
            )

        with col2:
            st.markdown("### →")

        with col3:
            st.caption("OUTPUT")
            st.image(
                output_path,
                width="stretch"
            )


# FAQs.
st.markdown("---")
st.markdown("## FAQs")

with st.expander("1. Is it a pretrained model?"):
    st.write("No, we train a model ourselves.")

with st.expander("2. Is it a free platform?"):
    st.write(
        "This demo is currently available as a free experience."
    )

with st.expander("3. Which styles of painting can be used?"):
    st.write(
        "You can use almost any painting style image such as "
        "impressionist, abstract, cubist, watercolor, sketch, "
        "and more."
    )

with st.expander("4. Tech stack of the project?"):
    st.write("Python, PyTorch, Flask.")

with st.expander("5. Which dataset / how big data?"):
    st.write(
        "The model is trained on large-scale image datasets."
    )