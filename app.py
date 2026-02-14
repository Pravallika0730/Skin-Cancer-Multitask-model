import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import timm
from torchvision.ops import FeaturePyramidNetwork
import torchvision.transforms as T
import cv2

# -------------------------
# Constants
# -------------------------
CLASS_ORDER = ['nevus', 'melanoma', 'seborrheic_keratosis']

FEATURE_NAMES = [
    'pigment_network',
    'streaks',
    'negative_network',
    'globules',
    'milia_like_cyst',
    'blotches',
    'rosettes'
]

NUM_CLASSES = len(CLASS_ORDER)
NUM_FEATURES = len(FEATURE_NAMES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Model Definition (MATCHES CHECKPOINT)
# -------------------------
class SimpleDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MultiTaskNet(nn.Module):
    def __init__(self, backbone_name='efficientnet_b4', pretrained=False):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        backbone_out = self.backbone.feature_info.channels()
        self.fpn = FeaturePyramidNetwork(backbone_out, 256)

        # ✅ Correct decoder names
        self.dec_b3 = SimpleDecoderBlock(256, 128)
        self.dec_b2 = SimpleDecoderBlock(128, 64)
        self.dec_b1 = SimpleDecoderBlock(64, 32)

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        fc_in = backbone_out[-1]

        self.class_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )

        # ✅ Correct feature head name
        self.feat_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_FEATURES)
        )

    def forward(self, x):
        feats = self.backbone(x)
        feats_dict = {f'p{i+1}': feats[i] for i in range(len(feats))}
        fpn_out = self.fpn(feats_dict)

        largest = sorted(
            fpn_out.values(),
            key=lambda f: f.shape[-2] * f.shape[-1],
            reverse=True
        )[0]

        x = self.dec_b3(largest)
        x = nn.functional.interpolate(x, scale_factor=2)

        x = self.dec_b2(x)
        x = nn.functional.interpolate(x, scale_factor=2)

        x = self.dec_b1(x)
        x = nn.functional.interpolate(x, size=(224, 224))

        seg = self.seg_head(x)

        pooled = self.pool(feats[-1])
        cls = self.class_head(pooled)
        feat = self.feat_head(pooled)

        return seg, cls, feat


# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    model = MultiTaskNet()

    checkpoint = torch.load(
        "multitask_model.pth",
        map_location=DEVICE,
        weights_only=False  # PyTorch 2.6 fix
    )

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

# -------------------------
# Image Preprocessing
# -------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# Streamlit UI
# -------------------------
st.title("🧬 Skin Cancer Multi-Task AI")
st.write("Classification • Dermoscopic Features • Lesion Segmentation")

uploaded = st.file_uploader(
    "Upload a dermoscopic image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        seg, cls, feat = model(x)

    # ---- Classification ----
    probs = torch.softmax(cls, dim=1)[0].cpu().numpy()
    pred_class = CLASS_ORDER[np.argmax(probs)]

    st.subheader("🔬 Diagnosis")
    for i, c in enumerate(CLASS_ORDER):
        st.write(f"{c}: **{probs[i]*100:.2f}%**")

    st.success(f"🩺 Predicted Class: **{pred_class.upper()}**")

    # ---- Feature Prediction ----
    st.subheader("🧠 Dermoscopic Features")
    feat_probs = torch.sigmoid(feat)[0].cpu().numpy()

    for i, f in enumerate(FEATURE_NAMES):
        st.write(f"{f}: **{feat_probs[i]*100:.1f}%**")

    # ---- Segmentation ----
    st.subheader("🩻 Lesion Segmentation")
    mask = torch.sigmoid(seg)[0, 0].cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255

    img_np = np.array(image.resize((224, 224)))
    overlay = img_np.copy()
    overlay[mask == 255] = [255, 0, 0]

    blended = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)
    st.image(blended, caption="Segmentation Overlay", use_container_width=True)
