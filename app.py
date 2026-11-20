from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile

import numpy as np
import streamlit as st
import torch
import SimpleITK as sitk
from PIL import Image

from core.model import UNet
from core.inference import predict_volume_masks

st.set_page_config(page_title="ProstateSeg AI", page_icon="🧠", layout="wide")
st.title("ProstateSeg AI")
st.caption("Protótipo de pesquisa • Segmentação Whole Gland em RM T2 • Não utilizar para diagnóstico")

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "unet_whole_gland_grouped_v2_best.pth"


@st.cache_resource(show_spinner="Carregando U-Net treinada...")
def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Carregue apenas checkpoints de fontes confiáveis.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device, checkpoint.get("epoch")


def load_uploaded_image(upload):
    name = upload.name.lower()
    if name.endswith(".nii.gz"):
        suffix = ".nii.gz"
    elif name.endswith(".nii"):
        suffix = ".nii"
    elif name.endswith(".mha"):
        suffix = ".mha"
    elif name.endswith(".nrrd"):
        suffix = ".nrrd"
    else:
        raise ValueError("Formato não suportado. Envie .nii, .nii.gz, .mha ou .nrrd")
    # SimpleITK trabalha com arquivos: tempfile é excluído imediatamente.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / ("input" + suffix)
        path.write_bytes(upload.getvalue())
        image = sitk.ReadImage(str(path))
        if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("A imagem precisa ser um volume 3D de escala de cinza.")
        volume = sitk.GetArrayFromImage(image)
        if not np.issubdtype(volume.dtype, np.number) or not np.isfinite(volume).all():
            raise ValueError("A imagem contém valores inválidos.")
        return image, volume


def mask_bytes(mask: np.ndarray, reference: sitk.Image, suffix: str):
    pred_image = sitk.GetImageFromArray(mask.astype(np.uint8))
    pred_image.CopyInformation(reference)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / ("whole_gland_mask" + suffix)
        sitk.WriteImage(pred_image, str(path))
        return path.read_bytes()


def grayscale(slice_2d):
    arr = np.asarray(slice_2d, dtype=np.float32)
    lo, hi = np.percentile(arr, (1, 99))
    if hi <= lo:
        return np.zeros((*arr.shape, 3), dtype=np.uint8)
    arr = (np.clip(arr, lo, hi) - lo) / (hi - lo)
    gray = (arr * 255).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def overlay_rgb(image_2d, mask_2d, alpha):
    rgb = grayscale(image_2d)
    rgb2 = rgb.astype(np.float32)
    foreground = mask_2d.astype(bool)
    rgb2[foreground] = (1-alpha)*rgb2[foreground] + alpha*np.array([0, 220, 180])
    return rgb2.astype(np.uint8)


def png_bytes(rgb):
    out = BytesIO()
    Image.fromarray(rgb).save(out, format="PNG")
    return out.getvalue()


with st.sidebar:
    st.subheader("Configurações")
    threshold = st.slider("Limiar de segmentação", 0.05, 0.95, 0.50, 0.05)
    alpha = st.slider("Transparência da máscara", 0.1, 0.9, 0.40, 0.05)
    st.info("Modelo: U-Net 2D treinada em T2. A inferência percorre todos os cortes axiais.")

if not CHECKPOINT.exists():
    st.error("Checkpoint não encontrado. Copie unet_whole_gland_grouped_v2_best.pth para checkpoints/ (veja README).")
    st.stop()

upload = st.file_uploader("Carregue um volume de RM T2", type=["nii", "gz", "mha", "nrrd"], help="É necessário um volume 3D. Uma captura PNG/JPG não preserva a geometria médica.")
if upload is None:
    st.info("Envie uma RM T2 para visualizar e segmentar a próstata.")
    st.stop()

try:
    # Recarrega somente se houver um novo upload.
    image, volume = load_uploaded_image(upload)
    st.write(f"**Dimensões:** {volume.shape[0]} cortes × {volume.shape[1]} × {volume.shape[2]} pixels")
    st.write(f"**Espaçamento (x, y, z):** {tuple(round(x, 3) for x in image.GetSpacing())} mm")
except Exception as exc:
    st.error(f"Não foi possível ler a RM: {exc}")
    st.stop()

upload_id = (upload.name, upload.size, hash(upload.getvalue()))
settings = (upload_id, threshold)
if st.session_state.get("result_settings") != settings:
    st.session_state.pop("predicted_mask", None)

if st.button("Segmentar próstata", type="primary"):
    try:
        model, device, epoch = load_model(str(CHECKPOINT))
        with st.spinner("Executando a inferência sobre todos os cortes..."):
            mask = predict_volume_masks(model=model, volume=volume, device=device, threshold=threshold, image_size=(256, 256))
        st.session_state.predicted_mask = mask
        st.session_state.result_settings = settings
        st.success(f"Inferência concluída ({device}; checkpoint época {epoch}).")
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        st.error(f"Erro na inferência: {exc}")

mask = st.session_state.get("predicted_mask")
if mask is None:
    idx = st.slider("Corte axial", 0, volume.shape[0]-1, volume.shape[0]//2)
    st.image(grayscale(volume[idx]), caption=f"RM T2 • corte {idx+1}/{volume.shape[0]}", width="stretch")
    st.stop()

if mask.shape != volume.shape:
    st.error("A máscara e o volume não têm as mesmas dimensões.")
    st.stop()

positive = np.flatnonzero(mask.reshape(mask.shape[0], -1).any(axis=1))
def_idx = int(positive[len(positive)//2]) if len(positive) else volume.shape[0]//2
idx = st.slider("Corte axial", 0, volume.shape[0]-1, def_idx)

col1, col2, col3 = st.columns(3)
with col1:
    st.image(grayscale(volume[idx]), caption="RM T2 original", width="stretch")
with col2:
    st.image(np.repeat((mask[idx]*255).astype(np.uint8)[:, :, None], 3, axis=2), caption="Máscara binária", width="stretch")
with col3:
    overlay = overlay_rgb(volume[idx], mask[idx], alpha)
    st.image(overlay, caption="Sobreposição", width="stretch")

st.caption(f"Cortes com segmentação: {len(positive)}/{len(mask)} • Voxels positivos: {int(mask.sum()):,}".replace(",", "."))
if not len(positive):
    st.warning("Nenhuma região foi detectada com esse limiar. Revise a imagem, o pré-processamento e o limiar antes de interpretar o resultado.")

st.subheader("Exportar resultado")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.download_button("Máscara NIfTI (.nii.gz)", data=mask_bytes(mask, image, ".nii.gz"), file_name="whole_gland_mask.nii.gz", mime="application/gzip")
with col_b:
    st.download_button("Máscara MetaImage (.mha)", data=mask_bytes(mask, image, ".mha"), file_name="whole_gland_mask.mha", mime="application/octet-stream")
with col_c:
    st.download_button("Sobreposição PNG (corte atual)", data=png_bytes(overlay), file_name=f"overlay_slice_{idx:03d}.png", mime="image/png")

st.warning("Uso exclusivo para pesquisa e demonstração. A saída identifica Whole Gland, não câncer de próstata; não constitui laudo ou diagnóstico.")
