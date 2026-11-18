"""
DPP-4 PREDICTOR
Streamlit application for DPP-4 activity prediction using a trained
1024-bit Morgan (ECFP6; radius 3) fingerprint SVC model.

Expected files in the same directory as this script:
    best_model.pkl
    DPPH_trainingSet_minimized.sdf
"""

from pathlib import Path
from io import BytesIO
import pickle
import warnings

import numpy as np
import streamlit as st
from PIL import Image
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import SimilarityMaps

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Application configuration
# ---------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_FILE = APP_DIR / "best_model.pkl"
TRAINING_SDF = APP_DIR / "DPPH_trainingSet_minimized.sdf"

FP_RADIUS = 3
FP_BITS = 1024
AD_THRESHOLD = 0.30

st.set_page_config(
    page_title="DPP-4 Predictor",
    page_icon="🧬",
    layout="wide"
)


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading trained prediction model...")
def load_model(model_path: str):
    """Load and cache a trusted, previously trained pickle model."""
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Model file was not found: {path.name}. "
            "Place best_model.pkl in the application directory."
        )

    # Only load pickle files from a trusted source.
    with path.open("rb") as file:
        model = pickle.load(file)

    if not hasattr(model, "predict"):
        raise TypeError("The loaded pickle object does not provide predict().")

    return model


# ---------------------------------------------------------------------
# Molecular fingerprint generation
# ---------------------------------------------------------------------

def generate_fingerprint(smiles: str, radius: int = FP_RADIUS):
    """Generate a Morgan bit vector and its NumPy representation."""
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None, None

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=FP_BITS,
        useFeatures=False,
        useChirality=True
    )

    array = np.zeros((FP_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)

    return mol, fingerprint, array


def fp_function(mol, atom_id=-1):
    """
    RDKit SimilarityMaps fingerprint function.

    Keep these settings consistent with the model's training fingerprints.
    """
    return SimilarityMaps.GetMorganFingerprint(
        mol,
        atomId=atom_id,
        radius=FP_RADIUS,
        nBits=FP_BITS,
        useFeatures=False,
        useChirality=True
    )


# ---------------------------------------------------------------------
# SVC score handling for SimilarityMaps
# ---------------------------------------------------------------------

def fingerprint_to_array(fingerprint):
    """Convert an RDKit fingerprint to the model's 1024-bit input array."""
    array = np.zeros((FP_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array.reshape(1, -1)


def class_1_score(fingerprint, model):
    """
    Return a continuous score for class 1.

    If the SVC was fitted with probability=True, use its class-1
    probability. Otherwise use decision_function(), which does not
    require probability calibration.
    """
    x = fingerprint_to_array(fingerprint)

    classes = list(getattr(model, "classes_", []))
    if not classes:
        raise ValueError(
            "The loaded model has no classes_ attribute. "
            "It may not be a fitted classifier."
        )

    if 1 not in classes:
        raise ValueError(
            f"Class label 1 was not found in model.classes_: {classes}"
        )

    class_1_index = classes.index(1)

    probability_is_available = (
        bool(getattr(model, "probability", False))
        and hasattr(model, "probA_")
        and hasattr(model, "probB_")
        and hasattr(model, "predict_proba")
    )

    if probability_is_available:
        return float(model.predict_proba(x)[0, class_1_index])

    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(x)).reshape(-1)

        # For binary SVC, a positive decision score points to classes_[1].
        if len(classes) == 2:
            return float(score[0] if class_1_index == 1 else -score[0])

        raise ValueError(
            "This application currently expects a binary SVC classifier."
        )

    raise ValueError(
        "The model supports neither calibrated probabilities nor "
        "decision_function()."
    )


def plot_similarity_map(mol, model):
    """Create a PNG molecular contribution map and return it as a PIL image."""
    drawer = Draw.MolDraw2DCairo(600, 600)

    SimilarityMaps.GetSimilarityMapForModel(
        mol,
        fp_function,
        lambda fingerprint: class_1_score(fingerprint, model),
        draw2d=drawer
    )

    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()

    # Decode the PNG bytes explicitly before passing the image to Streamlit.
    image = Image.open(BytesIO(png_bytes)).convert("RGB")
    return image


# ---------------------------------------------------------------------
# Applicability domain
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_training_fingerprints(sdf_path: str, radius: int, n_bits: int):
    """Read and cache valid training-set Morgan fingerprints from an SDF."""
    path = Path(sdf_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Training SDF was not found: {path.name}. "
            "Place it in the application directory."
        )

    fingerprints = []

    supplier = Chem.SDMolSupplier(str(path), removeHs=True)
    for mol in supplier:
        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=n_bits,
            useFeatures=False,
            useChirality=True
        )
        fingerprints.append(fp)

    if not fingerprints:
        raise ValueError(
            "No valid molecules were found in the training-set SDF file."
        )

    return fingerprints


def applicability(smiles: str, sdf_path: str, radius: int = FP_RADIUS):
    """Return the maximum Tanimoto similarity to the training compounds."""
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Cannot calculate applicability for invalid SMILES.")

    query_fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=FP_BITS,
        useFeatures=False,
        useChirality=True
    )

    training_fps = load_training_fingerprints(
        str(sdf_path),
        radius,
        FP_BITS
    )

    similarities = DataStructs.BulkTanimotoSimilarity(
        query_fp,
        training_fps
    )

    return float(max(similarities))


# ---------------------------------------------------------------------
# Prediction and display
# ---------------------------------------------------------------------

def predict_one(smiles: str, model):
    """Predict one compound and display its results."""
    mol, _fingerprint, fingerprint_array = generate_fingerprint(smiles)

    if mol is None:
        st.error(f"Invalid SMILES notation: {smiles}")
        return

    prediction = model.predict(fingerprint_array.reshape(1, -1))
    predicted_class = prediction[0]

    st.markdown("## Predicted activity, AD and similarity map")
    st.markdown("### DPP-4 activity")

    activity = (
        "Active (EC50/IC50 < 1000 nM)"
        if predicted_class == 1
        else "Inactive (EC50/IC50 >= 1000 nM)"
    )

    st.write(
        f"Predicted activity as per fingerprint-based model: **{activity}**"
    )

    # Applicability-domain calculation
    try:
        similarity = applicability(str(smiles), str(TRAINING_SDF), FP_RADIUS)

        if similarity >= AD_THRESHOLD:
            st.write(
                "The compound falls **within** the AD of the model "
                "(as per fingerprint-based model)."
            )
        else:
            st.write(
                "The compound falls **outside** the AD of the model "
                "(as per fingerprint-based model)."
            )

        st.write(
            f"Maximum similarity with training-set compounds: "
            f"**{similarity:.3f}**"
        )

    except Exception as exc:
        st.error(f"Applicability-domain calculation failed: {exc}")

    # Similarity map
    st.markdown("### Molecular similarity map")

    try:
        with st.spinner("Generating molecular similarity map..."):
            image = plot_similarity_map(mol, model)

        st.image(
            image,
            caption="Atom-level contribution map for the predicted class-1 score",
            use_container_width=True
        )

        if (
            bool(getattr(model, "probability", False))
            and hasattr(model, "probA_")
            and hasattr(model, "probB_")
        ):
            st.caption(
                "The map is based on the SVC class-1 probability."
            )
        else:
            st.caption(
                "The map is based on the SVC decision-function score, "
                "not a calibrated probability."
            )

    except Exception as exc:
        st.error(f"Could not generate the molecular similarity map: {exc}")


# ---------------------------------------------------------------------
# Main Streamlit application
# ---------------------------------------------------------------------

def main():
    st.title("DPP-4 PREDICTOR")

    st.subheader(
        "This app predicts the binding potential of chemical compounds "
        "towards Dipeptidyl peptidase-4 (DPP-4) using trained "
        "fingerprint-based models."
    )

    st.write(
        "Input a single SMILES notation or multiple SMILES notations "
        "separated by commas."
    )

    smiles_input = st.text_input("Enter SMILES notation:", "")

    if not smiles_input.strip():
        return

    smiles_list = [
        item.strip()
        for item in smiles_input.split(",")
        if item.strip()
    ]

    try:
        model = load_model(str(MODEL_FILE))
    except Exception as exc:
        st.error(f"Could not load the trained model: {exc}")
        st.stop()

    for index, smiles in enumerate(smiles_list):
        if len(smiles_list) > 1:
            st.divider()
            st.markdown(f"## Compound {index + 1}")

        predict_one(smiles, model)


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
    .footer {
        position: relative;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        margin-top: 30px;
        font-size: 13px;
        color: #333;
        border-radius: 5px;
    }
    .footer a {
        color: #2457a7;
        text-decoration: none;
    }
    </style>
    <div class="footer">
        Developed using Streamlit by Dr. Amit Kumar Halder,
        Professor, Dr. B. C. Roy College of Pharmacy and AHS, India |
        <a href="https://github.com/amitporto/DPP4_PREDICTOR"
           target="_blank">GitHub link</a>
    </div>
    """,
    unsafe_allow_html=True
)


if __name__ == "__main__":
    main()
