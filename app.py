import streamlit as st
import pickle
#from rdkit import Chem
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import AllChem
import numpy as np
from matplotlib import cm
#from ochem import mycalc
import pandas as pd
from sklearn.svm import SVC
#from xgboost import XGBClassifier
import warnings
from PIL import Image
from io import BytesIO

warnings.filterwarnings("ignore")


# Load the trained SVC model from pickle file
MODEL_FILE_alpha = 'best_model.pkl'  # Replace with your pickle file


# Function to load the model
def load_model(mymodel):
    with open(mymodel, 'rb') as f:
        model = pickle.load(f)
        #model.probability=True
        #print("I am here")
        #print(model)
    #model.probability = True
    return model

# Function to generate molecular fingerprint
def generate_fingerprint(smiles,rad):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=rad,nBits=1024,useFeatures=False,useChirality = True)
    arr = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return mol, arr

def fpFunction_a(m,atomId=-1):
    fp = SimilarityMaps.GetMorganFingerprint(
        m,
        atomId=atomId,
        radius=3,
        nBits=1024,
        useFeatures=False,
        useChirality=True
    )
    return fp


def getProba(fp, predictionFunction):
    return predictionFunction((fp,))[0][1]

def plot_similarity_map(mol, model, fpFunction):

    d = Draw.MolDraw2DCairo(600, 600)

    SimilarityMaps.GetSimilarityMapForModel(
        mol,
        fpFunction,
        lambda x: float(getProba(x, model.decision_function)),
        draw2d=d
    )

    d.FinishDrawing()

    # Convert RDKit PNG output into PIL image
    png_data = d.GetDrawingText()
    img = Image.open(BytesIO(png_data)).convert("RGB")

    return img






sdf1="DPPH_trainingSet_minimized.sdf"

#sdf2="LXRB_rs25_tr_minimized.sdf"


def applicability(smiles, mysdf, rad):

    similarities = []

    m = Chem.MolFromSmiles(smiles)

    if m is None:
        return None

    # Generate query fingerprint
    mg_ = AllChem.GetMorganFingerprintAsBitVect(
        m,
        radius=rad,
        nBits=1024,
        useFeatures=False,
        useChirality=True
    )

    # Compare with training set compounds
    for mol in Chem.SDMolSupplier(mysdf):

        if mol is None:
            continue

        mg = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=rad,
            nBits=1024,
            useFeatures=False,
            useChirality=True
        )

        similarities.append(
            DataStructs.FingerprintSimilarity(mg, mg_)
        )

    if not similarities:
        return None

    return max(similarities)



# Streamlit application
def main():
    st.title("DPP-4 PREDICTOR")

    st.subheader("This app predicts the binding potential of the chemical compounds towards Dipeptidyl peptidase-4 (DPP-4) using trained fingerprint-based models")
    st.write("Input single or multiple (separated by comma) SMILES notation(s) of chemical compound(s) to predict activity.")

    # Input SMILES notation
    smi = st.text_input("Enter SMILES Notation:", "")
    slist=smi.split(',')
    for smi in slist:
        if smi:
           # Generate fingerprint
           mol, fingerprint_a = generate_fingerprint(smi,3)
           #act_trans,fig2,l1,l2=mycalc('fatimaBest.pickle',smiles)

           if mol is None:
              st.error("Invalid SMILES notation. Please try again.")
           else:
               # Load the model
               model_a = load_model(MODEL_FILE_alpha)
               print(model_a)
               print("I am here")


               # Predict activity for fingerprint based model
               prediction = model_a.predict([fingerprint_a])
               st.write("## Predicted activity, AD and similarity map (as per fingerprint-based model)")
               st.write("## DPP-4 activity")
               activity = "Active (EC50/IC50 < 1000 nM)" if prediction[0] == 1 else "Inactive (EC50/IC50 >= 1000 nM)"               
               st.write(f"Predicted Activity as per fingerprint based model: **{activity}**")

               #Check applicability domain
               value=applicability(smi,sdf1,3)
               if value>=0.3:
                  st.write("The compound falls **within** AD of of the model (as per fingerprint based model)")
               else:
                  st.write("The compound falls **outside** AD of of the model (as per fingerprint based model)")
               st.write(f"Maximum similarity with training set compounds is: **{value:.3f}**")


               # Generate similarity map
               #d = Draw.MolDraw2DCairo(400, 400)


               #fig,_=SimilarityMaps.GetSimilarityMapForModel(mol, fpFunction, lambda x: getProba(x, model.predict_proba), colorMap=cm.PiYG_r)
               #st.pyplot(fig)
               res = plot_similarity_map(mol, model_a, fpFunction_a)

               st.image(res,caption="Molecular Similarity Map",use_container_width=True)
               st.markdown("**Colour scheme:**")
               st.markdown('<span style="color:green">Molecular environments that increase predicted probability of class 1 (Active)</span>', unsafe_allow_html=True)
               st.markdown('<span style="color:red">Molecular environments that decrease predicted probability of class 0 (Inctive)</span>', unsafe_allow_html=True)
               


               # Generate similarity map
               #d = Draw.MolDraw2DCairo(400, 400)


               



# Add a footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
    }
    </style>
    <div class="footer">
        Developed using Streamlit by Dr. Amit Kumar Halder, Professor, Dr. B. C. Roy College of Pharmacy and AHS, India | <a href="https://github.com/amitporto/DPP4_PREDICTOR" target="_blank">Github link</a>
    </div>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()


