from jaqpot_api_client import (
    Feature,
    DockerConfig,
    ModelVisibility,
    FeatureType,
    FeaturePossibleValue,
)
from jaqpotpy.models.docker_model import DockerModel
from jaqpotpy import Jaqpot

# All available Pyrosage models with descriptions
AVAILABLE_MODELS = [
    # Classification models
    FeaturePossibleValue(
        value="AMES", description="Mutagenicity prediction (Ames test)"
    ),
    FeaturePossibleValue(
        value="Endocrine_Disruption_NR-AR",
        description="Androgen receptor disruption prediction",
    ),
    FeaturePossibleValue(
        value="Endocrine_Disruption_NR-AhR",
        description="Aryl hydrocarbon receptor disruption prediction",
    ),
    FeaturePossibleValue(
        value="Endocrine_Disruption_NR-ER",
        description="Estrogen receptor disruption prediction",
    ),
    FeaturePossibleValue(
        value="Endocrine_Disruption_NR-aromatase",
        description="Aromatase disruption prediction",
    ),
    FeaturePossibleValue(
        value="Irritation_Corrosion_Eye_Corrosion",
        description="Eye corrosion prediction",
    ),
    FeaturePossibleValue(
        value="Irritation_Corrosion_Eye_Irritation",
        description="Eye irritation prediction",
    ),
    # Regression models
    FeaturePossibleValue(value="FBA", description="Bioaccumulation Factor prediction"),
    FeaturePossibleValue(value="FBC", description="Bioconcentration Factor prediction"),
    FeaturePossibleValue(
        value="kAOH", description="Aqueous hydroxyl rate constant prediction"
    ),
    FeaturePossibleValue(value="KH", description="Henry's Law Constant prediction"),
    FeaturePossibleValue(
        value="KOA", description="Octanol-Air Partition Coefficient prediction"
    ),
    FeaturePossibleValue(
        value="KOC", description="Soil/Water Partition Coefficient prediction"
    ),
    FeaturePossibleValue(
        value="KOW", description="Octanol-Water Partition Coefficient (LogP) prediction"
    ),
    FeaturePossibleValue(
        value="LC50", description="Aquatic toxicity (LC50) prediction"
    ),
    FeaturePossibleValue(
        value="LD50_Zhu", description="Acute oral toxicity (LD50) prediction"
    ),
    FeaturePossibleValue(value="pKa_acidic", description="Acidic pKa prediction"),
    FeaturePossibleValue(value="pKa_basic", description="Basic pKa prediction"),
    FeaturePossibleValue(
        value="PLV", description="Vapor pressure related property prediction"
    ),
    FeaturePossibleValue(value="SW", description="Water solubility prediction"),
    FeaturePossibleValue(value="tbiodeg", description="Biodegradation time prediction"),
    FeaturePossibleValue(
        value="TBP", description="Biodegradation related property prediction"
    ),
    FeaturePossibleValue(
        value="tfishbio", description="Fish bioaccumulation time prediction"
    ),
    FeaturePossibleValue(
        value="TMP", description="Melting point related property prediction"
    ),
]

# Define independent features
independent_features = [
    Feature(
        key="smiles",
        name="SMILES",
        feature_type=FeatureType.STRING,
        description="SMILES string representation of the molecular structure",
    ),
    Feature(
        key="model_name",
        name="Model Name",
        feature_type=FeatureType.CATEGORICAL,
        description="Name of the Pyrosage model to use for prediction",
        possible_values=AVAILABLE_MODELS,
    ),
]

# Define dependent features
dependent_features = [
    Feature(
        key="prediction",
        name="Prediction",
        feature_type=FeatureType.FLOAT,
        description="Model prediction value (probability for classification, continuous value for regression)",
    ),
]

# Create DockerConfig for Pyrosage model
docker_config = DockerConfig(
    app_name="pyrosage-model",
    docker_image="upcintua/jaqpot-pyrosage-model",  # Update this with your actual Docker image
)

# Instantiate the DockerModel
jaqpot_model = DockerModel(
    independent_features=independent_features,
    dependent_features=dependent_features,
    docker_config=docker_config,
)

# Create an instance of Jaqpot (ensure local Jaqpot is running)
jaqpot = Jaqpot()

# Login to Jaqpot (requires authorization from browser)
print("Logging into Jaqpot...")
jaqpot.login()

# Deploy the model on Jaqpot
print("Deploying Pyrosage model on Jaqpot...")
print(f"Available models: {', '.join([model.value for model in AVAILABLE_MODELS])}")

jaqpot_model.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="Pyrosage Environmental & Toxicity Predictors",
    description="""
    Pyrosage is a comprehensive collection of Graph Neural Network models for predicting environmental and toxicity properties of chemical compounds. 

    This service provides access to 24 different AttentiveFP-based models covering:

    Classification Models (7):
    - AMES: Mutagenicity prediction
    - Endocrine_Disruption_NR-AR: Androgen receptor disruption
    - Endocrine_Disruption_NR-AhR: Aryl hydrocarbon receptor disruption
    - Endocrine_Disruption_NR-ER: Estrogen receptor disruption  
    - Endocrine_Disruption_NR-aromatase: Aromatase disruption
    - Irritation_Corrosion_Eye_Corrosion: Eye corrosion prediction
    - Irritation_Corrosion_Eye_Irritation: Eye irritation prediction

    Regression Models (17):
    - FBA: Bioaccumulation Factor
    - FBC: Bioconcentration Factor
    - kAOH: Aqueous hydroxyl rate
    - KH: Henry's Law Constant
    - KOA: Octanol-Air Partition Coefficient
    - KOC: Soil/Water Partition Coefficient
    - KOW: Octanol-Water Partition Coefficient (LogP)
    - LC50: Aquatic toxicity
    - LD50_Zhu: Acute oral toxicity
    - pKa_acidic: Acidic pKa
    - pKa_basic: Basic pKa
    - PLV: Vapor pressure related
    - SW: Water solubility
    - tbiodeg: Biodegradation time
    - TBP: Biodegradation related
    - tfishbio: Fish bioaccumulation time
    - TMP: Melting point related

    Usage: Provide a SMILES string and select the desired model for prediction.
    The models use enhanced molecular graphs with attention-based fingerprints for accurate property prediction.
    """,
    visibility=ModelVisibility.PUBLIC,
)

print("Model deployed successfully!")
print("\nTo use the model:")
print("1. Provide a SMILES string (e.g., 'CCO' for ethanol)")
print("2. Select a model name from the available options")
print("3. The model will return a prediction value")
print("\nFor classification models: prediction is a probability (0-1)")
print("For regression models: prediction is a continuous value in the respective unit")
