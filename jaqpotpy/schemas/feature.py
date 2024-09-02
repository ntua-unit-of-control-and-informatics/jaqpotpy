"""Author: Ioannis Pitoskas (jpitoskas@gmail.com)"""


class Feature:
    @staticmethod
    def get_feature_names_and_possible_values_from_column_names(input_list):
        structured_data = {}

        for column_name in input_list:
            if "." in column_name:
                feature_name, categorical_value = column_name.split(".")
                if feature_name not in structured_data:
                    structured_data[feature_name] = []
                structured_data[feature_name].append(categorical_value)
            else:
                structured_data[column_name] = []

        feature_names_and_possible_values = [
            {"name": feature_name, "possibleValues": possible_values}
            for feature_name, possible_values in structured_data.items()
        ]

        return feature_names_and_possible_values

    def __init__(
        self,
        name,
        featureDependency,
        possibleValues,
        featureType,
        meta=None,
        description=None,
    ):
        if not isinstance(name, str):
            raise ValueError("'name' should be of type str")

        if featureDependency not in ["DEPENDENT", "INDEPENDENT"]:
            raise ValueError(
                f"possible values for 'featureDependency' are {['DEPENDENT', 'INDEPENDENT']}"
            )

        if possibleValues is not None and not isinstance(possibleValues, list):
            raise ValueError("'possibleValues' should be of type list")

        if possibleValues is not None:
            for possibleValue in possibleValues:
                if not isinstance(possibleValue, str):
                    raise ValueError("items in 'possibleValues' should be of type str")

        if meta is not None and not isinstance(meta, dict):
            raise ValueError("'meta' should be of type dict")

        featureType_allowed_values = [
            "CATEGORICAL",
            "SMILES",
            "INTEGER",
            "FLOAT",
            "TEXT",
        ]
        if featureType is not None and featureType not in featureType_allowed_values:
            raise ValueError(
                f"possible values for 'featureType' are {featureType_allowed_values}"
            )

        if description is not None and not isinstance(description, str):
            raise ValueError("'description' should be of type str")

        self.name = name
        self.featureDependency = featureDependency

        self.possibleValues = possibleValues if possibleValues is not None else []

        if self.possibleValues == [] and featureType is None:
            self.featureType = "FLOAT"
        if self.possibleValues != [] and featureType is None:
            self.featureType = "CATEGORICAL"
        if self.possibleValues != [] and featureType in ["FLOAT", "SMILES"]:
            raise ValueError(
                f"Feature of featureType '{featureType}' cannot have a finite set of possible values"
            )
        if self.possibleValues == [] and featureType in ["CATEGORICAL"]:
            raise ValueError(
                f"Feature of featureType '{featureType}' must have a non-empty set of possible values"
            )

        self.featureType = featureType

        self.meta = meta if meta is not None else {}
        self.description = description if description is not None else ""

    def to_json(self):
        feature_dict = {
            "meta": self.meta,
            "name": self.name,
            "description": self.description,
            "featureType": self.featureType,
            "featureDependency": self.featureDependency,
            "possibleValues": self.possibleValues,
        }
        return feature_dict

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_json()})"
