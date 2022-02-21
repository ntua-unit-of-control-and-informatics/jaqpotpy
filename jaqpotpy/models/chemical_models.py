from pydantic import BaseModel
from typing import Optional, Union, List


class ElementModel(BaseModel):
    annotation: Optional[str]
    atomic_number: Optional[int]
    atomic_radius: Optional[float]
    atomic_volume: Optional[float]
    block: Optional[str]
    boiling_point: Optional[float]
    density: Optional[float]
    description: Optional[str]
    dipole_polarizability: Optional[float]
    electron_affinity: Optional[float]
    electronic_configuration: Optional[str]
    evaporation_heat: Optional[float]
    fusion_heat: Optional[float]
    group_name: Optional[str]
    group_symbol: Optional[str]
    lattice_constant: Optional[float]
    lattice_structure: Optional[str]
    melting_point: Optional[float]
    name: Optional[str]
    period: Optional[int]
    series_name: Optional[str]
    series_color: Optional[str]
    specific_heat: Optional[float]
    symbol: Optional[str]
    thermal_conductivity: Optional[float]
    vdw_radius: Optional[float]
    covalent_radius_cordero: Optional[float]
    covalent_radius_pyykko: Optional[float]
    en_pauling: Optional[float]
    en_allen: Optional[float]
    jmol_color: Optional[str]
    cpk_color: Optional[str]
    proton_affinity: Optional[float]
    gas_basicity: Optional[float]
    heat_of_formation: Optional[float]
    c6: Optional[float]
    covalent_radius_bragg: Optional[float]
    vdw_radius_bondi: Optional[float]
    vdw_radius_truhlar: Optional[float]
    vdw_radius_rt: Optional[float]
    vdw_radius_batsanov: Optional[float]
    vdw_radius_dreiding: Optional[float]
    vdw_radius_uff: Optional[float]
    vdw_radius_mm3: Optional[float]
    abundance_crust: Optional[float]
    abundance_sea: Optional[float]
    molcas_gv_color: Optional[str]
    en_ghosh: Optional[float]
    vdw_radius_alvarez: Optional[float]
    c6_gb: Optional[float]
    atomic_weight: Optional[float]
    atomic_weight_uncertainty: Optional[float]
    is_monoisotopic: Optional[bool]
    is_radioactive: Optional[bool]
    cas: Optional[str]
    atomic_radius_rahm: Optional[float]
    geochemical_class: Optional[str]
    goldschmidt_class: Optional[str]
    metallic_radius: Optional[float]
    metallic_radius_c12: Optional[float]
    covalent_radius_pyykko_double: Optional[str]
    covalent_radius_pyykko_triple: Optional[str]
    discoverers: Optional[str]
    discovery_year: Optional[int]
    discovery_location: Optional[str]
    name_origin: Optional[str]
    sources: Optional[str]
    uses: Optional[str]
    mendeleev_number: Optional[int]
    dipole_polarizability_unc: Optional[float]
    pettifor_number: Optional[int]
    glawe_number: Optional[int]


class IonicRadiiModel(BaseModel):
    atomic_number: int
    charge: int
    econf: Optional[str]
    coordination: Optional[str]
    spin: Optional[str]
    crystal_radius: Optional[float]
    ionic_radius: Optional[float]
    origin: Optional[str]
    most_reliable: Optional[float]


class IonizationEnergiesModel(BaseModel):
    atomic_number: int
    degree: int
    energy: Optional[float]


class IsotopesModel(BaseModel):
    atomic_number: int
    mass: Optional[float]
    abundance: Optional[float]
    mass_number: Optional[int]
    mass_uncertainty: Optional[float]
    is_radioactive: Optional[int]
    half_life: Optional[float]
    half_life_unit: Optional[str]
    spin: Optional[float]
    g_factor: Optional[float]
    quadrupole_moment: Optional[float]


class OxidationStatesModel(BaseModel):
    atomic_number: int
    oxidation_state: Optional[int]


class ScreeningConstantsModel(BaseModel):
    atomic_number: Optional[int]
    n: Optional[int]
    s: Optional[str]
    screening: Optional[float]


