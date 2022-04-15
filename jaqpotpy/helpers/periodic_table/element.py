from pydantic import parse_obj_as
from typing import Union, List
from jaqpotpy.entities.chemical_models import ElementModel, IonicRadiiModel, IsotopesModel, \
    IonizationEnergiesModel, OxidationStatesModel, ScreeningConstantsModel
from jaqpotpy.helpers.periodic_table.base_classes import PeriodicTable


class Element(PeriodicTable):
    def __init__(self, element: Union[str, int]):
        super().__init__()
        if isinstance(element, str):
            if len(element) < 3:
                res = self._find_one('SELECT * FROM elements WHERE symbol="{}"'.format(element))
            else:
                res = self._find_one('SELECT * FROM elements WHERE name="{}"'.format(element))
        else:
            res = self._find_one('SELECT * FROM elements WHERE atomic_number="{}"'.format(element))

        try:
            g = self._find_one('SELECT name, symbol FROM groups WHERE group_id={}'.format(res['group_id']))
        except:
            g = {'name': None, 'symbol': None}
        try:
            s = self._find_one('SELECT name,color FROM series WHERE id={}'.format(res['series_id']))
        except:
            s = {'name': None, 'color': None}

        res['group_name'] = g['name']
        res['group_symbol'] = g['symbol']
        res['series_name'] = s['name']
        res['series_color'] = s['color']

        self._element = parse_obj_as(ElementModel, {k: None if not (v == v) else v for k, v in res.items()})

    @property
    def parameters(self) -> ElementModel:
        return self._element

    @property
    def ionic_radii(self) -> List[IonicRadiiModel]:
        res = self._find_many('SELECT * FROM ionicradii WHERE atomic_number={}'.format(self._element.atomic_number))
        return [parse_obj_as(IonicRadiiModel, {k: None if not (v == v) else v for k, v in rad.items()}) for rad in res]

    @property
    def ionization_energies(self) -> List[IonizationEnergiesModel]:
        res = self._find_many('SELECT * FROM ionizationenergies WHERE atomic_number={}'.format(self._element.atomic_number))
        return [parse_obj_as(IonizationEnergiesModel, {k: None if not (v == v) else v for k, v in rad.items()}) for rad in res]

    @property
    def isotopes(self) -> List[IsotopesModel]:
        res = self._find_many('SELECT * FROM isotopes WHERE atomic_number={}'.format(self._element.atomic_number))
        return [parse_obj_as(IsotopesModel, {k: None if not (v == v) else v for k, v in rad.items()}) for rad in res]

    @property
    def oxidation_states(self) -> List[OxidationStatesModel]:
        res = self._find_many('SELECT * FROM oxidationstates WHERE atomic_number={}'.format(self._element.atomic_number))
        return [parse_obj_as(OxidationStatesModel, {k: None if not (v == v) else v for k, v in rad.items()}) for rad in res]

    @property
    def screeningconstants(self) -> List[ScreeningConstantsModel]:
        res = self._find_many('SELECT * FROM screeningconstants WHERE atomic_number={}'.format(self._element.atomic_number))
        return [parse_obj_as(ScreeningConstantsModel, {k: None if not (v == v) else v for k, v in rad.items()}) for rad in res]
