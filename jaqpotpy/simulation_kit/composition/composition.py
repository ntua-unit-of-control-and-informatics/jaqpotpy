# from jaqpotpy.simulation_kit import Element
# from jaqpotpy.models.chemical_models import ElementModel
# from typing import Union, Tuple, Dict, List, Generator
# import re
# from jaqpotpy.simulation_kit.utils.utils import formula_double_format, gcd_float
# import warnings
# import collections
#
#
# class Composition(object):
#     """
#     Represents a Composition, which is essentially a {element:amount} mapping
#     type. Composition is written to be immutable and hashable,
#     unlike a standard Python dict.
#     Note that the key can be either an Element or a symbol of an element.
#
#     Works almost completely like a standard python dictionary, except that
#     __getitem__ is overridden to return 0 when an element is not found.
#     (somewhat like a defaultdict, except it is immutable).
#     Also adds more convenience methods relevant to compositions, e.g.,
#     get_fraction.
#
#     It should also be noted that many Composition related functionality takes
#     in a standard string as a convenient input. For example,
#     even though the internal representation of a Fe2O3 composition is
#     {Element("Fe"): 2, Element("O"): 3}, you can obtain the amount of Fe
#     simply by comp["Fe"] instead of the more verbose comp[Element("Fe")].
#     >>> comp = Composition("LiFePO4")
#     >>> comp.get_atomic_fraction(Element("Li"))
#     0.14285714285714285
#     >>> comp.num_atoms
#     7.0
#     >>> comp.reduced_formula
#     'LiFePO4'
#     >>> comp.formula
#     'Li1 Fe1 P1 O4'
#     >>> comp.get_wt_fraction(Element("Li"))
#     0.04399794666951898
#     >>> comp.num_atoms
#     7.0
#     """
#
#     # Tolerance in distinguishing different composition amounts.
#     # 1e-8 is fairly tight, but should cut out most floating point arithmetic
#     # errors.
#     amount_tolerance = 1e-8
#
#     # Special formula handling for peroxides and certain elements. This is so
#     # that formula output does not write LiO instead of Li2O2 for example.
#     special_formulas = {
#         "LiO": "Li2O2",
#         "NaO": "Na2O2",
#         "KO": "K2O2",
#         "HO": "H2O2",
#         "CsO": "Cs2O2",
#         "RbO": "Rb2O2",
#         "O": "O2",
#         "N": "N2",
#         "F": "F2",
#         "Cl": "Cl2",
#         "H": "H2",
#     }
#
#     oxi_prob = None  # prior probability of oxidation used by oxi_state_guesses
#
#     def __init__(self, *args, strict: bool = False, **kwargs):
#         """
#         Very flexible Composition construction, similar to the built-in Python
#         dict(). Also extended to allow simple string init.
#         Args:
#             Any form supported by the Python built-in {} function.
#             1. A dict of either {Element/Species: amount},
#                {string symbol:amount}, or {atomic number:amount} or any mixture
#                of these. E.g., {Element("Li"):2 ,Element("O"):1},
#                {"Li":2, "O":1}, {3:2, 8:1} all result in a Li2O composition.
#             2. Keyword arg initialization, similar to a dict, e.g.,
#                Composition(Li = 2, O = 1)
#             In addition, the Composition constructor also allows a single
#             string as an input formula. E.g., Composition("Li2O").
#             strict: Only allow valid Elements and Species in the Composition.
#             allow_negative: Whether to allow negative compositions. This
#                 argument must be popped from the **kwargs due to *args
#                 ambiguity.
#         """
#         self.allow_negative = kwargs.pop("allow_negative", False)
#         # it's much faster to recognize a composition and use the elmap than
#         # to pass the composition to {}
#         skip = False
#         if len(args) == 1 and isinstance(args[0], str):
#             elmap = self._parse_formula(args[0])  # type: ignore
#         else:
#             elmap = dict(*args, **kwargs)  # type: ignore
#             if isinstance(list(elmap.keys())[0], Element):
#                 skip = True
#
#         if not skip:
#             elamt = {}
#             self._natoms = 0
#             for k, v in elmap.items():
#                 if v < -Composition.amount_tolerance and not self.allow_negative:
#                     raise ValueError("Amounts in Composition cannot be negative!")
#                 if abs(v) >= Composition.amount_tolerance:
#                     elamt[Element(k)] = int(v)
#                     self._natoms += abs(int(v))
#         else:
#             elamt = elmap
#             self._natoms = sum(elmap.values())
#
#         self._data = elamt
#         self._elements = [el.parameters.symbol for el in elamt.keys()]
#         if strict and not self.valid:
#             raise ValueError("Composition is not valid, contains: {}".format(", ".join(map(str, self._elements))))
#
#     def __getitem__(self, item: Union[str, ElementModel]):
#         try:
#             sp = Element(item)
#             return self._data.get(sp, 0)
#         except ValueError as ex:
#             raise TypeError(f"Invalid key {item}, {type(item)} for Composition\nValueError exception:\n{ex}")
#
#     def __len__(self):
#         return len(self._data)
#
#     def __iter__(self):
#         return self._data.keys().__iter__()
#
#     def __contains__(self, item):
#         try:
#             sp = Element(item)
#             return sp in self._data
#         except ValueError as ex:
#             raise TypeError(f"Invalid key {item}, {type(item)} for Composition\nValueError exception:\n{ex}")
#
#     def __eq__(self, other):
#         #  elements with amounts < Composition.amount_tolerance don't show up
#         #  in the elmap, so checking len enables us to only check one
#         #  compositions elements
#         if len(self) != len(other):
#             return False
#
#         return all(abs(v - other[el]) <= Composition.amount_tolerance for el, v in self._data.items())
#
#     def __ge__(self, other):
#         """
#         Defines >= for Compositions. Should ONLY be used for defining a sort
#         order (the behavior is probably not what you'd expect)
#         """
#         for el in sorted(set(self._elements + other.elements)):
#             if other[el] - self[el] >= Composition.amount_tolerance:
#                 return False
#             if self[el] - other[el] >= Composition.amount_tolerance:
#                 return True
#         return True
#
#     def __ne__(self, other):
#         return not self.__eq__(other)
#
#     def __add__(self, other):
#         """
#         Adds two compositions. For example, an Fe2O3 composition + an FeO
#         composition gives a Fe3O4 composition.
#         """
#         # new_el_map = {}
#         # # new_el_map.update(self)
#         # for k, v in other.as_dict().items():
#         #     try:
#         #         new_el_map[Element(k)] += v
#         #     except:
#         #         new_el_map[Element(k)] = v
#         # return Composition(new_el_map, allow_negative=self.allow_negative)
#
#         for k,v in other.as_dict().items():
#             if k in self._elements:
#                 self._data[list(self._data.keys())[self._elements.index(k)]] += v
#             else:
#                 self._data[Element(k)] = v
#
#     def __sub__(self, other):
#         """
#         Subtracts two compositions. For example, an Fe2O3 composition - an FeO
#         composition gives an FeO2 composition.
#         Raises:
#             ValueError if the subtracted composition is greater than the
#             original composition in any of its elements, unless allow_negative
#             is True
#         """
#         new_el_map = {}
#         new_el_map.update(self)
#         for k, v in other.items():
#             new_el_map[Element(k)] -= v
#         return Composition(new_el_map, allow_negative=self.allow_negative)
#
#     def __mul__(self, other):
#         """
#         Multiply a Composition by an integer or a float.
#         Fe2O3 * 4 -> Fe8O12
#         """
#         if not isinstance(other, int):
#             return NotImplemented
#         return Composition({el: val * other for el, val in self._data.items()}, allow_negative=self.allow_negative)
#
#     __rmul__ = __mul__
#
#     def __truediv__(self, other):
#         if not isinstance(other, int):
#             return NotImplemented
#         return Composition({el: val / other for el, val in self._data.items()}, allow_negative=self.allow_negative)
#
#     __div__ = __truediv__
#
#     def __hash__(self):
#         """
#         hash based on the chemical system
#         """
#         return hash(frozenset(self._data.keys()))
#
#     @property
#     def average_electroneg(self) -> float:
#         """
#         Average electronegativity of the composition.
#
#         Returns
#         -------
#         float
#         """
#         return sum((el.parameters.en_pauling * abs(amt) for el, amt in self._data.items())) / self.num_atoms
#
#     @property
#     def total_electrons(self) -> float:
#         """
#         :return: Total number of electrons in composition.
#         """
#         return sum((el.parameters.atomic_number * abs(amt) for el, amt in self._data.items()))
#
#
#     @property
#     def covalent_radii(self) -> dict:
#         ret_dict = {}
#
#         for el in self._data.keys():
#             ret_dict[el.parameters.symbol] = [el.parameters.covalent_radius_pyykko]
#
#             try:
#                 ret_dict[el.parameters.symbol].append(float(el.parameters.covalent_radius_pyykko_double))
#             except:
#                 ret_dict[el.parameters.symbol].append(None)
#
#             try:
#                 ret_dict[el.parameters.symbol].append(float(el.parameters.covalent_radius_pyykko_triple))
#             except:
#                 ret_dict[el.parameters.symbol].append(None)
#
#         return ret_dict
#
#
#     @property
#     def is_element(self) -> bool:
#         """
#         True if composition is for an element.
#         """
#         return len(self._elements) == 1
#
#     def copy(self) -> "Composition":
#         """
#         :return: A copy of the composition.
#         """
#         return Composition(self, allow_negative=self.allow_negative)
#
#     @property
#     def formula(self) -> str:
#         """
#         Returns a formula string, with elements sorted by electronegativity,
#         e.g., Li4 Fe4 P4 O16.
#         """
#         sym_amt = self.get_el_amt_dict()
#         syms = sorted(sym_amt.keys(), key=lambda sym: Element(sym).parameters.en_pauling)
#         formula = [s + formula_double_format(sym_amt[s], False) for s in syms]
#         return " ".join(formula)
#
#     @property
#     def alphabetical_formula(self) -> str:
#         """
#         Returns a formula string, with elements sorted by alphabetically
#         e.g., Fe4 Li4 O16 P4.
#         """
#         return " ".join(sorted(self.formula.split(" ")))
#
#     @property
#     def fractional_composition(self) -> "Composition":
#         """
#         Returns the normalized composition which the number of species sum to
#         1.
#         Returns:
#             Normalized composition which the number of species sum to 1.
#         """
#         return self / self._natoms
#
#     @property
#     def reduced_composition(self) -> "Composition":
#         """
#         Returns the reduced composition,i.e. amounts normalized by greatest
#         common denominator. e.g., Composition("FePO4") for
#         Composition("Fe4P4O16").
#         """
#         return self.get_reduced_composition_and_factor()[0]
#
#     def get_reduced_composition_and_factor(self) -> Tuple["Composition", float]:
#         """
#         Calculates a reduced composition and factor.
#         Returns:
#             A normalized composition and a multiplicative factor, i.e.,
#             Li4Fe4P4O16 returns (Composition("LiFePO4"), 4).
#         """
#         factor = self.get_reduced_formula_and_factor()[1]
#         return self / factor, factor
#
#     def get_reduced_formula_and_factor(self) -> Tuple[str, float]:
#         """
#         Calculates a reduced formula and factor.
#
#         Returns:
#             A pretty normalized formula and a multiplicative factor, i.e.,
#             Li4Fe4P4O16 returns (LiFePO4, 4).
#         """
#         all_int = all(abs(x - round(x)) < Composition.amount_tolerance for x in self._data.values())
#         if not all_int:
#             return self.formula.replace(" ", ""), 1
#         d = {k: int(round(v)) for k, v in self.get_el_amt_dict().items()}
#         (formula, factor) = reduce_formula(d)
#
#         if formula in Composition.special_formulas:
#             formula = Composition.special_formulas[formula]
#             factor /= 2
#
#         return formula, factor
#
#     def get_integer_formula_and_factor(self) -> Tuple[str, float]:
#         """
#         Calculates an integer formula and factor.
#
#         Returns:
#             A pretty normalized formula and a multiplicative factor, i.e.,
#             Li0.5O0.25 returns (Li2O, 0.25). O0.25 returns (O2, 0.125)
#         """
#         el_amt = self.get_el_amt_dict()
#         g = gcd_float(list(el_amt.values()))
#
#         d = {k: round(v / g) for k, v in el_amt.items()}
#         (formula, factor) = reduce_formula(d)
#         if formula in Composition.special_formulas:
#             formula = Composition.special_formulas[formula]
#             factor /= 2
#         return formula, factor * g
#
#     @property
#     def reduced_formula(self) -> str:
#         """
#         Returns a pretty normalized formula, i.e., LiFePO4 instead of
#         Li4Fe4P4O16.
#         """
#         return self.get_reduced_formula_and_factor()[0]
#
#     @property
#     def hill_formula(self) -> str:
#         """
#         :return: Hill formula. The Hill system (or Hill notation) is a system
#         of writing empirical chemical formulas, molecular chemical formulas and
#         components of a condensed formula such that the number of carbon atoms
#         in a molecule is indicated first, the number of hydrogen atoms next,
#         and then the number of all other chemical elements subsequently, in
#         alphabetical order of the chemical symbols. When the formula contains
#         no carbon, all the elements, including hydrogen, are listed
#         alphabetically.
#         """
#         elements = sorted(self._elements)
#         if "C" in elements:
#             elements = ["C"] + [el for el in elements if el != "C"]
#
#         formula = ["{}{}".format(el, formula_double_format(self.as_dict()[el]) if self.as_dict()[el] != 1 else "") for el in elements]
#         return " ".join(formula)
#
#     @property
#     def elements(self) -> List[ElementModel]:
#         """
#         Returns view of elements in Composition.
#         """
#         return self._elements
#
#     def __str__(self):
#         return " ".join([f"{k}{formula_double_format(v, ignore_ones=False)}" for k, v in self.as_dict().items()])
#
#     def to_pretty_string(self) -> str:
#         """
#         Returns:
#             str: Same as output __str__() but without spaces.
#         """
#         return re.sub(r"\s+", "", self.__str__())
#
#     @property
#     def num_atoms(self) -> float:
#         """
#         Total number of atoms in Composition. For negative amounts, sum
#         of absolute values
#         """
#         return self._natoms
#
#     @property
#     def weight(self) -> float:
#         """
#         Total molecular weight of Composition
#         """
#         return sum(amount * el.parameters.atomic_weight for el, amount in self._data.items())
#
#     def get_atomic_fraction(self, el: Union[str, ElementModel]) -> float:
#         """
#         Calculate atomic fraction of an Element or Species.
#         Args:
#             el (Element/Species): Element or Species to get fraction for.
#         Returns:
#             Atomic fraction for element el in Composition
#         """
#         return abs(self[el]) / self._natoms
#
#     def get_wt_fraction(self, el: Union[str, ElementModel]) -> float:
#         """
#         Calculate weight fraction of an Element or Species.
#         Args:
#             el (Element/Species): Element or Species to get fraction for.
#         Returns:
#             Weight fraction for element el in Composition
#         """
#         return Element(el).parameters.atomic_weight * abs(self[el]) / self.weight
#
#     # def contains_element_type(self, category: str) -> bool:
#     #     """
#     #     Check if Composition contains any elements matching a given category.
#     #     Args:
#     #         category (str): one of "noble_gas", "transition_metal",
#     #             "post_transition_metal", "rare_earth_metal", "metal", "metalloid",
#     #             "alkali", "alkaline", "halogen", "chalcogen", "lanthanoid",
#     #             "actinoid", "quadrupolar", "s-block", "p-block", "d-block", "f-block"
#     #     Returns:
#     #         True if any elements in Composition match category, otherwise False
#     #     """
#     #
#     #     allowed_categories = (
#     #         "noble_gas",
#     #         "transition_metal",
#     #         "post_transition_metal",
#     #         "rare_earth_metal",
#     #         "metal",
#     #         "metalloid",
#     #         "alkali",
#     #         "alkaline",
#     #         "halogen",
#     #         "chalcogen",
#     #         "lanthanoid",
#     #         "actinoid",
#     #         "quadrupolar",
#     #         "s-block",
#     #         "p-block",
#     #         "d-block",
#     #         "f-block",
#     #     )
#     #
#     #     if category not in allowed_categories:
#     #         raise ValueError("Please pick a category from: {}".format(", ".join(allowed_categories)))
#     #
#     #     if "block" in category:
#     #         return any(category[0] in el.block for el in self.elements)
#     #     return any(getattr(el, f"is_{category}") for el in self.elements)
#
#     def _parse_formula(self, formula: str) -> Dict[str, float]:
#         """
#         Args:
#             formula (str): A string formula, e.g. Fe2O3, Li3Fe2(PO4)3
#         Returns:
#             Composition with that formula.
#         Notes:
#             In the case of Metallofullerene formula (e.g. Y3N@C80),
#             the @ mark will be dropped and passed to parser.
#         """
#         # for Metallofullerene like "Y3N@C80"
#         formula = formula.replace("@", "")
#
#         def get_sym_dict(form: str, factor: Union[int, float]) -> Dict[str, float]:
#             sym_dict: Dict[str, float] = collections.defaultdict(float)
#             for m in re.finditer(r"([A-Z][a-z]*)\s*([-*\.e\d]*)", form):
#                 el = m.group(1)
#                 amt = 1.0
#                 if m.group(2).strip() != "":
#                     amt = float(m.group(2))
#                 sym_dict[el] += amt * factor
#                 form = form.replace(m.group(), "", 1)
#             if form.strip():
#                 raise ValueError(f"{form} is an invalid formula!")
#             return sym_dict
#
#         m = re.search(r"\(([^\(\)]+)\)\s*([\.e\d]*)", formula)
#         if m:
#             factor = 1.0
#             if m.group(2) != "":
#                 factor = float(m.group(2))
#             unit_sym_dict = get_sym_dict(m.group(1), factor)
#             expanded_sym = "".join([f"{el}{amt}" for el, amt in unit_sym_dict.items()])
#             expanded_formula = formula.replace(m.group(), expanded_sym)
#             return self._parse_formula(expanded_formula)
#         return get_sym_dict(formula, 1)
#
#     # @property
#     # def anonymized_formula(self) -> str:
#     #     """
#     #     An anonymized formula. Unique species are arranged in ordering of
#     #     increasing amounts and assigned ascending alphabets. Useful for
#     #     prototyping formulas. For example, all stoichiometric perovskites have
#     #     anonymized_formula ABC3.
#     #     """
#     #     reduced = self.element_composition
#     #     if all(x == int(x) for x in self.values()):
#     #         reduced /= gcd(*(int(i) for i in self.values()))
#     #
#     #     anon = ""
#     #     for e, amt in zip(string.ascii_uppercase, sorted(reduced.values())):
#     #         if amt == 1:
#     #             amt_str = ""
#     #         elif abs(amt % 1) < 1e-8:
#     #             amt_str = str(int(amt))
#     #         else:
#     #             amt_str = str(amt)
#     #         anon += f"{e}{amt_str}"
#     #     return anon
#
#     @property
#     def chemical_system(self) -> str:
#         """
#         Get the chemical system of a Composition, for example "O-Si" for
#         SiO2. Chemical system is a string of a list of elements
#         sorted alphabetically and joined by dashes, by convention for use
#         in database keys.
#         """
#         return "-".join(sorted(el for el in self._elements))
#
#     # @property
#     # def valid(self) -> bool:
#     #     """
#     #     Returns True if Composition contains valid elements or species and
#     #     False if the Composition contains any dummy species.
#     #     """
#     #     return not any(isinstance(el, DummySpecies) for el in self.elements)
#
#     def __repr__(self) -> str:
#         return "Comp: " + self.formula
#
#     @classmethod
#     def from_dict(cls, d) -> "Composition":
#         """
#         Creates a composition from a dict generated by as_dict(). Strictly not
#         necessary given that the standard constructor already takes in such an
#         input, but this method preserves the standard pymatgen API of having
#         from_dict methods to reconstitute objects generated by as_dict(). Allows
#         for easier introspection.
#         Args:
#             d (dict): {symbol: amount} dict.
#         """
#         return cls(d)
#
#     def get_el_amt_dict(self) -> Dict[str, float]:
#         """
#         Returns:
#             Dict with element symbol and (unreduced) amount e.g.,
#             {"Fe": 4.0, "O":6.0} or {"Fe3+": 4.0, "O2-":6.0}
#         """
#         return {e.parameters.symbol: a for e,a in self._data.items()}
#
#     def as_dict(self) -> Dict[str, float]:
#         """
#         Returns:
#             dict with species symbol and (unreduced) amount e.g.,
#             {"Fe": 4.0, "O":6.0} or {"Fe3+": 4.0, "O2-":6.0}
#         """
#         return {e.parameters.symbol: a for e,a in self._data.items()}
#
#     @property
#     def to_reduced_dict(self) -> dict:
#         """
#         Returns:
#             Dict with element symbol and reduced amount e.g.,
#             {"Fe": 2.0, "O":3.0}
#         """
#         return self.get_reduced_composition_and_factor()[0].as_dict()
#
#     @property
#     def to_data_dict(self) -> dict:
#         """
#         Returns:
#             A dict with many keys and values relating to Composition/Formula,
#             including reduced_cell_composition, unit_cell_composition,
#             reduced_cell_formula, elements and nelements.
#         """
#         return {
#             "reduced_cell_composition": self.get_reduced_composition_and_factor()[0],
#             "unit_cell_composition": self.as_dict(),
#             "reduced_cell_formula": self.reduced_formula,
#             "elements": list(self.as_dict().keys()),
#             "nelements": len(self.as_dict().keys()),
#         }
#
#
#     def replace(self, elem_map: Dict[str, Union[str, Dict[str, Union[int, float]]]]) -> "Composition":
#         """
#         Replace elements in a composition. Returns a new Composition, leaving the old one unchanged.
#         Args:
#             elem_map (dict[str, str | dict[str, int | float]]): dict of elements or species to swap. E.g.
#                 {"Li": "Na"} performs a Li for Na substitution. The target can be a {species: factor} dict. For
#                 example, in Fe2O3 you could map {"Fe": {"Mg": 0.5, "Cu":0.5}} to obtain MgCuO3.
#         Returns:
#             Composition: New object with elements remapped according to elem_map.
#         """
#
#         # drop inapplicable substitutions
#         invalid_elems = [key for key in elem_map if key not in self]
#         if invalid_elems:
#             warnings.warn(
#                 "Some elements to be substituted are not present in composition. Please check your input. "
#                 f"Problematic element = {invalid_elems}; {self}"
#             )
#         for elem in invalid_elems:
#             elem_map.pop(elem)
#
#         new_comp = self.as_dict()
#
#         for old_elem, new_elem in elem_map.items():
#             amount = new_comp.pop(old_elem)
#
#             if isinstance(new_elem, dict):
#                 for el, factor in new_elem.items():
#                     new_comp[el] = factor * amount
#             else:
#                 new_comp[new_elem] = amount
#
#         return Composition(new_comp)
#
#
#     @staticmethod
#     def ranked_compositions_from_indeterminate_formula(
#         fuzzy_formula: str, lock_if_strict: bool = True
#     ) -> List["Composition"]:
#         """
#         Takes in a formula where capitalization might not be correctly entered,
#         and suggests a ranked list of potential Composition matches.
#         Author: Anubhav Jain
#         Args:
#             fuzzy_formula (str): A formula string, such as "co2o3" or "MN",
#                 that may or may not have multiple interpretations
#             lock_if_strict (bool): If true, a properly entered formula will
#                 only return the one correct interpretation. For example,
#                 "Co1" will only return "Co1" if true, but will return both
#                 "Co1" and "C1 O1" if false.
#         Returns:
#             A ranked list of potential Composition matches
#         """
#
#         # if we have an exact match and the user specifies lock_if_strict, just
#         # return the exact match!
#         if lock_if_strict:
#             # the strict composition parsing might throw an error, we can ignore
#             # it and just get on with fuzzy matching
#             try:
#                 comp = Composition(fuzzy_formula)
#                 return [comp]
#             except ValueError:
#                 pass
#
#         all_matches = Composition._comps_from_fuzzy_formula(fuzzy_formula)
#         # remove duplicates
#         uniq_matches = list(set(all_matches))
#         # sort matches by rank descending
#         ranked_matches = sorted(uniq_matches, key=lambda match: (match[1], match[0]), reverse=True)
#
#         return [m[0] for m in ranked_matches]
#
#     @staticmethod
#     def _comps_from_fuzzy_formula(
#         fuzzy_formula: str,
#         m_dict: Dict[str, float] = None,
#         m_points: int = 0,
#         factor: Union[int, float] = 1,
#     ) -> Generator[Tuple["Composition", int], None, None]:
#         """
#         A recursive helper method for formula parsing that helps in
#         interpreting and ranking indeterminate formulas.
#         Author: Anubhav Jain
#         Args:
#             fuzzy_formula (str): A formula string, such as "co2o3" or "MN",
#                 that may or may not have multiple interpretations.
#             m_dict (dict): A symbol:amt dictionary from the previously parsed
#                 formula.
#             m_points: Number of points gained from the previously parsed
#                 formula.
#             factor: Coefficient for this parse, e.g. (PO4)2 will feed in PO4
#                 as the fuzzy_formula with a coefficient of 2.
#         Returns:
#             list[tuple[Composition, int]]: A list of tuples, with the first element being a Composition
#                 and the second element being the number of points awarded that Composition interpretation.
#         """
#         m_dict = m_dict or {}
#
#         def _parse_chomp_and_rank(m, f, m_dict, m_points):
#             """
#             A helper method for formula parsing that helps in interpreting and
#             ranking indeterminate formulas
#             Author: Anubhav Jain
#             Args:
#                 m: A regex match, with the first group being the element and
#                     the second group being the amount
#                 f: The formula part containing the match
#                 m_dict: A symbol:amt dictionary from the previously parsed
#                     formula
#                 m_points: Number of points gained from the previously parsed
#                     formula
#             Returns:
#                 A tuple of (f, m_dict, points) where m_dict now contains data
#                 from the match and the match has been removed (chomped) from
#                 the formula f. The "goodness" of the match determines the
#                 number of points returned for chomping. Returns
#                 (None, None, None) if no element could be found...
#             """
#
#             points = 0
#             # Points awarded if the first element of the element is correctly
#             # specified as a capital
#             points_first_capital = 100
#             # Points awarded if the second letter of the element is correctly
#             # specified as lowercase
#             points_second_lowercase = 100
#
#             # get element and amount from regex match
#             el = m.group(1)
#             if len(el) > 2 or len(el) < 1:
#                 raise ValueError("Invalid element symbol entered!")
#             amt = float(m.group(2)) if m.group(2).strip() != "" else 1
#
#             # convert the element string to proper [uppercase,lowercase] format
#             # and award points if it is already in that format
#             char1 = el[0]
#             char2 = el[1] if len(el) > 1 else ""
#
#             if char1 == char1.upper():
#                 points += points_first_capital
#             if char2 and char2 == char2.lower():
#                 points += points_second_lowercase
#
#             el = char1.upper() + char2.lower()
#
#             # if it's a valid element, chomp and add to the points
#             if Element.is_valid_symbol(el):
#                 if el in m_dict:
#                     m_dict[el] += amt * factor
#                 else:
#                     m_dict[el] = amt * factor
#                 return f.replace(m.group(), "", 1), m_dict, m_points + points
#
#             # else return None
#             return None, None, None
#
#         fuzzy_formula = fuzzy_formula.strip()
#
#         if len(fuzzy_formula) == 0:
#             # The entire formula has been parsed into m_dict. Return the
#             # corresponding Composition and number of points
#             if m_dict:
#                 yield (Composition.from_dict(m_dict), m_points)
#         else:
#             # if there is a parenthesis, remove it and match the remaining stuff
#             # with the appropriate factor
#             for mp in re.finditer(r"\(([^\(\)]+)\)([\.\d]*)", fuzzy_formula):
#                 mp_points = m_points
#                 mp_form = fuzzy_formula.replace(mp.group(), " ", 1)
#                 mp_dict = dict(m_dict)
#                 mp_factor = 1 if mp.group(2) == "" else float(mp.group(2))
#                 # Match the stuff inside the parenthesis with the appropriate
#                 # factor
#                 for match in Composition._comps_from_fuzzy_formula(mp.group(1), mp_dict, mp_points, factor=mp_factor):
#                     only_me = True
#                     # Match the stuff outside the parentheses and return the
#                     # sum.
#
#                     for match2 in Composition._comps_from_fuzzy_formula(mp_form, mp_dict, mp_points, factor=1):
#                         only_me = False
#                         yield (match[0] + match2[0], match[1] + match2[1])
#                     # if the stuff inside the parenthesis is nothing, then just
#                     # return the stuff inside the parentheses
#                     if only_me:
#                         yield match
#                 return
#
#             # try to match the single-letter elements
#             m1 = re.match(r"([A-z])([\.\d]*)", fuzzy_formula)
#             if m1:
#                 m_points1 = m_points
#                 m_form1 = fuzzy_formula
#                 m_dict1 = dict(m_dict)
#                 (m_form1, m_dict1, m_points1) = _parse_chomp_and_rank(m1, m_form1, m_dict1, m_points1)
#                 if m_dict1:
#                     # there was a real match
#                     for match in Composition._comps_from_fuzzy_formula(m_form1, m_dict1, m_points1, factor):
#                         yield match
#
#             # try to match two-letter elements
#             m2 = re.match(r"([A-z]{2})([\.\d]*)", fuzzy_formula)
#             if m2:
#                 m_points2 = m_points
#                 m_form2 = fuzzy_formula
#                 m_dict2 = dict(m_dict)
#                 (m_form2, m_dict2, m_points2) = _parse_chomp_and_rank(m2, m_form2, m_dict2, m_points2)
#                 if m_dict2:
#                     # there was a real match
#                     for match in Composition._comps_from_fuzzy_formula(m_form2, m_dict2, m_points2, factor):
#                         yield match
#
#
# def reduce_formula(sym_amt) -> Tuple[str, float]:
#     """
#     Helper method to reduce a sym_amt dict to a reduced formula and factor.
#     Args:
#         sym_amt (dict): {symbol: amount}.
#
#     Returns:
#         (reduced_formula, factor).
#     """
#     syms = sorted(sym_amt.keys(), key=lambda x: [Element(x).parameters.en_pauling, x])
#
#     syms = list(filter(lambda x: abs(sym_amt[x]) > Composition.amount_tolerance, syms))
#
#     factor = 1
#     # Enforce integers for doing gcd.
#     if all(int(i) == i for i in sym_amt.values()):
#         factor = abs(gcd_float(int(i) for i in sym_amt.values()))
#
#     polyanion = []
#     # if the composition contains a poly anion
#     if len(syms) >= 3 and Element(syms[-1]).parameters.en_pauling - Element(syms[-2]).parameters.en_pauling < 1.65:
#         poly_sym_amt = {syms[i]: sym_amt[syms[i]] / factor for i in [-2, -1]}
#         (poly_form, poly_factor) = reduce_formula(poly_sym_amt)
#
#         if poly_factor != 1:
#             polyanion.append(f"({poly_form}){int(poly_factor)}")
#
#     syms = syms[: len(syms) - 2 if polyanion else len(syms)]
#
#     reduced_form = []
#     for s in syms:
#         normamt = sym_amt[s] * 1.0 / factor
#         reduced_form.append(s)
#         reduced_form.append(formula_double_format(normamt))
#
#     reduced_form = "".join(reduced_form + polyanion)  # type: ignore
#     return reduced_form, factor  # type: ignore