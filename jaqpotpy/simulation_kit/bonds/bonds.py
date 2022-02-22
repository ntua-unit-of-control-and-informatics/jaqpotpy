# import collections
# import json
# import os
# import warnings
# import inspect
# import logging
# import numpy as np
# from typing import Optional, List, Union
#
# from jaqpotpy.simulation_kit import Element
# from jaqpotpy.simulation_kit.sites.sites import Site, PeriodicSite
#
# _print_threshold = 10
# logger = logging.getLogger(__name__)
#
#
# class Bond(object):
#     def __init__(self, sites: List[Union[Site, PeriodicSite]]):
#         if len(sites) != 2:
#             raise ValueError('Two sites should be passed in order to create a both. No more, no less!')
#         self.__site1 = sites[0]
#         self.__site2 = sites[1]
#
#     @property
#     def site1(self):
#         return self.__site1
#
#     @property
#     def site2(self):
#         return self.__site2
#
#     @property
#     def sites(self):
#         return [self.__site1, self.__site2]
#
#     @property
#     def bond_type(self) -> str:
#         """
#         Get the type of the bond
#         """
#         comp1 = self.__site1.species
#         comp2 = self.__site2.species
#         dif = abs(comp1.average_electroneg - comp2.average_electroneg)
#
#         if dif < 0.4:
#             return "NonPolarCovalent"
#         elif dif < 1.7:
#             return "PolarCovalent"
#         else:
#             return "Ionic"
#
#     def bond_length(self, order:int=None) -> dict:
#         """
#         Get the type of the bond
#         """
#         comp1 = self.__site1.species
#         comp2 = self.__site2.species
#         correction_const = 9.0 * abs(comp1.average_electroneg - comp2.average_electroneg)
#
#         comp1_radii = comp1.covalent_radii
#         comp2_radii = comp2.covalent_radii
#
#         # ret_dict = {'{} - {}'.format(el1, el2) : None for el1 in comp1_radii.keys() for el2 in comp2_radii.keys()}
#         ret_dict = {}
#
#         if order:
#             if order not in [1, 2, 3]:
#                 raise ValueError('Bond with order {} does note exist. Order parameter should be order<=3, order>=1.')
#             for el1 in comp1_radii.keys():
#                 for el2 in comp2_radii.keys():
#                     if comp1_radii[el1][order] and comp2_radii[el2][order]:
#                         ret_dict['{} - {}'.format(el1, el2)] = comp1_radii[el1][order] + comp2_radii[el2][order] - correction_const
#                     else:
#                         ret_dict['{} - {}'.format(el1, el2)] = None
#         else:
#             for el1 in comp1_radii.keys():
#                 for el2 in comp2_radii.keys():
#                     index = 4
#                     try:
#                         index = comp1_radii[el1].index(None)
#                     except:
#                         pass
#
#                     try:
#                         index = comp2_radii[el2].index(None)
#                     except:
#                         pass
#                     ret_dict['{} - {}'.format(el1, el2)] = [comp1_radii[el1][i] + comp2_radii[el2][i] - correction_const if i < index else None for i in range(3)]
#         return ret_dict
#
#     def __call__(self, sites: List[Union[Site, PeriodicSite]]):
#         """
#         Create Bonds.
#         """
#         return self.__init__(self, sites)
#
#     def __repr__(self) -> str:
#         """
#         Convert self to repr representation.
#         Returns
#         -------
#         str
#           The string represents the class.
#         """
#         args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
#         args_names = [arg for arg in args_spec.args if arg != 'self']
#         args_info = ''
#         for arg_name in args_names:
#             value = self.__dict__[arg_name]
#             # for str
#             if isinstance(value, str):
#                 value = "'" + value + "'"
#             # for list
#             if isinstance(value, list):
#                 threshold = get_print_threshold()
#                 value = np.array2string(np.array(value), threshold=threshold)
#             args_info += arg_name + '=' + str(value) + ', '
#         return self.__class__.__name__ + '[' + args_info[:-2] + ']'
#
#     def __str__(self) -> str:
#         """
#         Convert self to str representation.
#         Returns
#         -------
#         str
#           The string represents the class.
#         """
#         args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
#         args_names = [arg for arg in args_spec.args if arg != 'self']
#         args_num = len(args_names)
#         args_default_values = [None for _ in range(args_num)]
#         if args_spec.defaults is not None:
#             defaults = list(args_spec.defaults)
#             args_default_values[-len(defaults):] = defaults
#
#         override_args_info = ''
#         for arg_name, default in zip(args_names, args_default_values):
#             if arg_name in self.__dict__:
#                 arg_value = self.__dict__[arg_name]
#                 # validation
#                 # skip list
#                 if isinstance(arg_value, list):
#                     continue
#                 if isinstance(arg_value, str):
#                     # skip path string
#                     if "\\/." in arg_value or "/" in arg_value or '.' in arg_value:
#                         continue
#                 # main logic
#                 if default != arg_value:
#                     override_args_info += '_' + arg_name + '_' + str(arg_value)
#         return self.__class__.__name__ + override_args_info
#
