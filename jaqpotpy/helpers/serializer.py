import json
import jaqpotpy.helpers.helpers as hel
import numpy as np


class JaqpotSerializer(json.JSONEncoder):

    # def __init__(self, *args, **kwargs):
    #     json.JSONEncoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def default(self, o):
        delete = []
        if isinstance(o, np.ndarray):
            if o.flags['C_CONTIGUOUS']:
                obj_data = o.data
            else:
                cont_obj = np.ascontiguousarray(o)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            return o.tolist()
        if isinstance(o, np.int64):
            return int(o)
        if type(o).__name__ == 'bool_':
            return bool(o)
        if type(o).__name__ == 'mappingproxy':
            return {'mappingproxy': " "}
        if type(o).__name__ == 'set':
            return {'set': " "}
        try:
            for key in o.__dict__:
                if getattr(o, key) is None or "":
                    delete.append(key)
        except AttributeError:
            return {type(o).__name__: " "}

        for keytod in delete:
            delattr(o, keytod)
        # except AttributeError:
        #     delete.append(key)

        return o.__dict__



# class JaqpotDeserializer(json.JSONDecoder):
#
#     def __init__(self, *args, **kwargs):
#         json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
#
#     def object_hook(self, obj):
#         print("From des " + str(obj))
#         # id = obj["_id"]
#         # print(id)
#         # print(getattr(obj, "_id"))
#         # setattr()
#         return obj




# fe = hel.create_feature("sadf", "adf")

# fe1 = hel.clear_entity(fe.__dict__)
# print(fe1)

# print(fe.__dict__)
# print(json.dumps(fe.__dict__))
# print(json.dumps(fe, cls=JaqpotSerializer))
# print(json.dumps(fe.__dict__, cls=JaqpotSerializer))
''