from tdc.utils import retrieve_label_name_list
from tdc.single_pred import QM

label_list = retrieve_label_name_list('QM7b')
data = QM(name = 'QM7b', label_name = label_list[4])
split = data.get_split()

print(split['train'])

print(split['train']['Drug'][1])
