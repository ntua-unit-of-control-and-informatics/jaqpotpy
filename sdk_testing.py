from jaqpotpy.jaqpot import Jaqpot

jaqpot = Jaqpot()
jaqpot.login()
model = jaqpot.get_model_by_id(model_id="1834")
