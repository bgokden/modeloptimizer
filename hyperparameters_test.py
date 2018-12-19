from hyperparameters import *

x0 = Hyperparameter("default_hyperparameter", 0.6, -1.0, 1.0)
x1 = RealHyperparameter("real_hyperparameter", 11, -1.0, 1.0)
x2 = IntegerHyperparameter("integer_hyperparameter", 0, -5, 60)
x3 = ChoiceHyperparameter("choice_hyperparameter", 0, ['type1','type2', 'type3', 'type4'])

print(x0.get())
print(x1.get())
print(x2.get())
print(x3.get())

x1.randomize()
print(x1.get_name(), x1.get())

x2.randomize()
print(x2.get_name(), x2.get())

x3.randomize()
print(x3.get_name(), x3.get())
print(x3)

x0 = Hyperparameter("default_hyperparameter", 0.0, -1.0, 1.0)
x0.randomize()
print(x0.get_name(), x0.get())

print(x0)

hps = HyperparameterSet()

x4 = hps.get(name = "first", type= "integerr", hint = 5)
print(x4)

x5 = hps.get(name = "first", type = "integer", hint = 5)
print(x5)

x6 = hps.get(name = "second", type="iterator")
print(x6)

x7 = hps.get(name = "optimizer", type="keras_optimizer")
print("x7:",x7)

x8 = hps.get(name = "dimention1", type="dimention", maximum = 100)
print(x8)

x9 = hps.get()
print(x9)

x10 = hps.get()
print(x10)

x11 = hps.get()
print(x11)

x12 = hps.get()
print(x12)

print(hps)

print("optimizer1", hps.get(name="optimizer1", type="keras_optimizer") )

hps_randomized = hps.randomized()

hps_randomized.get(name = "loss_field", type="real")

a, b = hps.cross_merge(hps_randomized)

print("a:")
a.print_all()
print("b:")
b.print_all()

x13 = hps.get(scope='inner_scope')
hps.print_all()

print("optimizer1", hps.get(name="optimizer1", type="keras_optimizer") )

import jsonpickle
encoded = jsonpickle.encode(hps)

hps2 = jsonpickle.decode(encoded)
hps2.print_all()

print("optimizer1", hps.get(name="optimizer1", type="keras_optimizer") )


c = ChoiceHyperparameter("choice_hyperparameter", 'type3', ['type1','type2', 'type3', 'type4'])

print("c", hps.get(name="c", type="choice", hint='type3', choices = ['type1','type2', 'type3', 'type4']) )
