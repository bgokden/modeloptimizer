import random
import sys
import copy


class Hyperparameter:

    def __init__(self, name, kind = 'default', value = None, lower_limit = 0.0, upper_limit = 1.0, choices = [], randomize_step = 0.2, fixed = False):
        self.kind = kind
        self.name = name
        self.choices = choices
        if len(choices) == 0:
            self.lower_limit = lower_limit
            self.upper_limit = upper_limit
        else:
            self.lower_limit = 0
            self.upper_limit = len(choices)-1
        self.value = self.__get_or_else(value, random.uniform(self.lower_limit, self.upper_limit) )
        self.apply_limits()
        self.randomize_step = randomize_step
        self.average = (self.lower_limit + self.upper_limit) / 2.0
        self.fixed = fixed

    def __get_or_else(self, value, default):
        if value is None:
            return default
        if len(self.choices) > 0:
            if type(value) == type(self.choices[0]):
                return self.choices.index(value)
        return value

    def get_kind(self):
        return self.kind

    def get_name(self):
        return self.name

    def get(self):
        if len(self.choices) == 0:
            return self.value
        return self.choices[round(self.value)]

    def apply_limits(self):
        if self.value < self.lower_limit:
            self.value = self.lower_limit
        elif self.value > self.upper_limit:
            self.value = self.upper_limit

    def randomize(self):
        if not self.fixed:
            if len(self.choices) == 0:
                self.value = self.value + self.step()
            else:
                # choices are not linear
                self.value = random.uniform(self.lower_limit, self.upper_limit)
            self.apply_limits()

    def step(self):
        return self.randomize_step * (random.uniform(self.lower_limit, self.upper_limit) - self.average)

    def __str__(self):
        return "("+str(self.name)+","+str(self.kind)+","+str(self.get())+","+str(self.lower_limit)+","+str(self.upper_limit)+","+str(self.choices)+")"

class RealHyperparameter(Hyperparameter):

    def __init__(self, name, value = 0.5, lower_limit = 0.0, upper_limit = 1.0, randomize_step = 0.2, fixed = False):
        Hyperparameter.__init__(self, name, 'real', value, lower_limit, upper_limit, [], randomize_step, fixed)

class IntegerHyperparameter(Hyperparameter):

    def __init__(self, name, value = 5, lower_limit = 0, upper_limit = 10, randomize_step = 0.2, fixed = False):
        Hyperparameter.__init__(self, name, 'integer', value, lower_limit, upper_limit, [], randomize_step, fixed)

    def get(self):
        return int(round(super().get()))


class ChoiceHyperparameter(Hyperparameter):

    def __init__(self, name, value = 0, choices = ['default'], fixed = False):
        Hyperparameter.__init__(self, name, 'choice', value, 0, 0, choices, 0.2, fixed)


class HyperparameterSet:

    def __init__(self, hps = {}):
        self.hps = hps
        self.per_scope_counter = { '' : 0 }

    def get_and_increment_per_scope_counter(self, scope=''):
        if scope in self.per_scope_counter:
            counter = self.per_scope_counter[scope]
            current = counter
            counter += 1
            self.per_scope_counter[scope] = counter
            return current
        else:
            self.per_scope_counter[scope] = 1
            return 0

    def get_next_name(self, scope = '', type='default'):
        name = ''
        if len(str(scope)) > 0:
            name = str(scope)+"_"
        return name+str(type)+"_"+str(self.get_and_increment_per_scope_counter(scope))

    def get_set(self):
        return self.hps

    def get(self, name = None, scope = '', type = None, hint = None, choices = [], minimum = 0, maximum = 1):
        if type is None:
            type = 'default'
        if name is None:
            name = self.get_next_name(scope, type)
        if name in self.hps:
            return self.hps[name].get()
        else:
            switcher = {
                "default": lambda: Hyperparameter(name, hint, minimum, maximum),
                "real": lambda: RealHyperparameter(name, hint, minimum, maximum),
                "integer": lambda: IntegerHyperparameter(name, hint, minimum, maximum),
                "choice": lambda: ChoiceHyperparameter(name, hint, choices),
                "iterator": lambda: IntegerHyperparameter(name, hint, 0, maximum ),
                "dimension": lambda: IntegerHyperparameter(name, hint, 1, maximum ),
                "dropout": lambda: RealHyperparameter(name, hint, 0, maximum ),
                "loss_min_delta": lambda: RealHyperparameter(name, hint, 0, 0.1 ),
                "batch_size": lambda: IntegerHyperparameter(name, hint, 1, maximum ),
                "keras_optimizer": lambda: ChoiceHyperparameter(name, hint, choices = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam',  'adamax', 'nadam']),
                "keras_activation": lambda: ChoiceHyperparameter(name, hint, choices = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear']),
            }
            hp = switcher.get(type, lambda: Hyperparameter(name, hint, minimum, maximum))
            self.hps[name] = hp()
            return self.hps[name].get()

    def put(self, name, hyperparameter):
        self.hps[name] = hyperparameter

    def randomized(self, with_delete = True):
        hps = copy.deepcopy(self.hps)
        delete_list = []
        for key in hps:
            if with_delete and random.random() < 0.2:
                delete_list.append(key)
            else:
                hps[key].randomize()
        for key in delete_list:
            hps.pop(key, None)
        return HyperparameterSet(hps)

    def cross_merge(self, other):
        hpsA = copy.deepcopy(self.hps)
        hpsB = copy.deepcopy(other.get_set())
        for key in hpsB:
            if key not in hpsA:
                hpsA[key] = hpsB[key]
        for key in hpsA:
            if key in hpsB and random.random() < 0.5:
                temp = hpsA[key]
                hpsA[key] = hpsB[key]
                hpsB[key] = temp
            else:
                hpsB[key] = hpsA[key]
        return HyperparameterSet(hpsA), HyperparameterSet(hpsB)

    def print_all(self):
        for key in self.hps:
            print(self.hps[key].get_name(), self.hps[key].get())
