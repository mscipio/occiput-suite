# ilang - Inference Language
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 


# Exceptions
class UnexpectedParameterType(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr("Unexpected parameter type: " + str(self.value))


class ParameterError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr("Parameter error: " + str(self.value))


class InconsistentGraph(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr("Parameter error: " + str(self.value))


class NotInitialized(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr("Not initialized: " + str(self.value))


class NoCompatibleSampler(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr("Not compatible sampler: " + str(self.value))


class ModelUndefined(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr("Model method undefined: " + self.msg)
