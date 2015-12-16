# https://github.com/fchollet/keras/pull/928/files

import theano.tensor as T
from keras.layers.core import Layer


def add_shared_node(self, layer, name, inputs=[], merge_mode=None, concat_axis=-1, dot_axes=-1, outputs=[], create_output=False):

    if name in self.namespace:
        raise Exception('Duplicate node identifier: ' + name)
    for o in outputs:
        if o in self.namespace:
            raise Exception('Duplicate node identifier: ' + o)
    if merge_mode:
        if merge_mode not in {'sum', 'ave', 'mul', 'dot', 'cos', 'concat', 'join'}:
            raise Exception("Invalid merge mode")
    layers = []
    for i in range(len(inputs)):
        input = inputs[i]
        if input in self.nodes:
            n = self.nodes[input]
            if hasattr(n, 'get_output_at'):#is it a siamese layer?
                if n.merge_mode is None:
                    for j in range(len(n.inputs)):
                        sh = SiameseHead(j)
                        sh.previous = n
                        layers.append(sh)
                else:
                    layers.append(n)
        elif input in self.inputs:
            n = self.inputs[input]
            layers.append(n)
        else:
            raise Exception('Unknown identifier: ' + n)
    s = Siamese(layer, layers, merge_mode, concat_axis=concat_axis, dot_axes=dot_axes)
    s.set_name(name)
    self.namespace.add(name)
    self.nodes[name] = s
    self.node_config.append({'name': name,
                            'inputs': inputs,
                            'merge_mode': merge_mode,
                            'concat_axis': concat_axis,
                            'dot_axes': dot_axes,
                            'create_output': create_output if merge_mode else False})
    if not merge_mode:
        for i in range(len(outputs)):
            sh = SiameseHead(i)
            sh.previous = s
            sh_name = outputs[i]
            sh.set_name(sh_name)
            self.namespace.add(sh_name)
            self.nodes[sh_name] = sh
            self.node_config.append({'name': sh_name,
                                    'inputs': [s],
                                    'create_output': create_output})
            if create_output:
                self.add_output(sh_name, input=sh_name)

    if create_output and merge_mode:
        if merge_mode == 'join':
            raise Exception("Output can not be of type OrderedDict")
        self.add_output(name, input=name)


class Siamese(Layer):
    def __init__(self, layer, inputs, merge_mode=None, concat_axis=1, dot_axes=-1):

        if merge_mode not in ['sum', 'mul', 'concat', 'ave', 'join', 'cos', 'dot', None]:
            raise Exception("Invalid merge mode: " + str(mode))

        if merge_mode in {'cos', 'dot'}:
            if len(inputs) > 2:
                raise Exception(mode + " merge takes exactly 2 layers")
            shape1 = inputs[0].output_shape
            shape2 = inputs[1].output_shape
            n1 = len(shape1)
            n2 = len(shape2)
            if mode == 'dot':
                if type(dot_axes) == int:
                    if dot_axes < 0:
                        dot_axes = [range(dot_axes % n1, n1), range(dot_axes % n2, n2)]
                    else:
                        dot_axes = [range(n1 - dot_axes, n2), range(1, dot_axes + 1)]
                for i in range(len(dot_axes[0])):
                    if shape1[dot_axes[0][i]] != shape2[dot_axes[1][i]]:
                        raise Exception(" Dot incompatible layers can not be merged using dot mode")

        self.layer = layer
        self.inputs = inputs
        self.params = []
        self.merge_mode = merge_mode
        self.concat_axis = concat_axis
        self.dot_axes = dot_axes
        layer.set_previous(inputs[0])
        self.regularizers = []
        self.constraints = []
        self.updates = []
        layers = [layer] + inputs
        for l in layers:
            params, regs, consts, updates = l.get_params()
            self.regularizers += regs
            self.updates += updates
            # params and constraints have the same size
            for p, c in zip(params, consts):
                if p not in self.params:
                    self.params.append(p)
                    self.constraints.append(c)

    @property
    def output_shape(self):
        if merge_mode is None:
            return self.layer.output_shape
        input_shapes = [self.layer.output_shape]*len(self.inputs)
        if self.merge_mode in ['sum', 'mul', 'ave']:
            return input_shapes[0]
        elif self.merge_mode == 'concat':
            output_shape = list(input_shapes[0])
            for shape in input_shapes[1:]:
                output_shape[self.concat_axis] += shape[self.concat_axis]
            return tuple(output_shape)
        elif self.merge_mode == 'join':
            return None
        elif self.merge_mode == 'dot':
            shape1 = list(input_shapes[0])
            shape2 = list(input_shapes[1])
            for i in self.dot_axes[0]:
                shape1.pop(i)
            for i in self.dot_axes[1]:
                shape2.pop(i)
            shape = shape1 + shape2[1:]
            if len(shape) == 1:
                shape.append(1)
            return tuple(shape)
        elif self.merge_mode == 'cos':
            return tuple(input_shapes[0][0], 1)

    def get_params(self):
        return self.params, self.regularizers, self.constraints, self.updates

    def get_output_at(self, head, train=False):
        self.layer.previous = self.inputs[head]
        return self.layer.get_output(train)

    def get_output_join(self, train=False):
        o = OrderedDict()
        for i in range(len(inputs)):
            X = self.get_output_at(i, train)
            if X.name is None:
                raise ValueError("merge_mode='join' only works with named inputs")
            o[X.name] = X
        return o

    def get_output_sum(self, train=False):
        s = self.get_output_at(0, train)
        for i in range(1, len(self.inputs)):
            s += self.get_output_at(i, train)
        return s

    def get_output_ave(self, train=False):
        n = len(self.inputs)
        s = self.get_output_at(0, train)
        for i in range(1, n):
            s += self.get_output_at(i, train)
        s /= n
        return s

    def get_output_concat(self, train=False):
        inputs = [self.get_output_at(i, train) for i in range(len(self.inputs))]
        return T.concatenate(inputs, axis=self.concat_axis)

    def get_output_mul(self, train=False):
        s = self.get_output_at(0, train)
        for i in range(1, len(self.inputs)):
            s *= self.get_output_at(i, train)
        return s

    def get_output_dot(self, train=False):
        l1 = self.get_output_at(0, train)
        l2 = self.get_output_at(1, train)
        output = T.batched_tensordot(l1, l2, self.dot_axes)
        output = output.dimshuffle((0, 'x'))
        return output

    def get_output_cos(self, train=False):
        l1 = self.get_output_at(0, train)
        l2 = self.get_output_at(1, train)
        output, _ = theano.scan(lambda v1, v2: T.dot(v1, v2)/T.sqrt(T.dot(v1, v1) * T.dot(v2, v2)), sequences=[l1, l2], outputs_info=None)
        output = output.dimshuffle((0, 'x'))
        return output

    def get_output(self, train=False):
        mode = self.merge_mode
        if mode == 'join':
            return self.get_output_join(train)
        elif mode == 'concat':
            return self.get_output_concat(train)
        elif mode == 'sum':
            return self.get_output_sum(train)
        elif mode == 'ave':
            return self.get_output_ave(train)
        elif mode == 'mul':
            return self.get_output_mul(train)
        elif mode == 'dot':
            return self.get_output_dot(train)
        elif mode == 'cos':
            return self.get_output_dot(train)

    def get_input(self, train=False):
        res = []
        for i in range(len(self.inputs)):
            o = self.inputs[i].get_input(train)
            if not type(o) == list:
                o = [o]
            for output in o:
                if output not in res:
                    res.append(output)
        return res

    @property
    def input(self):
        return self.get_input()

    def supports_masked_input(self):
        return False

    def get_output_mask(self, train=None):
        return None

    def get_weights(self):
        weights = layer.get_weights()
        for m in self.inputs:
            weights += m.get_weights()
        return weights

    def set_weights(self, weights):
        nb_param = len(self.layer.params)
        self.layer.set_weights(weights[:nb_param])
        weights = weights[nb_param:]
        for i in range(len(self.inputs)):
            nb_param = len(self.inputs[i].params)
            self.inputs[i].set_weights(weights[:nb_param])
            weights = weights[nb_param:]

    def get_config(self):

        config = {"name": self.__class__.__name__,
                  "layer": self.layer.get_config,
                  "inputs": [m.get_config() for m in self.inputs],
                  "merge_mode": self.merge_mode,
                  "concat_axis": self.concat_axis,
                  "dot_axes": self.dot_axes
                  }
        base_config = super(Siamese, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SiameseHead(Layer):
    def __init__(self, head):
        self.head = head
        self.params = []
    def get_output(self, train=False):
        return self.get_input(train)

    @property
    def input_shape(self):
        return self.previous.layer.output_shape

    def get_input(self, train=False):
        return self.previous.get_output_at(self.head, train)

    def get_config(self):

        config = {"name": self.__class__.__name__,
                  "head": self.head
                  }

        base_config = super(SiameseHead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def set_previous(self, layer):
        self.previous = layer


def add_shared_layer(layer,inputs):
    input_layers = [l.layers[-1] for l in inputs]
    s = Siamese(layer, input_layers)
    for i in range(len(inputs)):
        sh = SiameseHead(i)
        inputs[i].add (s)
        inputs[i].add(sh)
