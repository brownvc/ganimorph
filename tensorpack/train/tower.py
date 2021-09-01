# -*- coding: utf-8 -*-
# File: tower.py

import tensorflow as tf
import six
from abc import abstractmethod, ABCMeta

from ..utils.argtools import call_only_once, memoized
from ..utils.develop import HIDE_DOC
from ..utils import logger
from ..input_source import PlaceholderInput
from ..predict.base import OnlinePredictor

from ..tfutils.tower import TowerFuncWrapper, get_current_tower_context, PredictTowerContext
from ..tfutils.gradproc import FilterNoneGrad

from .base import Trainer

__all__ = ['SingleCostTrainer', 'TowerTrainer']


class TowerTrainer(Trainer):
    """
    Base trainers for models that can be built by calling a tower function under a :class:`TowerContext`.

    This is required by some features that replicates the model
    automatically, e.g. creating a predictor.

    To use features of :class:`TowerTrainer`, set `tower_func` and use it to build the graph.
    Note that `tower_func` can only be set once per instance.
    """

    _tower_func = None
    _predictors = []
    """
    List of OnlinePredictor ever created for this trainer.
    It is maintained for internal use.
    """

    @call_only_once
    def _set_tower_func(self, tower_func):
        assert isinstance(tower_func, TowerFuncWrapper), tower_func
        self._tower_func = tower_func

    @property
    def tower_func(self):
        """
        A :class:`TowerFuncWrapper` instance.
        See [tutorial on tower function](http://tensorpack.readthedocs.io/tutorial/trainer.html#tower-trainer)
        for more information.
        """
        return self._tower_func

    @tower_func.setter
    def tower_func(self, val):
        self._set_tower_func(val)

    @property
    def inputs_desc(self):
        """
        Returns:
            list[InputDesc]: metainfo about the inputs to the tower.
        """
        return self.tower_func.inputs_desc

    @property
    def towers(self):
        """
        Returns:
            a :class:`TowerTensorHandles` object, to
            access the tower handles by either indices or names.

        It is accessbile only after the graph is set up.
        With :meth:`towers`, you can then access many attributes of each tower:

        Example:

        .. code-block:: python

            # Access the conv1/output tensor in the first training tower
            trainer.towers.training()[0].get_tensor('conv1/output')
        """
        return self.tower_func.towers

    def get_predictor(self, input_names, output_names, device=0):
        """
        This method will build the tower under ``TowerContext(is_training=False)``,
        and returns a callable predictor with input placeholders & output tensors in this tower.

        Args:
            input_names (list): list of input names, matching the inputs declared for the trainer.
            output_names(list): list of tensor names without the tower prefix.
            device (int): build the predictor on device '/gpu:{device}' or use -1 for '/cpu:0'.

        Returns:
            an :class:`OnlinePredictor`.

        Example:

        .. code-block:: none

            # in the graph:
            interesting_tensor = tf.identity(x, name='fun')
            # in _setup_graph callback method:
            self._predictor = self.trainer.get_predictor(['input1', 'input2'], ['fun'])
            # After session is initialized (see Tutorials - Write a Callback), can use it by:
            outputs = self._predictor(input1, input2)

        The CycleGAN example and DQN example have more concrete use of this method.
        """
        assert self.tower_func is not None, "Must set tower_func on the trainer to use get_predictor()!"
        tower_name = 'tower-pred-{}'.format(device) if device >= 0 else 'tower-pred-cpu'
        device_id = device
        device = '/gpu:{}'.format(device_id) if device_id >= 0 else '/cpu:0'

        try:
            tower = self.tower_func.towers[tower_name]
            assert tower is not None, "This is a bug!"
        except KeyError:
            tower = None

        if tower is None:
            input = PlaceholderInput()
            input.setup(self.inputs_desc)

            vs_name = self._vs_name_for_predictor(device_id)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True), \
                    tf.device(device), PredictTowerContext(
                        tower_name, vs_name=vs_name):
                logger.info("Building graph for predict tower '{}' on device {} {}...".format(
                    tower_name, device,
                    "with variable scope '{}'".format(vs_name) if vs_name else ''))
                self.tower_func(*input.get_input_tensors())
            tower = self.tower_func.towers[tower_name]
        input_tensors = tower.get_tensors(input_names)
        output_tensors = tower.get_tensors(output_names)
        predictor = OnlinePredictor(input_tensors, output_tensors)
        self._predictors.append(predictor)
        return predictor

    @HIDE_DOC
    @call_only_once
    def initialize(self, session_creator, session_init):
        super(TowerTrainer, self).initialize(session_creator, session_init)
        # Predictors are created before creating the session, so they don't have an associated session.
        for pred in self._predictors:
            pred.sess = self.sess

    def _vs_name_for_predictor(self, device):
        towers = self.towers.training()
        available_ids = list(range(len(towers)))
        if device in available_ids:
            return towers[device].vs_name
        else:
            return towers[0].vs_name


@six.add_metaclass(ABCMeta)
class SingleCostTrainer(TowerTrainer):
    """
    Base class for single-cost trainer.

    Single-cost trainer has a :meth:`setup_graph` method which takes
    (inputs_desc, input, get_cost_fn, get_opt_fn), and build the training graph from them.

    To use a :class:`SingleCostTrainer` object, call `trainer.setup_graph(...); trainer.train(...)`.
    """

    COLOCATE_GRADIENTS_WITH_OPS = True
    """
    See `tf.gradients`. It sometimes can heavily affect performance when backward op does
    not support the device of forward op.
    """

    GATE_GRADIENTS = False
    """See `tf.gradients`. """

    AGGREGATION_METHOD = tf.AggregationMethod.DEFAULT
    """See `tf.gradients`. """

    @call_only_once
    def setup_graph(self, inputs_desc, input, get_cost_fn, get_opt_fn):
        """
        Responsible for building the main training graph for single-cost training.

        Args:
            inputs_desc ([InputDesc]):
            input (InputSource):
            get_cost_fn ([tf.Tensor] -> tf.Tensor): callable, takes some input tensors and return a cost tensor.
            get_opt_fn (-> tf.train.Optimizer): callable which returns an
                optimizer. Will only be called once.

        Note:
            `get_cost_fn` will be part of the tower function.
            It must follows the `rules of tower function.
            <http://tensorpack.readthedocs.io/en/latest/tutorial/trainer.html#tower-trainer>`_.
        """
        get_cost_fn = TowerFuncWrapper(get_cost_fn, inputs_desc)
        get_opt_fn = memoized(get_opt_fn)
        self.tower_func = get_cost_fn

        # TODO setup may want to register monitor as well??
        input_callbacks = self._setup_input(inputs_desc, input)
        train_callbacks = self._setup_graph(input, get_cost_fn, get_opt_fn)
        self.register_callback(input_callbacks + train_callbacks)

    @abstractmethod
    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        """
        Implement the logic to build the graph, with an :class:`InputSource`
        that's been setup already.

        Returns:
            [Callback]: list of callbacks needed
        """

    def _setup_input(self, inputs_desc, input):
        assert not input.setup_done()
        return input.setup(inputs_desc)

    def _make_get_grad_fn(self, input, get_cost_fn, get_opt_fn):
        """
        Internal use only.

        Returns:
            a get_grad_fn for GraphBuilder to use.
        """
        assert input.setup_done()

        def get_grad_fn():
            ctx = get_current_tower_context()
            cost = get_cost_fn(*input.get_input_tensors())
            assert isinstance(cost, tf.Tensor), cost
            assert cost.shape.ndims == 0, "Cost must be a scalar, but found {}!".format(cost)
            if not ctx.is_training:
                return None     # this is the tower function, could be called for inference

            if ctx.has_own_variables:
                varlist = ctx.get_collection_in_tower(tf.GraphKeys.TRAINABLE_VARIABLES)
            else:
                varlist = tf.trainable_variables()
            opt = get_opt_fn()
            grads = opt.compute_gradients(
                cost, var_list=varlist,
                gate_gradients=self.GATE_GRADIENTS,
                colocate_gradients_with_ops=self.COLOCATE_GRADIENTS_WITH_OPS,
                aggregation_method=self.AGGREGATION_METHOD)
            grads = FilterNoneGrad().process(grads)
            return grads

        return get_grad_fn
