from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

resnet_arg_scope = resnet_utils.resnet_arg_scope

@add_arg_scope
def naive(inputs,
          depth,
          stride,
          rate=1,
          outputs_collections=None,
          scope=None):
  """naive residual unit.
  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.
  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.
  Returns:
    The ResNet unit's output.
  """
  with variable_scope.variable_scope(scope, 'naive_v1', [inputs]) as sc:
    depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
        shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
        shortcut = layers.conv2d(
          inputs,
          depth, [1, 1],
          stride=stride,
          activation_fn=None,
          scope='shortcut')

    residual = resnet_utils.conv2d_same(
        residual, depth, 3, stride, rate=rate, scope='conv1')
    residual = layers.conv2d(
        residual, depth, [3, 3], stride, activation_fn=None, scope='conv2')

    output = nn_ops.relu(shortcut + residual)

    return utils.collect_named_outputs(outputs_collections, sc.name, output)

def resnet_v1_block(scope, base_depth, num_units, stride):
  """Helper function for creating a resnet_v1 bottleneck block.
  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.
  Returns:
    A resnet_v1 bottleneck block.
  """
  return resnet_utils.Block(scope, naive, [{
      'depth': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth,
      'stride': stride
  }])

def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              reuse=None,
              scope=None):
    with variable_scope.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with arg_scope([layers.conv2d, naive, resnet_utils.stack_blocks_dense],outputs_collections=end_points_collection):
            with arg_scope([layers.batch_norm], is_training=is_training):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
                if global_pool:
                  # Global average pooling.
                    net = math_ops.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                if num_classes is not None:
                    net = layers.conv2d(net,num_classes, [1, 1],activation_fn=None,normalizer_fn=None,scope='logits')
                end_points = utils.convert_collection_to_dict(end_points_collection)
                if num_classes is not None:
                    end_points['predictions'] = layers_lib.softmax(net, scope='predictions')
                return net, end_points

def resnet_v1_18(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v1_18'):
  """ResNet-18 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1_block('block1', base_depth=64, num_units=2, stride=2),
      resnet_v1_block('block2', base_depth=128, num_units=2, stride=2),
      resnet_v1_block('block3', base_depth=256, num_units=2, stride=2),
      resnet_v1_block('block4', base_depth=512, num_units=2, stride=1),
  ];
    return resnet_v1(
      inputs,
      blocks,
      num_classes,
      is_training,
      global_pool,
      output_stride,
      include_root_block=True,
      reuse=reuse,
      scope=scope)