import numpy as np

yaml_template = """
!obj:pylearn2.train.Train {
    dataset: &train !pkl: "data/train_prep.pkl",
    model: !obj:pylearn2.models.mlp.MLP {
        batch_size: 40,
        input_space: !obj:pylearn2.space.Conv2DSpace {
            shape: [48, 48],
            num_channels: 1
        },
        layers: [ !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                     layer_name: 'h2',
                     output_channels: 64,
                     irange: .05,
                     kernel_shape: [5, 5],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     max_kernel_norm: %(max_kernel_norm1)f,
                     border_mode: 'full'
                 }, !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                     layer_name: 'h3',
                     output_channels: 64,
                     irange: .05,
                     kernel_shape: [5, 5],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     max_kernel_norm: %(max_kernel_norm2)f,
                     border_mode: 'full'
                 }, !obj:pylearn2.models.mlp.Softmax {
                     max_col_norm: 1.9365,
                     layer_name: 'y',
                     n_classes: 7,
                     istdev: .05
                 }
                ],
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 40,
        learning_rate: %(learning_rate)f,
        init_momentum: %(init_momentum)f,
        monitoring_dataset:
            {
                'train' : *train,
                'valid' : !pkl: "data/valid_prep.pkl"
            },
        cost: !obj:pylearn2.costs.cost.SumOfCosts { costs: [
            !obj:pylearn2.costs.cost.MethodCost {
                method: 'cost_from_X',
                supervised: 1
            }, !obj:pylearn2.models.mlp.WeightDecay {
                coeffs: [ .0, .0, .0]
            }
            ]
        },
        termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased {
            channel_name: "valid_y_misclass",
            prop_decrease: 0.,
            N: 200
        }
    },
    extensions:
        [ !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
             channel_name: 'valid_y_misclass',
             save_path: %(save_path)s
        }, !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
            start: 1,
            saturate: 10,
            final_momentum: .99
        }
    ]
}
"""
n = 25
param_sets = []
learning_rates = np.random.uniform(low=0.002, high = 0.005, size = n)
learning_rates[0] = 0.003795
for i in xrange(n) :
  param_sets.append({'max_kernel_norm1': 1.9365, 'max_kernel_norm2' : 1.9365, 'learning_rate': learning_rates[i], 'init_momentum': 0.0, 'save_path': ''.join(['"conv/best_pkl/conv_best', str(i), '.pkl"'])})

for i, param_set in enumerate(param_sets):
  yaml_str = yaml_template % param_set
  title = ''.join(['yaml/conv_yaml', str(i), '.yaml'])
  f = open(title, 'w')
  f.write(yaml_str)
  f.close()


