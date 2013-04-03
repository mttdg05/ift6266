import numpy as np


yaml_template="""
!obj:pylearn2.train.Train {
    dataset: &train !pkl: "data/train_prep.pkl",
    model: !obj:pylearn2.models.mlp.MLP {
        layers: [
                 !obj:pylearn2.models.mlp.SoftmaxPool {
                     layer_name: 'h1',
                     detector_layer_dim: %(detector_layer_dim1)i,
                     pool_size: 1,
                     sparse_init: 5,
                 },  !obj:pylearn2.models.mlp.SoftmaxPool {
                     layer_name: 'h2',
                     detector_layer_dim: %(detector_layer_dim2)i,
                     pool_size: 1,
                     sparse_init: 5,
                 },  !obj:pylearn2.models.mlp.SoftmaxPool {
                     layer_name: 'h3',
                     detector_layer_dim: %(detector_layer_dim3)i,
                     pool_size: 1,
                     sparse_init: 5,
                 },  !obj:pylearn2.models.mlp.Softmax {
                     layer_name: 'y',
                     n_classes: 7,
                     irange: 0.
                 }
                ],
        nvis: 2304,
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: %(batch_size)i,
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
                coeffs: [ %(weight_decay_coeff1)f, %(weight_decay_coeff2)f, %(weight_decay_coeff3)f, %(weight_decay_coeff4)f]
            }
            ]
        },

        termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased {
            channel_name: "valid_y_misclass",
            prop_decrease: 0.0,
            N: 10
        }
    },
    extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
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

n = 10
param_sets = []
learning_rates = np.random.uniform(low=0.002, high = 0.005, size = n)
#param_sets.append({'detector_layer_dim1': 100, 'detector_layer_dim2': 200, 'detector_layer_dim3': 300, 'batch_size': 100, 'learning_rate': 0.004, 'init_momentum': 0.99, 'weight_decay_coeff1': 0.0, 'weight_decay_coeff2': 0.0,  'weight_decay_coeff3': 0.0, 'weight_decay_coeff4': 0.0, 'save_path': ''.join(['"one_mlp_layer/softmaxPool/best_pkl/mlp_best', str(i), '.pkl"'])})

for i in xrange(n) :
  param_sets.append({'detector_layer_dim1': 100, 'detector_layer_dim2': 200, 'detector_layer_dim3': 400, 'batch_size': 100, 'learning_rate': 0.004, 'init_momentum': 0.99, 'weight_decay_coeff1': 0.0, 'weight_decay_coeff2': 0.0,  'weight_decay_coeff3': 0.0, 'weight_decay_coeff4': 0.0, 'save_path': ''.join(['"one_mlp_layer/softmaxPool/best_pkl/mlp_best', str(i), '.pkl"'])})

for i, param_set in enumerate(param_sets):
  yaml_str = yaml_template % param_set
  title = ''.join(['yaml/mlp_yaml', str(i), '.yaml'])
  f = open(title, 'w')
  f.write(yaml_str)
  f.close()

