import itertools
import numpy as np
import pandas as pd

if __name__ == '__main__':
    num_task = 64
    sample_per_task = 10
    amp_range = [0.1, 5.0]
    phase_range = [0, np.pi / 2]
    input_range = [-5, 5]

    amp = np.random.uniform(amp_range[0], amp_range[1], [num_task])
    phase = np.random.uniform(phase_range[0], phase_range[1], [num_task])

    outputs = np.zeros([num_task, sample_per_task])
    inputs = np.zeros([num_task, sample_per_task])
    for i in range(num_task):
        inputs[i] = np.random.uniform(input_range[0], input_range[1], [sample_per_task])
        outputs[i] = amp[i] * np.sin(inputs[i] - phase[i])

    df = pd.DataFrame({'task_id': [item for p in range(0, num_task) for item in [p] * sample_per_task],
                  'amp': [item for a in amp for item in [a] * sample_per_task],
                  'phase': [item for p in phase for item in [p] * sample_per_task], 'X': list(itertools.chain(*inputs)),
                  'y': list(itertools.chain(*outputs))})
    df.to_csv('data_train.csv', index=None)

    num_task = 1
    sample_per_task = 500
    amp_range = [0.01, 0.02]
    phase_range = [np.pi / 2, np.pi]
    input_range = [-5, 5]

    amp = np.random.uniform(amp_range[0], amp_range[1], [num_task])
    phase = np.random.uniform(phase_range[0], phase_range[1], [num_task])

    outputs = np.zeros([num_task, sample_per_task])
    inputs = np.zeros([num_task, sample_per_task])
    for i in range(num_task):
        inputs[i] = np.random.uniform(input_range[0], input_range[1], [sample_per_task])
        outputs[i] = amp[i] * np.sin(inputs[i] - phase[i])

    df = pd.DataFrame({'task_id': [item for p in range(0, num_task) for item in [p] * sample_per_task],
                       'amp': [item for a in amp for item in [a] * sample_per_task],
                       'phase': [item for p in phase for item in [p] * sample_per_task],
                       'X': list(itertools.chain(*inputs)),
                       'y': list(itertools.chain(*outputs))})
    df.to_csv('data_test.csv', index=None)
