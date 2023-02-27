from source.gd_modifications import *
import numpy as np


def main() -> None:
    num_objects = 100
    dimension = 5

    x = np.random.rand(num_objects, dimension)
    y = np.random.rand(num_objects)

    descent_config = {
        'descent_name': 'some name that we will replace in the future',
        'kwargs': {
            'dimension': dimension,
            'loss': 'MAE',
            'max_iter': 52,
            'momentum': 0.9,
            'iter': 50,
        }
    }

    for descent_name in ['full', 'stochastic', 'momentum', 'adam']:
        descent_config['descent_name'] = descent_name
        descent = get_descent(descent_config)
        print(descent.__dict__)

        diff = descent.iterations(x, y)
        gradient = descent.calc_gradient(x, y)
        predictions = descent.predict(x)

        print(descent.calc_loss(x, y), '\n')

        assert gradient.shape[0] == dimension, f'Gradient failed for descent {descent_name}'
        assert diff.shape[0] == dimension, f'Weights failed for descent {descent_name}'
        assert predictions.shape == y.shape, f'Prediction failed for descent {descent_name}'
        assert descent.calc_loss(x, y) < 10.0


if __name__ == "__main__":
    main()
