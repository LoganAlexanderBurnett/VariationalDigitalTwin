from __future__ import annotations

import numpy as np
import torch
from joblib import Parallel, delayed


def _align_model_batch_size_with_input(model, current_seq):
    expected_batch = int(current_seq.shape[0])
    if getattr(model, 'init_x', None) is None or int(model.init_x.shape[0]) != expected_batch:
        model.set_batch_size(expected_batch)


def predict_deterministic_run(model, current_seq):
    _align_model_batch_size_with_input(model, current_seq)
    with torch.no_grad():
        pred_tensor, state = model.predict(current_seq)
    return pred_tensor.detach().cpu().numpy().ravel(), state


def predict_with_uncertainty(model, current_seq, mc_samples=100, n_jobs=6):
    base_train = torch.nn.Module.train
    base_train(model, False)
    _align_model_batch_size_with_input(model, current_seq)

    def sample_prediction():
        with torch.no_grad():
            pred_tensor, _ = model.predict(current_seq)
        return pred_tensor.detach().cpu().numpy().ravel()

    preds = Parallel(n_jobs=n_jobs)(delayed(sample_prediction)() for _ in range(mc_samples))
    stack = np.stack(preds, axis=0)
    mean = stack.mean(axis=0)
    lower = np.percentile(stack, 2.5, axis=0)
    upper = np.percentile(stack, 97.5, axis=0)
    std = stack.std(axis=0)
    return {
        'samples': stack,
        'mean': mean,
        'lower': lower,
        'upper': upper,
        'std': std,
    }
