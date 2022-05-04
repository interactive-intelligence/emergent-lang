import shapedata
import numpy as np
import sklearn

def dlsm_acc(outa, outb, y):
  '''
  Returns the accuracy, averaging two output predictions.
  '''
  return sklearn.metrics.accuracy_score(np.round((outa+outb)/2), y)

def alec_ood_eval(model, data, color_spec=None, num_batches=10):
  '''
  Assumes model can be directly called on inputs like model(x).
  Assumes the dataset passed in is of type shapedata.AlecOODShapeData.
  Returns a dictionary with accuracies for in-distribution, out-of-distribution,
    and out-of-distribution + color spec (if not None) accuracies, evaluated
    across a specified number of batches.
  '''
  accs = {'id':[], 'ood':[], 'ood_color':[]}
  for eval_batch in range(num_batches):
    (x1, x1_shapes), (x2, x2_shapes), y = data.create_batch()
    outa, outb = model(np.concatenate([x1, x2]))
    accs['id'].append(dlsm_acc(outa, outb, y))
  for eval_batch in range(num_batches):
    (x1, x1_shapes), (x2, x2_shapes), y = data.create_batch_ood()
    outa, outb = model(np.concatenate([x1, x2]))
    accs['ood'].append(dlsm_acc(outa, outb, y))
  if color_spec and color_spec != []:
    for eval_batch in range(num_batches):
      (x1, x1_shapes), (x2, x2_shapes), y = data.create_batch_ood(color_spec=color_spec)
      outa, outb = model(np.concatenate([x1, x2]))
      accs['ood_color'].append(dlsm_acc(outa, outb, y))
  for key in accs:
    accs[key] = sum(accs[key])/len(accs[key])
  return accs  
