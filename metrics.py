import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error

def nlp_calculate_mse(gt_decoded, pred_decoded):
    gt_output, gt_poi = text_to_mob(gt_decoded)
    pred_output, pred_poi = text_to_mob(pred_decoded)
    return mean_squared_error(gt_output, pred_output), mean_squared_error(gt_poi, pred_poi)

def nlp_calculate(gt_decoded, pred_decoded):
    gt_output, gt_poi = text_to_mob(gt_decoded)
    pred_output, pred_poi = text_to_mob(pred_decoded)
    rmse, mae = np_evaluate(gt_output, pred_output)
    return rmse, mae

def np_evaluate(gt_output, pred_output):
    rmse = mean_squared_error(gt_output, pred_output, squared = False)
    mae = mean_absolute_error(gt_output, pred_output)
    return rmse, mae

def text_to_mob(text_data):
    output_data = []
    poi_id = []
    for line in text_data:
        out = int(line.split(" ")[3])
        poi = int(line.split(" ")[-1].replace(".", ""))
        output_data.append(out)
        poi_id.append(poi)
    output = np.reshape(output_data, [len(text_data), 1])
    poi_id = np.reshape(poi_id, [len(text_data), 1])
    return output, poi_id