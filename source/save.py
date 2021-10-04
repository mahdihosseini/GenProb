import csv
from datetime import datetime

def get_name(bench, dataset, hp):
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    name = "outputs/results-" + date_time +"-"+bench+"-"+dataset+"-"+hp+ ".csv"
    print("Writing to: ", name)

    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['test_acc','test_loss','train_acc','train_loss','gap','in_S_BE','in_C_BE','in_ER_BE','in_spec_BE','in_fro_BE','out_S_BE','out_C_BE','out_ER_BE','out_spec_BE','out_fro_BE',
        'in_S_AE','in_C_AE','in_ER_AE','in_spec_AE','in_fro_AE','out_S_AE','out_C_AE','out_ER_AE','out_spec_AE','out_fro_AE','path','model_id','type','in_weight_BE', 'out_weight_BE',
        'in_weight_AE', 'out_weight_AE'])

    return name

def write(name, text):
    with open(name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(text)
        writer.writerow([])
    return
