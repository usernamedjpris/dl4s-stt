import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import datetime as d
import glob
import plotext as pltt
import inquirer


def get_data_from_epoch(epoch):
    data = []
    for e in epoch:
        table = e.split('{')[1][:-1].split(',')
        k = [x.split(":")[0].split('"')[1] for x in table]
        v = [x.split(":")[1] for x in table]
        values = []
        for x in v:
            if '"' in x:
                values.append(float(x.split('"')[1]))
            else:
                values.append(float(x))
        v = values
        d = {k:v for (k,v) in zip(k,v)}
        data.append(d)
    return data


def get_data_from_raw_log(args):

    with open(args.file, 'r') as inputfile:
        content = inputfile.read()

    c = content.split('\n')[270:]

    train_epoch = [x for x in c if '[train_inner]' in x]
    valid_epoch = [x for x in c if '[valid]' in x]

    train_data = get_data_from_epoch(train_epoch)
    train_data = pd.DataFrame.from_dict(train_data)
    train_data = train_data.set_index("epoch")
    
    valid_data = get_data_from_epoch(valid_epoch)
    valid_data = pd.DataFrame.from_dict(valid_data)
    valid_data = valid_data.set_index("epoch")

    return valid_data, train_data

def main(args):

    valid_data, train_data = get_data_from_raw_log(args)

    if args.epoch == "t":
        data = train_data
        x = list(data["num_updates"])
    else:
        data = valid_data
        x = list(data["valid_num_updates"])


    questions = [
    inquirer.Checkbox('cols',
                        message="Valeurs pour le graphe",
                        choices=list(data.columns),
                        ),
    ]
    answers = inquirer.prompt(questions)

    pltt.subplots(len(answers["cols"]), 1)
    for i in range(len(answers["cols"])):

        # Print dans le terminal avec plottext
        pltt.subplot(i+1,1)
        pltt.plot(x, data[answers["cols"][i]])
        pltt.title(answers["cols"][i])

        if args.save == 'y':
            # Sauvegarde de plot propres avec matplotlib
            plt.plot(x, data[answers["cols"][i]])
            plt.title(answers["cols"][i])
            plt.xlabel("num_updates")
            plt.ylabel(answers["cols"][i])

            plt.savefig(os.path.join(args.output_dir, answers["cols"][i]))
            plt.clf()

    
    pltt.show()



if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", default=None, type=str,
                        required=True, help="Chemin vers le hydra log")
    parser.add_argument("-e", "--epoch", default='t', type=str,
                        required=True, help="(t/v) Info sur les epoch train ou valid")
    parser.add_argument("-s", "--save", default='n', type=str,
                        required=False, help="y/n si vous voulez sauvegarder les graphes")
    parser.add_argument("-o", "--output_dir", default='.', type=str,
                        required=False, help="Dossier de sortie des graphes")



    args = parser.parse_args()
    main(args)
