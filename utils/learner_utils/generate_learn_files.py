import subprocess

import numpy as np

from utils.learner_utils.conf_file_generator import PosNegConfFile, ClassLearnerConfFile
from utils.learner_utils.kb_file_generator import write_kb_file, write_test_kb_file, write_owl_file


def generate_learn_files(df, target_concept, path_to_conf_file, path_to_kb, path_to_lp, learner_exec_time,
                         noise_percentage, pos_neg=True):
    #write_test_kb_file(path_to_lp, df, target_concept)
    if pos_neg:
        write_kb_file(path_to_kb, df, target_concept)
        #write_owl_file(path_to_kb, df, target_concept)
        pos = df[df[target_concept] == 1]
        neg = df[df[target_concept] == 0]
        pos_arr = pos["sample_id"].to_numpy(dtype=int)
        neg_arr = neg["sample_id"].to_numpy(dtype=int)
        conf_file = PosNegConfFile(target_concept, pos_arr, neg_arr, path_to_conf_file, path_to_kb, reasoner="cwr",
                                   execution_time=learner_exec_time, noise_percentage=noise_percentage)
    else:
        write_test_kb_file(path_to_kb, df, target_concept)
        conf_file = ClassLearnerConfFile(target_concept, path_to_conf_file, path_to_lp, reasoner="cwr",
                                         execution_time=learner_exec_time, noise_percentage=noise_percentage)

    conf_file.write_conf_file()


def run_experiment(path_to_conf_file, path_to_learner_output):
    out = open(path_to_learner_output, 'w')
    subprocess.call(['C:/Users/loure/Desktop/Tese/dllearner-1.5.0/bin/cli.bat', path_to_conf_file],
                    stdout=out, stderr=subprocess.DEVNULL)
    out.close()


def calculate_error(true_df, mnns_df):
    error = 0
    for idx, row in true_df.iterrows():
        for idx_m, row_m in mnns_df[mnns_df["sample_id"] == row["sample_id"]].iterrows():
            for col in list(true_df.columns):
                if row[col] != row_m[col]:
                    error += 1
                    break
    return (error/len(true_df))*100


def test_all_concepts(df, target_concept, path_to_conf_files, path_to_kb_files, path_to_lp_files,
                      path_to_learner_output, labels_mode, learner_exec_time, noise_percentage, test, real_noise=0):
    print(f"[INFO] Testing all concepts using {labels_mode} labels")
    path_to_conf_file = f"{path_to_conf_files}{target_concept}_{labels_mode}_{noise_percentage}_all_{round(real_noise)}_" \
                        f"{len(df)}_{test}.conf"
    path_to_kb = f"{path_to_kb_files}{target_concept}_{labels_mode}_{noise_percentage}_all_{round(real_noise)}_{len(df)}_" \
                 f"{test}.kb"
    path_to_lp = f"{path_to_lp_files}{target_concept}_{labels_mode}_{noise_percentage}_all_{round(real_noise)}_{len(df)}_" \
                 f"{test}.lp"
    path_to_learner_output = f'{path_to_learner_output}{target_concept}_{labels_mode}_{noise_percentage}_all_' \
                             f'{round(real_noise)}_{len(df)}_{test}.txt'
    generate_learn_files(df, target_concept, path_to_conf_file, path_to_kb, path_to_lp, learner_exec_time,
                         noise_percentage)
    run_experiment(path_to_conf_file, path_to_learner_output)


def test_extracted_concepts(df, target_concept, path_to_conf_files, path_to_kb_files,
                            path_to_lp_files, path_to_learner_output, labels_mode, learner_exec_time, noise_percentage,
                            test, real_noise=0):
    print(f"[INFO] Testing extracted concepts using {labels_mode} labels")
    path_to_conf_file = f"{path_to_conf_files}{target_concept}_{labels_mode}_{noise_percentage}_extracted_" \
                        f"{round(real_noise)}_{len(df)}_{test}.conf"
    path_to_kb = f"{path_to_kb_files}{target_concept}_{labels_mode}_{noise_percentage}_extracted_{round(real_noise)}_" \
                 f"{len(df)}_{test}.kb"
    path_to_lp = f"{path_to_lp_files}{target_concept}_{labels_mode}_{noise_percentage}_extracted_{round(real_noise)}_" \
                 f"{len(df)}_{test}.lp"
    path_to_learner_output = f'{path_to_learner_output}{target_concept}_{labels_mode}_{noise_percentage}_extracted_' \
                             f'{round(real_noise)}_{len(df)}_{test}.txt'
    generate_learn_files(df, target_concept, path_to_conf_file, path_to_kb, path_to_lp, learner_exec_time,
                         noise_percentage)
    run_experiment(path_to_conf_file, path_to_learner_output)


def test_concepts(df, extracted_concepts, target_concept, path_to_conf_files, path_to_kb_files, path_to_lp_files,
                  path_to_learner_output, labels_mode, learner_exec_time, noise_percentage, test):
    test_all_concepts(df, target_concept, path_to_conf_files, path_to_kb_files, path_to_lp_files,
                      path_to_learner_output, labels_mode, learner_exec_time, noise_percentage, test)
    test_extracted_concepts(df, extracted_concepts, target_concept, path_to_conf_files, path_to_kb_files,
                            path_to_lp_files, path_to_learner_output, labels_mode, learner_exec_time, noise_percentage,
                            test)
