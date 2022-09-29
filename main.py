import multiprocessing
import os

import pandas as pd

from concept_encoding import encode_concepts
from dataset_processing import get_samples
from utils.learner_utils.generate_learn_files import test_extracted_concepts
from utils.seed_generator import SeedGenerator

PATH_TO_DATASET = "dataset/sns1_labels_ready.csv"
FEATURES_IDX = range(1, 15)
TARGET_CONCEPT = "target"
SEED_GENERATOR = SeedGenerator()


def sample_df(df, target_concept, target_value, test, seed, path_to_sampled_dfs,
              samples_id_column="sample_id", noise_percentage=0, csv_num=0):
    distribution = {target_value: None}
    samples_df = get_samples(df, target_concept, seed, id_column=samples_id_column, distribution=distribution).copy()
    samples_df[target_concept] = samples_df.apply(lambda row: int(row[target_concept] == target_value), axis=1)
    os.makedirs(path_to_sampled_dfs, exist_ok=True)
    concepts = list(samples_df.columns)
    samples_df.columns = encode_concepts(concepts, keep=[samples_id_column, target_concept])
    samples_df.to_csv(f'{path_to_sampled_dfs}{csv_num}_{len(samples_df)}_{noise_percentage}_{seed}_{test}.csv',
                      index=False)


def run_experiment_learning(target_concept, path_to_kb_files, path_to_conf_files,
                            path_to_lp_files, path_to_learner_output, learner_exec_time, path_to_sampled_dfs,
                            csv_num=0):
    df_name = list(filter(lambda s: s.split('_')[0] == str(csv_num), os.listdir(path_to_sampled_dfs)))[0]
    df = pd.read_csv(f'{path_to_sampled_dfs}{df_name}')
    df_name_split = df_name.split('_')
    noise_percentage = int(df_name_split[2])
    test = int(df_name_split[4][:-4])
    print(f"[INFO] LEARNING with {len(df)} samples... {test + 1}")
    test_extracted_concepts(df, target_concept, path_to_conf_files, path_to_kb_files,
                            path_to_lp_files, path_to_learner_output, "", learner_exec_time, noise_percentage,
                            test)


def run_experiment_learning_multi(target_concept, path_to_kb_files, path_to_conf_files, path_to_lp_files,
                                  path_to_learner_output, path_to_sampled_dfs, learner_exec_time, n_execs, n_procs=1):
    processes = []
    args = (target_concept, path_to_kb_files, path_to_conf_files, path_to_lp_files, path_to_learner_output,
            learner_exec_time, path_to_sampled_dfs)
    for ex in range(0, n_execs):
        if len(processes) == n_procs:
            for p in processes:
                p.join()
            processes = []
        p = multiprocessing.Process(target=run_experiment_learning, args=args, kwargs={'csv_num': ex})
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def main(file, path_to_data):
    df = pd.read_csv(path_to_data)
    repetitions = 10
    noise_percentages = [20]
    n_procs = 3
    learner_exec_time = 1
    n_execs = repetitions * len(noise_percentages)

    for target_value in range(1, 11):
        experiment_root = f"{file}/target{target_value}/"
        path_to_kb_files = f"{experiment_root}leaner/kb_conf/"
        path_to_conf_files = f"{experiment_root}leaner/kb_conf/"
        path_to_lp_files = f"{experiment_root}leaner/lp/"
        path_to_learner_output = f"{experiment_root}leaner/results/"
        path_to_sampled_dfs = f"{experiment_root}sampled_dfs/"
        os.makedirs(path_to_learner_output, exist_ok=True)
        os.makedirs(path_to_kb_files, exist_ok=True)
        os.makedirs(path_to_lp_files, exist_ok=True)
        os.makedirs(path_to_conf_files, exist_ok=True)
        os.makedirs(path_to_sampled_dfs, exist_ok=True)
        csv_num = 0

        for _, noise in enumerate(noise_percentages):
            for r in range(repetitions):
                print(path_to_data)
                sample_df(df, TARGET_CONCEPT, target_value, r, SEED_GENERATOR.next(), path_to_sampled_dfs,
                          noise_percentage=noise, csv_num=csv_num)
                csv_num += 1

        for ex in range(0, n_execs):
            run_experiment_learning(TARGET_CONCEPT, path_to_kb_files, path_to_conf_files, path_to_lp_files,
                                      path_to_learner_output, learner_exec_time, path_to_sampled_dfs, csv_num=ex )
        #run_experiment_learning_multi(TARGET_CONCEPT, path_to_kb_files, path_to_conf_files, path_to_lp_files,
        #                              path_to_learner_output, path_to_sampled_dfs, learner_exec_time, n_execs,
        #                              n_procs=n_procs)


if __name__ == "__main__":
     main(file='results1_sns1',path_to_data=PATH_TO_DATASET)
