import os
import subprocess


def write_kb_file(path_to_kb, df, target_concept):
    file_block = []
    df= df.astype({"sample_id":int}) #test
    preds = df.columns.values.tolist()
    preds.remove("sample_id")
    preds.remove(target_concept)
    for idx, row in df.iterrows():
        file_block.append(f"sample(sample_{row['sample_id']}).\n")
        for pred in preds:
            if row[pred] == 1:
                file_block.append(f"{pred.lower()}(sample_{row['sample_id']}).\n")

        file_block.append("\n")
    out = open(path_to_kb, 'w')
    for line in file_block:
        out.write(line)


def write_test_kb_file(path_to_kb, df, target_concept):
    file_block = []
    preds = df.columns.values.tolist()
    preds.remove("sample_id")
    preds.remove(target_concept)
    for idx, row in df.iterrows():
        file_block.append(f"sample(train_{row['sample_id']}).\n")
        file_block.append(f"thing(sample_{row['sample_id']}).\n")
        if row[target_concept] == 1:
            file_block.append(f"{target_concept.lower()}(sample_{row['sample_id']}).\n")
        for pred in preds:
            if row[pred] == 1:
                file_block.append(f"{pred.lower()}(sample_{row['sample_id']}).\n")

        file_block.append("\n")
    out = open((path_to_kb[:-2] + 'lp'), 'w')
    for line in file_block:
        out.write(line)


def write_owl_file(path_to_kb, df, target_concept):
    temp_file = f"{path_to_kb[:-2]}.tmp"
    df_less = df.drop(columns=[target_concept])
    df_less.to_csv(temp_file, index=False)
    subprocess.call(['java', '-jar', "owl_file_generator/generate_owl_files.jar", temp_file, path_to_kb])
    os.remove(temp_file)


def write_conf_file(problem_class, target_concept, df, path_to_conf, path_to_kb, reasoner_type, learner_exec_time):
    pos = df[df[target_concept] == 1]
    neg = df[df[target_concept] == 0]
    pos_arr = pos["sample_id"].to_numpy(dtype=int)
    neg_arr = neg["sample_id"].to_numpy(dtype=int)
    conf_file = problem_class(target_concept, pos_arr, neg_arr, path_to_conf, path_to_kb, reasoner_type, learner_exec_time)
