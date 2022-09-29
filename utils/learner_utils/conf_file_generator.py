
class ConfFile:
    def __init__(self, conf_file, kb_file, problem_type, reasoner, alg, execution_time, noise_percentage=0):
        self.problem_type = problem_type
        self.reasoner = reasoner
        self.alg = alg
        self.execution_time = execution_time
        self.conf_file = conf_file
        self.file_block = []
        self.kb_file = kb_file
        self.noise_percentage = noise_percentage

    def write_file_header(self):
        self.file_block.append('prefixes = [ ("kb","http://localhost/foo#") ]\n')
        self.file_block.append('ks.type = "KB File"\n')
        self.file_block.append(f'ks.fileName = "{self.kb_file}"\n')
        self.file_block.append('\n')
        self.file_block.append(f'reasoner.type = "{self.reasoner}"\n')
        self.file_block.append('reasoner.sources = { ks }\n')
        self.file_block.append('\n')
        self.file_block.append(f'lp.type = "{self.problem_type}"\n')

    def write_file_tailer(self):
        self.file_block.append(f'alg.type = "{self.alg}"\n')
        self.file_block.append(f'alg.maxExecutionTimeInSeconds = {self.execution_time}\n')
        self.file_block.append(f'alg.noisePercentage = {self.noise_percentage}\n')
        # self.file_block.append(f'alg.maxDepth = 30\n')        

    def dump(self):
        out = open(self.conf_file, 'w')
        for s in self.file_block:
            out.write(s)
        out.flush()
        out.close()


class ClassLearnerConfFile(ConfFile):
    def __init__(self, concept, conf_file, kb_file, reasoner="oar", alg="celoe", execution_time=10, noise_percentage=0):
        super().__init__(conf_file, kb_file.split("/")[-1], "clp", reasoner, alg, execution_time, noise_percentage)
        self.concept = concept.lower()

    def write_problem_info(self):
        self.file_block.append(f'lp.classToDescribe = "kb:{self.concept}"\n')
        self.file_block.append(f'alg.ignoredConcepts = {{"kb:{self.concept}", "kb:train"}}\n')

    def write_conf_file(self):
        self.write_file_header()
        self.write_problem_info()
        self.write_file_tailer()
        self.dump()


class PosNegConfFile(ConfFile):
    def __init__(self, concept, pos, neg, conf_file, kb_file, reasoner="oar", alg="celoe", execution_time=10,
                 noise_percentage=0):
        super().__init__(conf_file, kb_file.split("/")[-1], "posNegStandard", reasoner, alg, execution_time,
                         noise_percentage)
        self.concept = concept
        self.pos = pos
        self.neg = neg

    def write_examples(self, ex):
        for i in range(ex.shape[0]):
            self.file_block.append(f'"kb:sample_{ex[i]}"')
            if i < (ex.shape[0]-1):
                self.file_block.append(',')
        self.file_block.append('}\n')

    def write_problem_info(self):
        self.file_block.append('lp.positiveExamples = {')
        self.write_examples(self.pos)
        self.file_block.append('lp.negativeExamples = {')
        self.write_examples(self.neg)
        self.file_block.append('\n')

    def write_conf_file(self):
        self.write_file_header()
        self.write_problem_info()
        self.write_file_tailer()
        self.dump()
