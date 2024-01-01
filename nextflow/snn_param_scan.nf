
def aging_env = "$HOME/miniconda3/envs/aging"

params.config_file = "$HOME/code/ontogeny/configs/03-augmentation-scan.toml"
params.stage = 1
params.seed = 0
params.stageList = "${params.stage}".tokenize(',') as List

// tmp = '/some/path/to/file.txt'
// println file(tmp).getParent() / 'model.pt'


process create_grid {
    executor 'local'
    conda aging_env

    input:
    val config_file
    val stage

    output:
    stdout emit: file_name

    script:
    """
    python $HOME/code/ontogeny/scripts/07-batch-scan-hpparams.py \
    $HOME/code/ontogeny/configs/00-sizenorm_training_template.toml \
    $config_file --stage $stage --reset-run --seed $params.seed
    """
}

process run_grid {
    label 'gpu'
    memory 20.GB
    time { 3.h + (task.attempt - 1) * 30.m }
    conda aging_env

    input:
    val config_file

    output:
    val config_file

    script:
    """
    python $HOME/code/ontogeny/scripts/03-train-size-norm.py $config_file --checkpoint
    """
}

// TODO: add a process to run age classifier or something
workflow {
    configs = create_grid(Channel.value(params.config_file), Channel.fromList(params.stageList))
    configs = configs.map { file(it.trim()).readLines() }
        .flatten()

    run_grid(configs).view()
}
