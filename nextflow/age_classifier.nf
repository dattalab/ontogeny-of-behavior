
def aging_env = "$HOME/miniconda3/envs/aging"

params.model_folder = "/n/groups/datta/win/longtogeny/size_norm/models/bottleneck_param_scan_00"


process transform_data {
    label 'gpu'
    memory 30.GB
    time { 40.m * task.attempt }
    conda aging_env

    input:
    val model_path

    output:
    stdout emit: file_name

    script:
    println model_path

    """
    #!/bin/env python
    from aging.size_norm.validation import transform_data
    from aging.organization.paths import ValidationPaths

    val_paths = ValidationPaths()
    transform_save_path = transform_data(val_paths.age_classifier, "${model_path}")
    print(str(transform_save_path))
    """
}


// process age_classification {

// }

workflow {
    files = file("$params.model_folder/**/model.pt")
    if (files.size() == 0) {
        files = file("$params.model_folder/model.pt")
        ch = Channel.from(files)
    } else {
        ch = Channel.fromList(files)
    }
    received = transform_data(ch)
    received.view()
}