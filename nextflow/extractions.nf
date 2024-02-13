
def aging_env = "$HOME/miniconda3/envs/aging"
def moseq_env = "$HOME/miniconda3/envs/moseq2-app"

params.size_norm_name = "win_size_norm_frames_v7"
params.snn_path = ""

process find_extractable_files {
    executor "local"
    conda aging_env

    output:
    path "extractable_files.txt"

    script:
    """
    #!/bin/env python
    from aging.organization.paths import get_experiment_grouped_depth_files
    from aging.organization.util import not_extracted, multi_filter, no_depth_doubles

    files = get_experiment_grouped_depth_files()

    with open("extractable_files.txt", "w") as f:
        for exp, v in files.items():
            filtered_files = list(multi_filter(no_depth_doubles, not_extracted, seq=v))
            if len(filtered_files) > 0:
                f.write("\\n".join(map(str, filtered_files)))
                f.write("\\n")
    """
}

process extract {
    label "short"
    memory 10.GB
    cpus 1
    time { 40.m * task.attempt }
    maxRetries 1
    conda moseq_env

    input:
    val depth_file

    output:
    val depth_file

    script:
    """
    {
    moseq2-extract extract "${depth_file}" --config-file "/n/groups/datta/win/longtogeny/data/extractions/config.yaml"
    } || {
    echo "Extract command for ${depth_file} did not work"
    }
    """
}

process compress {
    label "short"
    cpus 1
    memory 13.GB
    time { 75.m * task.attempt }
    maxRetries 1
    conda moseq_env

    input:
    val depth_file

    script:
    """
    moseq2-extract convert-raw-to-avi "${depth_file}" --delete
    """
}

process find_files_to_normalize {
    executor "local"
    conda aging_env

    input:
    val extraction_results

    output:
    path "files_to_normalize.txt"

    script:
    """
    #!/bin/env python
    from aging.organization.paths import get_experiment_grouped_files
    from aging.size_norm.apply import hasnt_key

    extracted_files = get_experiment_grouped_files()

    test = hasnt_key(key="${params.size_norm_name}")

    with open("files_to_normalize.txt", "w") as f:
        for k, v in extracted_files.items():
            for file in filter(test, v):
                f.write(str(file) + "\\n")
    """
}


process size_normalize {
    label "gpu"
    memory 8.GB
    time { 35.m * task.attempt }
    maxRetries 1
    conda aging_env

    input:
    val file_collection

    script:
    """
    #!/bin/env python
    import torch
    from pathlib import Path
    from aging.size_norm.apply import predict_and_save

    collection = "${file_collection}"
    collection = [s.strip() for s in collection[1:-1].split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load("${params.snn_path}", map_location=device)

    def predict(file):
        try:
            predict_and_save(Path(file), model, "${params.size_norm_name}", rescale=False, clean_noise=True)
        except Exception as e:
            print("Exception for file", file)
            print(e)
            print('---' * 3)

    for file in collection:
        predict(file)
    """
}

workflow {
    files = find_extractable_files()
    files = files.map { it.readLines() }
        .flatten()
        .filter { it != "" && it != null && it != "\n" }
    files = extract(files)
    avi_files = files.filter { it.endsWith(".dat") }
    compress(avi_files)
    // collect extractions, because they won't affect output of this find function
    norm_files = find_files_to_normalize(files.collect())
    norm_files = norm_files.map { it.readLines() }
        .flatten()
        .filter { it != "" && it != null && it != "\n" }
        .collate(25)
    size_normalize(norm_files)
}
