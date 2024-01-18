def aging_env = "$HOME/miniconda3/envs/aging"
def moseq_env = "$HOME/miniconda3/envs/moseq2-app"

params.moseq_folder = "/n/groups/datta/win/longtogeny/data/ontogeny/version_11"
params.size_norm_name = "win_size_norm_frames_v7"

process organize_extractions {
    executor 'local'
    conda aging_env

    input:
    val moseq_folder
    
    output:
    val moseq_folder
    
    script:
    """
    #!/bin/env python
    from aging.moseq_modeling.pca import organize_files

    organize_files("${params.size_norm_name}", "${moseq_folder}/all_data_pca")
    """
}

process apply_pca {
    label "short"
    memory 13.GB
    time 130.m
    conda moseq_env
    maxRetries 2

    input:
    val moseq_folder
    
    output:
    val moseq_folder
    
    script:
    """
    moseq2-pca apply-pca \
        -i "${moseq_folder}/all_data_pca" \
        -o "${moseq_folder}/all_data_pca" \
        --h5-path ${params.size_norm_name} -n 50 --cluster-type slurm \
        --pca-file "${moseq_folder}/_pca/pca.h5" \
        --timeout 8 -w 02:00:00 -m 13GB -q short \
        --dask-cache-path /n/scratch/users/w/wg41/tmp \
        --batch-apply --overwrite-pca-apply 1
    """
}

process apply_moseq_model {
    label "gpu"
    memory 15.GB
    time 45.m
    conda "$HOME/miniconda3/envs/jax-moseq-og"
    maxRetries 2

    input:
    val moseq_folder
    
    output:
    val moseq_folder
    
    script:
    """
    #!/bin/env python
    from aging.moseq_modeling.arhmm import apply_arhmm

    apply_arhmm("${moseq_folder}", batch_size=80, robust=False)
    """
}

process get_experiment_names {
    executor 'local'
    conda aging_env

    input:
    val moseq_folder
    
    output:
    path "experiments.txt"
    
    script:
    """
    #!/bin/env python
    from aging.organization.paths import get_experiment_grouped_files

    files = get_experiment_grouped_files()
    with open("experiments.txt", "w") as f:
        f.write("\n".join(map(str, files.keys())))
    """
}

process create_dataframe {
    label 'short'
    cpus 10
    memory 75.GB
    time { 2.h * task.attempt }
    maxRetries 2

    input:
    val experiment
    
    output:
    val experiment
    
    script:
    """
    #!/bin/env python
    from aging.moseq_modeling.dataframe import aggregate_into_dataframe, filter_session_length, add_mouse_id, corrections, mouse_filter

    df = aggregate_into_dataframe(
        "${experiment}",
        "${params.moseq_folder}",
        "${params.size_norm_name}"
    )
    df = filter_session_length(df, experiment="${experiment}")
    df = corrections(df, "${experiment}")
    df = add_mouse_id(df, "${experiment}")
    df = mouse_filter(df, "${experiment}")
    df.to_parquet("${params.moseq_folder}/${experiment}_syllable_df_v00.parquet", compression="brotli")
    """
}

process create_usage_dataframe {
    label 'short'
    cpus 1
    memory = 60.GB
    time 10.m

    input:
    val experiment
    
    output:
    val experiment
    
    script:
    """
    #!/bin/env python
    import pandas as pd
    from aging.moseq_modeling.dataframe import create_usage_dataframe, normalize_dataframe, filter_high_usage, experiment_specific_filter

    df = pd.read_parquet("${params.moseq_folder}/${experiment}_syllable_df_v00.parquet")

    # syllable counts (with raw syllable labels)
    df = create_usage_dataframe(df)
    df = filter_high_usage(df)
    df.to_parquet("${params.moseq_folder}/${experiment}_raw_counts_matrix_v00.parquet")

    # normalized syllable usage (should sum to 1)
    norm_df = normalize_dataframe(df)
    norm_df.to_parquet("${params.moseq_folder}/${experiment}_raw_usage_matrix_v00.parquet")

    # filter for bad sessions for both counts and usage
    df = experiment_specific_filter(df, "${experiment}")
    df.to_parquet("${params.moseq_folder}/${experiment}_filtered_counts_matrix_v01.parquet")
    
    norm_df = experiment_specific_filter(norm_df, "${experiment}")
    norm_df.to_parquet("${params.moseq_folder}/${experiment}_filtered_usage_matrix_v01.parquet")
    """
}


process relabel_dataframe {
    executor 'local'
    conda aging_env

    input:
    val experiments

    script:
    """
    #!/bin/env python

    """
}

workflow {
    out = organize_extractions(params.moseq_folder)
    out = apply_pca(out)
    out = apply_moseq_model(out)
    // TODO: make a list of dataframes then send to create_dataframe
    experiment_path = get_experiment_names(out).map { it.readLines() }
        .flatten()
        .filter { it != "" && it != null && it != "\n" }
    dfs = create_dataframe(experiment_path)
    dfs = create_usage_dataframe(dfs)
    relabel_dataframe(dfs.collect())
}