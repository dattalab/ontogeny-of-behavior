def aging_env = "$HOME/miniconda3/envs/aging"
def moseq_env = "$HOME/miniconda3/envs/moseq2-app"

params.moseq_folder = "/n/groups/datta/win/longtogeny/data/ontogeny/version_11"
params.size_norm_name = "win_size_norm_frames_v7"
params.df_version = 0
params.old = 1

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
    time { 60.m * task.attempt }
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
    from toolz import dissoc
    from aging.organization.paths import get_experiment_grouped_files

    files = get_experiment_grouped_files()
    files = dissoc(files, "dana_ontogeny_dana_ontogeny")
    with open("experiments.txt", "w") as f:
        f.write("\\n".join(map(str, files.keys())))
    """
}

process create_dataframe {
    label 'short'
    cpus 11
    memory {
        if (experiment.contains("longtogeny_v2"))
            return 100.GB
        else if (experiment.contains("longtogeny"))
            return 90.GB
        else if (experiment.contains("ontogeny"))
            return 40.GB
        else
            return 50.GB
    }
    time { 90.m * task.attempt }
    maxRetries 2
    conda aging_env

    input:
    val experiment
    
    output:
    val experiment
    
    script:
    """
    #!/bin/env python
    from aging.moseq_modeling.dataframe import (
        aggregate_into_dataframe, filter_session_length,
        add_mouse_id, corrections, mouse_filter
    )

    df_version = int(${params.df_version})

    df = aggregate_into_dataframe(
        "${experiment}",
        "${params.moseq_folder}",
        "${params.size_norm_name}",
        old=bool(${params.old}),
    )
    if df is not None:
        print("Filtering sessions")
        df = filter_session_length(df, experiment="${experiment}")
        print("Correcting errors")
        df = corrections(df, "${experiment}")
        print("Adding mouse ids")
        df = add_mouse_id(df, "${experiment}")
        print("Filtering unwanted mice")
        df = mouse_filter(df, "${experiment}")
        print("Saving dataframe")
        df.to_parquet(f"${params.moseq_folder}/${experiment}_syllable_df_v{df_version:02d}.parquet", compression="brotli")
    """
}

process create_usage_dataframe {
    label 'short'
    cpus 1
    memory {
        if (experiment.contains("longtogeny_v2"))
            return 100.GB
        else if (experiment.contains("longtogeny"))
            return 90.GB
        else if (experiment.contains("ontogeny"))
            return 40.GB
        else
            return 50.GB
    }
    time 10.m
    conda aging_env

    input:
    val experiment
    
    output:
    val experiment
    
    script:
    """
    #!/bin/env python
    import pandas as pd
    from pathlib import Path
    from aging.moseq_modeling.dataframe import (
        create_usage_dataframe, normalize_dataframe, filter_high_usage
    )

    df_version = int(${params.df_version})

    file = Path(f"${params.moseq_folder}/${experiment}_syllable_df_v{df_version:02d}.parquet")

    if file.exists():
        df = pd.read_parquet(file)

        # syllable counts (with raw syllable labels)
        df = create_usage_dataframe(df)
        df = filter_high_usage(df)
        df.to_parquet(f"${params.moseq_folder}/${experiment}_raw_counts_matrix_v{df_version:02d}.parquet")

        # normalized syllable usage (should sum to 1)
        norm_df = normalize_dataframe(df)
        norm_df.to_parquet(f"${params.moseq_folder}/${experiment}_raw_usage_matrix_v{df_version:02d}.parquet")
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
    import pandas as pd
    from pathlib import Path
    from aging.behavior.syllables import relabel_by_usage

    df_version = int(${params.df_version})
    folder = Path("${params.moseq_folder}")

    files = sorted(folder.glob(f"*raw_*_matrix_v{df_version:02d}.parquet"))

    def get_usage_map():
        df = pd.read_parquet(folder / f"ontogeny_males_raw_counts_matrix_v{df_version:02d}.parquet")
        counts = df.sum().sort_values(ascending=False).index.astype(int)
        leftovers = set(range(100)) - set(counts)
        counts = list(counts) + list(leftovers)
        return {syll: i for i, syll in enumerate(counts)}
    
    usage_map = get_usage_map()
    for file in files:
        df = pd.read_parquet(file)
        df.columns = [usage_map[syll] for syll in df.columns]
        df.to_parquet(file.with_name(file.name.replace("raw", "relabeled")))
    """
}

workflow {
    out = organize_extractions(params.moseq_folder)
    out = apply_pca(out)
    out = apply_moseq_model(out)
    // experiment_path = get_experiment_names("").map { it.readLines() }
    experiment_path = get_experiment_names(out).map { it.readLines() }
        .flatten()
        .filter { it != "" && it != null && it != "\n" }
    dfs = create_dataframe(experiment_path)
    dfs = create_usage_dataframe(dfs)
    relabel_dataframe(dfs.collect())
}
