
def aging_env = "$HOME/miniconda3/envs/aging"
def rapids_env = "$HOME/miniconda3/envs/rapids-23.08"

params.model_folder = "/n/groups/datta/win/longtogeny/size_norm/models/bottleneck_param_scan_00"


process transform_data {
    label 'gpu'
    memory 30.GB
    time { 15.m * task.attempt }
    conda aging_env
    maxRetries 1

    input:
    val model_path

    output:
    stdout emit: file_name

    script:
    """
    #!/bin/env python

    from aging.size_norm.validation import transform_data
    from aging.organization.paths import ValidationPaths

    val_paths = ValidationPaths()
    transform_save_path = transform_data(val_paths.age_classifier, "${model_path}", data_source="age_classifier")
    print(str(transform_save_path))
    """
}

process transform_dynamics_data {
    label 'gpu'
    memory 30.GB
    time { 30.m * task.attempt }
    conda aging_env
    maxRetries 1

    input:
    val model_path

    output:
    stdout emit: file_name

    script:
    """
    #!/bin/env python

    from aging.size_norm.validation import transform_data
    from aging.organization.paths import ValidationPaths

    val_paths = ValidationPaths()

    transform_save_path = transform_data(val_paths.dynamics, "${model_path}", data_source="dynamics")
    print(str(transform_save_path))
    """
}

process age_classification {
    label 'gpu_quad'
    memory 25.GB
    time { 30.m * task.attempt }
    conda rapids_env
    maxRetries 3

    input:
    val transform_save_path

    output:
    val transform_save_path

    script:
    """
    #!/bin/env python

    from pathlib import Path
    from aging.size_norm.validation import age_classifiers, plot_classification, link_results_folder
    from aging.organization.paths import ValidationPaths, create_plot_path

    save_path = Path("${transform_save_path}")
    val_paths = ValidationPaths()

    out_folder = create_plot_path(val_paths.classifier_pipeline, "_classifier")

    results = age_classifiers(save_path, debug=False, thinning=4)
    plot_classification(results, out_folder, is_transformed=True)

    results = age_classifiers(val_paths.age_classifier, debug=False, thinning=4)
    plot_classification(results, out_folder, is_transformed=False)

    link_results_folder(out_folder, save_path.parents[1])
    """
}

process pose_manifold {
    label 'short'
    memory 30.GB
    time 10.m
    conda aging_env
    maxRetries 0

    input:
    val transform_save_path

    output:
    val transform_save_path

    script:
    """
    #!/bin/env python

    from pathlib import Path
    from aging.size_norm.validation import link_results_folder, pca_pose_manifold
    from aging.organization.paths import ValidationPaths, create_plot_path

    save_path = Path("${transform_save_path}")
    val_paths = ValidationPaths()

    out_folder = create_plot_path(val_paths.manifold_pipeline, "_pose_manifold")

    pca_pose_manifold(val_paths.age_classifier, save_path, out_folder)
    link_results_folder(out_folder, save_path.parents[1])
    """
}

process changepoints {
    label 'short'
    memory 20.GB
    time { 70.m * task.attempt }
    conda aging_env
    maxRetries 0

    input:
    val transform_save_path

    output:
    val transform_save_path

    script:
    """
    #!/bin/env python
    from pathlib import Path
    from aging.size_norm.validation import link_results_folder, compute_changepoints, plot_changepoints, plot_changepoint_correlations
    from aging.organization.paths import ValidationPaths, create_plot_path

    save_path = Path("${transform_save_path}")
    val_paths = ValidationPaths()
    out_folder = create_plot_path(val_paths.changepoints_pipeline, "_changepoints")

    pre_cps, post_cps = compute_changepoints(val_paths.dynamics, save_path)

    plot_changepoints(pre_cps, post_cps, out_folder)
    plot_changepoint_correlations(pre_cps, post_cps, out_folder)

    link_results_folder(out_folder, save_path.parents[1])
    """
}

process dynamics {
    label 'short'
    memory 25.GB
    time { 45.m * task.attempt }
    conda aging_env
    maxRetries 0

    input:
    val transform_save_path

    output:
    val transform_save_path

    script:
    """
    #!/bin/env python
    from pathlib import Path
    from aging.size_norm.validation import link_results_folder, compute_dynamics, plot_dynamics
    from aging.organization.paths import ValidationPaths, create_plot_path

    save_path = Path("${transform_save_path}")
    val_paths = ValidationPaths()
    out_folder = create_plot_path(val_paths.dynamics_pipeline, "_dynamics")

    scalars = compute_dynamics(val_paths.dynamics, save_path)
    plot_dynamics(scalars, out_folder)

    link_results_folder(out_folder, save_path.parents[1])
    """
}

process select_pose_visualization {
    label 'short'
    memory 15.GB
    time 10.m
    conda aging_env
    maxRetries 0

    input:
    val transform_save_path

    output:
    val transform_save_path

    script:
    """
    #!/bin/env python

    from pathlib import Path
    from aging.organization.paths import ValidationPaths, create_plot_path
    from aging.size_norm.validation import link_results_folder, plot_poses

    save_path = Path("${transform_save_path}")
    val_paths = ValidationPaths()

    out_folder = create_plot_path(val_paths.poses_pipeline, "_poses")

    plot_poses(val_paths.age_classifier, save_path, out_folder)

    link_results_folder(out_folder, save_path.parents[1])
    """
}

process size_predictions {
    label 'short'
    memory 15.GB
    time 6.m
    conda aging_env
    maxRetries 0

    input:
    val transform_save_path

    output:
    val transform_save_path

    script:
    """
    #!/bin/env python
    from pathlib import Path
    from aging.organization.paths import ValidationPaths, create_plot_path
    from aging.size_norm.validation import link_results_folder, mouse_size_predictions

    save_path = Path("${transform_save_path}")
    val_paths = ValidationPaths()
    out_folder = create_plot_path(val_paths.size_predictions_pipeline, "_size_predictions")

    mouse_size_predictions(val_paths.age_classifier, save_path, out_folder)

    link_results_folder(out_folder, save_path.parents[1])
    """
}

workflow {
    files = file("$params.model_folder/**/model.pt")
    if (files.size() == 0) {
        files = file("$params.model_folder/model.pt")
        ch = Channel.from(files)
    } else {
        ch = Channel.fromList(files)
    }
    received = transform_data(ch).view().map { it.trim() }
    dynamics_files = transform_dynamics_data(ch).map { it.trim() }

    age_classification(received)
    pose_manifold(received)
    select_pose_visualization(received)
    size_predictions(received)

    changepoints(dynamics_files)
    dynamics(dynamics_files)
}