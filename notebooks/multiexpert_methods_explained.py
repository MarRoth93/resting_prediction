# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.23,<0.24",
#     "matplotlib>=3.7",
#     "numpy>=1.24",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Two ways to build a shared brain language

    This notebook uses **small synthetic data** to make the central ideas
    behind Hybrid-CHA and connectivity-SRM visible.

    - The coloured matrices are imaginary resting-state connectivity maps.
    - The dots are imaginary responses to shared images.
    - The two-dimensional space stands in for the pipeline's 100-dimensional
      shared space.

    The examples preserve the logic of the implemented methods, but they are
    intentionally simplified and are **not a scientific benchmark**.
    """)
    return


@app.cell
def _(mo):
    noise_control = mo.ui.slider(
        start=0.02,
        stop=0.30,
        step=0.01,
        value=0.10,
        label="Synthetic measurement noise",
        show_value=True,
    )
    subject_control = mo.ui.slider(
        start=1,
        stop=4,
        step=1,
        value=1,
        label="Subject to inspect",
        show_value=True,
    )
    seed_control = mo.ui.slider(
        start=1,
        stop=20,
        step=1,
        value=7,
        label="Synthetic dataset",
        show_value=True,
    )
    availability_control = mo.ui.dropdown(
        options=[
            "Both experts",
            "Hybrid-CHA only",
            "connectivity-SRM only",
        ],
        value="Both experts",
        label="Experts available to fusion",
    )
    mo.hstack(
        [
            noise_control,
            subject_control,
            seed_control,
            availability_control,
        ],
        justify="space-around",
        gap=2,
    )
    return availability_control, noise_control, seed_control, subject_control


@app.cell
def _(np):
    def rotation_matrix(angle):
        return np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ],
            dtype=np.float64,
        )

    def orthogonal_factor(matrix):
        left, _, right_t = np.linalg.svd(matrix, full_matrices=False)
        return left @ right_t

    def procrustes_rotation(source, target):
        source_centered = source - source.mean(axis=0, keepdims=True)
        target_centered = target - target.mean(axis=0, keepdims=True)
        return orthogonal_factor(source_centered.T @ target_centered)

    def normalized_columns(matrix):
        scale = np.linalg.norm(matrix, axis=0, keepdims=True)
        return matrix / np.maximum(scale, 1e-12)

    def make_synthetic_dataset(seed, noise):
        rng = np.random.default_rng(int(seed))
        n_subjects = 4
        n_landmarks = 18
        n_voxels = 24
        n_task_samples = 60
        latent_dim = 2

        landmark_angle = np.linspace(0, 2 * np.pi, n_landmarks, endpoint=False)
        shared_rest = np.column_stack(
            [
                1.2 * np.cos(landmark_angle) + 0.25 * np.cos(3 * landmark_angle),
                0.9 * np.sin(landmark_angle) + 0.20 * np.sin(2 * landmark_angle),
            ]
        )
        shared_rest = normalized_columns(shared_rest) * np.sqrt(n_landmarks)

        task_angle = np.linspace(0, 2 * np.pi, n_task_samples, endpoint=False)
        shared_task = np.column_stack(
            [
                np.cos(task_angle) + 0.30 * np.sin(3 * task_angle),
                np.sin(task_angle) + 0.20 * np.cos(2 * task_angle),
            ]
        )

        connectivity = {}
        task_responses = {}
        true_subject_maps = {}
        for subject_index in range(n_subjects):
            raw_basis = rng.normal(size=(n_voxels, latent_dim))
            subject_basis, _ = np.linalg.qr(raw_basis)
            subject_basis = subject_basis[:, :latent_dim]
            true_subject_maps[subject_index] = subject_basis

            connectivity[subject_index] = (
                shared_rest @ subject_basis.T
                + float(noise) * rng.normal(size=(n_landmarks, n_voxels))
            )
            task_responses[subject_index] = (
                shared_task @ subject_basis.T
                + 0.55
                * float(noise)
                * rng.normal(size=(n_task_samples, n_voxels))
            )

        return {
            "connectivity": connectivity,
            "task_responses": task_responses,
            "shared_rest": shared_rest,
            "shared_task": shared_task,
            "true_subject_maps": true_subject_maps,
            "n_subjects": n_subjects,
            "n_landmarks": n_landmarks,
            "n_voxels": n_voxels,
            "n_task_samples": n_task_samples,
            "latent_dim": latent_dim,
        }

    def fit_simplified_hybrid_cha(dataset, seed):
        rng = np.random.default_rng(int(seed) + 1000)
        personal_maps = {}
        projected_task = {}

        # Each SVD finds the correct personal subspace, but its internal axes
        # can be rotated without changing that subspace. The synthetic twist
        # makes this harmless ambiguity visible.
        coordinate_twists = np.linspace(-0.75, 0.75, dataset["n_subjects"])
        coordinate_twists += rng.normal(scale=0.06, size=dataset["n_subjects"])
        for subject_index in range(dataset["n_subjects"]):
            _, _, right_t = np.linalg.svd(
                dataset["connectivity"][subject_index],
                full_matrices=False,
            )
            personal_map = right_t[: dataset["latent_dim"]].T
            personal_map = personal_map @ rotation_matrix(
                float(coordinate_twists[subject_index])
            )
            personal_maps[subject_index] = personal_map
            projected_task[subject_index] = (
                dataset["task_responses"][subject_index] @ personal_map
            )

        initial_template = np.mean(list(projected_task.values()), axis=0)
        template = initial_template.copy()
        rotations = {}
        aligned_task = {}
        history = []
        for _iteration in range(12):
            for subject_index in range(dataset["n_subjects"]):
                rotations[subject_index] = procrustes_rotation(
                    projected_task[subject_index],
                    template,
                )
                aligned_task[subject_index] = (
                    projected_task[subject_index] @ rotations[subject_index]
                )
            new_template = np.mean(list(aligned_task.values()), axis=0)
            change = np.linalg.norm(new_template - template) / max(
                np.linalg.norm(template),
                1e-12,
            )
            history.append(float(change))
            template = new_template

        before_error = np.mean(
            [
                np.linalg.norm(values - initial_template) / np.sqrt(values.size)
                for values in projected_task.values()
            ]
        )
        after_error = np.mean(
            [
                np.linalg.norm(values - template) / np.sqrt(values.size)
                for values in aligned_task.values()
            ]
        )
        effective_maps = {
            subject_index: personal_maps[subject_index] @ rotations[subject_index]
            for subject_index in range(dataset["n_subjects"])
        }
        return {
            "personal_maps": personal_maps,
            "effective_maps": effective_maps,
            "projected_task": projected_task,
            "aligned_task": aligned_task,
            "initial_template": initial_template,
            "template": template,
            "history": history,
            "before_error": float(before_error),
            "after_error": float(after_error),
        }

    def fit_simplified_csrm(dataset):
        connectivity = dataset["connectivity"]
        weights = {}
        fingerprints = {}
        for subject_index in range(dataset["n_subjects"]):
            _, _, right_t = np.linalg.svd(
                connectivity[subject_index],
                full_matrices=False,
            )
            weights[subject_index] = right_t[: dataset["latent_dim"]].T
            fingerprints[subject_index] = (
                connectivity[subject_index] @ weights[subject_index]
            )

        reference = fingerprints[0]
        for subject_index in range(1, dataset["n_subjects"]):
            orient = orthogonal_factor(
                fingerprints[subject_index].T @ reference
            )
            weights[subject_index] = weights[subject_index] @ orient

        shared = np.mean(
            [
                connectivity[subject_index] @ weights[subject_index]
                for subject_index in range(dataset["n_subjects"])
            ],
            axis=0,
        )
        objective_history = []
        for _iteration in range(20):
            for subject_index in range(dataset["n_subjects"]):
                weights[subject_index] = orthogonal_factor(
                    connectivity[subject_index].T @ shared
                )
            shared = np.mean(
                [
                    connectivity[subject_index] @ weights[subject_index]
                    for subject_index in range(dataset["n_subjects"])
                ],
                axis=0,
            )
            relative_errors = []
            for subject_index in range(dataset["n_subjects"]):
                reconstruction = shared @ weights[subject_index].T
                relative_errors.append(
                    np.linalg.norm(connectivity[subject_index] - reconstruction)
                    / np.linalg.norm(connectivity[subject_index])
                )
            objective_history.append(float(np.mean(relative_errors)))

        fitted_fingerprints = {
            subject_index: connectivity[subject_index] @ weights[subject_index]
            for subject_index in range(dataset["n_subjects"])
        }
        return {
            "shared": shared,
            "weights": weights,
            "fitted_fingerprints": fitted_fingerprints,
            "objective_history": objective_history,
        }

    def make_fusion_example(dataset, seed, noise):
        rng = np.random.default_rng(int(seed) + 2000)
        truth = dataset["task_responses"][0]
        n_voxels = truth.shape[1]
        split = n_voxels // 2
        region_indices = [np.arange(0, split), np.arange(split, n_voxels)]

        cha_noise_scale = np.concatenate(
            [np.full(split, 0.35), np.full(n_voxels - split, 0.90)]
        )
        srm_noise_scale = np.concatenate(
            [np.full(split, 0.85), np.full(n_voxels - split, 0.30)]
        )
        base_noise = 0.20 + 2.3 * float(noise)
        cha_prediction = truth + base_noise * rng.normal(size=truth.shape) * cha_noise_scale
        srm_prediction = truth + base_noise * rng.normal(size=truth.shape) * srm_noise_scale

        train_rows = np.arange(0, truth.shape[0] // 2)
        learned_cha_weights = []
        for voxels in region_indices:
            difference = cha_prediction[np.ix_(train_rows, voxels)] - srm_prediction[
                np.ix_(train_rows, voxels)
            ]
            target_from_srm = truth[np.ix_(train_rows, voxels)] - srm_prediction[
                np.ix_(train_rows, voxels)
            ]
            numerator = np.sum(difference * target_from_srm)
            denominator = max(float(np.sum(difference * difference)), 1e-12)
            learned_cha_weights.append(float(np.clip(numerator / denominator, 0.0, 1.0)))

        learned_prediction = np.empty_like(truth)
        equal_prediction = 0.5 * (cha_prediction + srm_prediction)
        for region_index, voxels in enumerate(region_indices):
            cha_weight = learned_cha_weights[region_index]
            learned_prediction[:, voxels] = (
                cha_weight * cha_prediction[:, voxels]
                + (1.0 - cha_weight) * srm_prediction[:, voxels]
            )

        def rmse(values):
            return float(np.sqrt(np.mean((truth - values) ** 2)))

        return {
            "truth": truth,
            "cha_prediction": cha_prediction,
            "srm_prediction": srm_prediction,
            "equal_prediction": equal_prediction,
            "learned_prediction": learned_prediction,
            "cha_weights": np.asarray(learned_cha_weights),
            "errors": {
                "Hybrid-CHA": rmse(cha_prediction),
                "connectivity-SRM": rmse(srm_prediction),
                "Equal average": rmse(equal_prediction),
                "Learned fusion": rmse(learned_prediction),
            },
            "region_indices": region_indices,
        }

    return (
        fit_simplified_csrm,
        fit_simplified_hybrid_cha,
        make_fusion_example,
        make_synthetic_dataset,
    )


@app.cell
def _(make_synthetic_dataset, noise_control, seed_control):
    synthetic_data = make_synthetic_dataset(
        seed=seed_control.value,
        noise=noise_control.value,
    )
    return (synthetic_data,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The shared starting point: resting-state connectivity

    Each row below is a reference brain landmark. Each column is a voxel in
    the synthetic visual brain. A colour shows how strongly they move
    together during rest.

    Both methods receive this same kind of matrix. Their difference is what
    they do with it next.
    """)
    return


@app.cell
def _(plt, subject_control, synthetic_data):
    inspected_subject = int(subject_control.value) - 1
    connectivity_figure, connectivity_axis = plt.subplots(figsize=(11, 3.8))
    connectivity_image = connectivity_axis.imshow(
        synthetic_data["connectivity"][inspected_subject],
        aspect="auto",
        cmap="coolwarm",
    )
    connectivity_axis.set_title(
        f"Synthetic resting-state connectivity — subject {inspected_subject + 1}"
    )
    connectivity_axis.set_xlabel("Visual-brain voxels")
    connectivity_axis.set_ylabel("Reference landmarks")
    connectivity_figure.colorbar(
        connectivity_image,
        ax=connectivity_axis,
        label="Moves together ← 0 → moves oppositely",
    )
    connectivity_figure.tight_layout()
    connectivity_figure
    return (inspected_subject,)


@app.cell
def _(fit_simplified_hybrid_cha, seed_control, synthetic_data):
    hybrid_result = fit_simplified_hybrid_cha(
        synthetic_data,
        seed=seed_control.value,
    )
    return (hybrid_result,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Hybrid-CHA: personal map first, task-guided orientation second

    1. Each subject gets a compact personal map from REST connectivity.
    2. The axes of different personal maps can point in different directions.
    3. Responses to the same images reveal how those axes should be rotated.
    4. After rotation, the same image should occupy a similar place for each
       subject.

    The lines below connect matching synthetic images. Shorter lines mean
    the selected subject and the group template agree more closely.
    """)
    return


@app.cell
def _(hybrid_result, inspected_subject, plt):
    hybrid_figure, hybrid_axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sample_rows = range(0, 60, 4)

    before_points = hybrid_result["projected_task"][inspected_subject]
    before_template = hybrid_result["initial_template"]
    hybrid_axes[0].scatter(
        before_template[:, 0],
        before_template[:, 1],
        s=22,
        alpha=0.35,
        label="Group template",
        color="#606060",
    )
    hybrid_axes[0].scatter(
        before_points[:, 0],
        before_points[:, 1],
        s=22,
        alpha=0.65,
        label=f"Subject {inspected_subject + 1}",
        color="#d95f02",
    )
    for _row in sample_rows:
        hybrid_axes[0].plot(
            [before_template[_row, 0], before_points[_row, 0]],
            [before_template[_row, 1], before_points[_row, 1]],
            color="#bdbdbd",
            linewidth=0.8,
        )
    hybrid_axes[0].set_title("Before task-guided rotation")
    hybrid_axes[0].legend(loc="best")

    after_points = hybrid_result["aligned_task"][inspected_subject]
    after_template = hybrid_result["template"]
    hybrid_axes[1].scatter(
        after_template[:, 0],
        after_template[:, 1],
        s=22,
        alpha=0.35,
        label="Group template",
        color="#606060",
    )
    hybrid_axes[1].scatter(
        after_points[:, 0],
        after_points[:, 1],
        s=22,
        alpha=0.65,
        label=f"Subject {inspected_subject + 1}",
        color="#1b9e77",
    )
    for _row in sample_rows:
        hybrid_axes[1].plot(
            [after_template[_row, 0], after_points[_row, 0]],
            [after_template[_row, 1], after_points[_row, 1]],
            color="#bdbdbd",
            linewidth=0.8,
        )
    hybrid_axes[1].set_title("After task-guided rotation")
    hybrid_axes[1].legend(loc="best")

    for hybrid_axis in hybrid_axes:
        hybrid_axis.set_xlabel("Shared direction 1")
        hybrid_axis.set_ylabel("Shared direction 2")
        hybrid_axis.axhline(0, color="#dddddd", linewidth=0.7)
        hybrid_axis.axvline(0, color="#dddddd", linewidth=0.7)
        hybrid_axis.set_aspect("equal", adjustable="datalim")
    hybrid_figure.tight_layout()
    hybrid_figure
    return


@app.cell
def _(hybrid_result, mo):
    hybrid_improvement = 100.0 * (
        1.0 - hybrid_result["after_error"] / hybrid_result["before_error"]
    )
    mo.callout(
        mo.md(
            f"""
            **What changed?** Average mismatch fell from
            `{hybrid_result['before_error']:.3f}` to
            `{hybrid_result['after_error']:.3f}` — a synthetic improvement of
            **{hybrid_improvement:.1f}%**.

            Hybrid-CHA uses shared task responses during training to decide how
            each personal REST-derived map should be oriented.
            """
        ),
        kind="success",
    )
    return


@app.cell
def _(fit_simplified_csrm, synthetic_data):
    csrm_result = fit_simplified_csrm(synthetic_data)
    return (csrm_result,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Connectivity-SRM: common REST pattern and personal maps together

    Connectivity-SRM does not use shared image responses to create its main
    shared space. It alternates between two questions:

    1. Given the current group pattern, how should each subject map onto it?
    2. Given the current subject maps, what should the group pattern be?

    Repeating those steps makes the group pattern and personal maps agree.
    In the left panel, the selected subject's fitted landmark pattern should
    overlap the learned shared pattern. The right panel shows the fitting
    error falling over iterations.
    """)
    return


@app.cell
def _(csrm_result, inspected_subject, np, plt):
    csrm_figure, csrm_axes = plt.subplots(1, 2, figsize=(13, 4.8))
    shared_pattern = csrm_result["shared"]
    subject_pattern = csrm_result["fitted_fingerprints"][inspected_subject]
    csrm_axes[0].scatter(
        shared_pattern[:, 0],
        shared_pattern[:, 1],
        s=75,
        marker="o",
        alpha=0.55,
        color="#7570b3",
        label="Learned group pattern",
    )
    csrm_axes[0].scatter(
        subject_pattern[:, 0],
        subject_pattern[:, 1],
        s=45,
        marker="x",
        color="#e7298a",
        label=f"Subject {inspected_subject + 1} fitted to group",
    )
    for _row in range(shared_pattern.shape[0]):
        csrm_axes[0].plot(
            [shared_pattern[_row, 0], subject_pattern[_row, 0]],
            [shared_pattern[_row, 1], subject_pattern[_row, 1]],
            color="#cccccc",
            linewidth=0.7,
        )
    csrm_axes[0].set_title("Shared REST pattern and one subject's fit")
    csrm_axes[0].set_xlabel("Shared direction 1")
    csrm_axes[0].set_ylabel("Shared direction 2")
    csrm_axes[0].legend(loc="best")
    csrm_axes[0].set_aspect("equal", adjustable="datalim")

    objective_values = np.asarray(csrm_result["objective_history"])
    csrm_axes[1].plot(
        np.arange(1, objective_values.size + 1),
        objective_values,
        marker="o",
        markersize=4,
        color="#7570b3",
    )
    csrm_axes[1].set_title("Alternating group/person updates")
    csrm_axes[1].set_xlabel("Update round")
    csrm_axes[1].set_ylabel("Unexplained connectivity (lower is better)")
    csrm_axes[1].grid(alpha=0.25)
    csrm_figure.tight_layout()
    csrm_figure
    return


@app.cell
def _(csrm_result, mo):
    csrm_start = csrm_result["objective_history"][0]
    csrm_end = csrm_result["objective_history"][-1]
    mo.callout(
        mo.md(
            f"""
            **What changed?** The unexplained synthetic REST connectivity fell
            from `{csrm_start:.3f}` to `{csrm_end:.3f}`.

            Connectivity-SRM learns one shared REST pattern and each person's
            map to that pattern at the same time. Task responses are not needed
            for this main fitting step.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. What is the essential difference?

    | Hybrid-CHA | connectivity-SRM |
    |---|---|
    | Builds a compact personal map first | Learns the group pattern and personal maps together |
    | Uses shared task responses to orient training subjects | Learns its main shared structure from REST alone |
    | Asks: “How should this personal map be rotated?” | Asks: “What common REST pattern explains everyone?” |
    | Strong candidate for person-specific task orientation | Strong candidate for stable group-level structure |

    These are different assumptions about how to find a shared language.
    Neither one is guaranteed to be better everywhere.
    """)
    return


@app.cell
def _(make_fusion_example, noise_control, seed_control, synthetic_data):
    fusion_example = make_fusion_example(
        synthetic_data,
        seed=seed_control.value,
        noise=noise_control.value,
    )
    return (fusion_example,)


@app.cell
def _(availability_control, fusion_example, np):
    if availability_control.value == "Both experts":
        active_prediction = fusion_example["learned_prediction"]
        active_label = "Learned regional fusion"
    elif availability_control.value == "Hybrid-CHA only":
        active_prediction = fusion_example["cha_prediction"]
        active_label = "Hybrid-CHA fallback"
    else:
        active_prediction = fusion_example["srm_prediction"]
        active_label = "connectivity-SRM fallback"
    active_error = float(
        np.sqrt(np.mean((fusion_example["truth"] - active_prediction) ** 2))
    )
    return active_error, active_label, active_prediction


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Fusion: learn which expert is useful in each region

    The example below deliberately makes Hybrid-CHA cleaner in synthetic
    region A and connectivity-SRM cleaner in synthetic region B.

    The fusion rule sees training examples, measures which mixture best
    matches the known response in each region, and then applies those
    learned weights to later examples. The real pipeline learns this rule
    with a neural fusion module rather than the tiny calculation used here.

    Use the **Experts available to fusion** menu above to imitate method
    dropout: with both experts present, use the learned mixture; with one
    missing, fall back to the surviving expert.
    """)
    return


@app.cell
def _(active_label, active_prediction, fusion_example, np, plt):
    fusion_figure, fusion_axes = plt.subplots(1, 3, figsize=(16, 4.6))

    error_names = list(fusion_example["errors"])
    error_values = [fusion_example["errors"][name] for name in error_names]
    error_colours = ["#d95f02", "#7570b3", "#999999", "#1b9e77"]
    fusion_axes[0].bar(
        np.arange(len(error_names)),
        error_values,
        color=error_colours,
    )
    fusion_axes[0].set_xticks(np.arange(len(error_names)))
    fusion_axes[0].set_xticklabels(error_names, rotation=25, ha="right")
    fusion_axes[0].set_ylabel("Prediction error (lower is better)")
    fusion_axes[0].set_title("Which prediction is closest?")
    fusion_axes[0].grid(axis="y", alpha=0.25)

    region_names = ["Region A", "Region B"]
    cha_weights = fusion_example["cha_weights"]
    srm_weights = 1.0 - cha_weights
    fusion_axes[1].bar(region_names, cha_weights, label="Hybrid-CHA", color="#d95f02")
    fusion_axes[1].bar(
        region_names,
        srm_weights,
        bottom=cha_weights,
        label="connectivity-SRM",
        color="#7570b3",
    )
    fusion_axes[1].set_ylim(0, 1)
    fusion_axes[1].set_ylabel("Learned share of final prediction")
    fusion_axes[1].set_title("Different regions prefer different experts")
    fusion_axes[1].legend(loc="upper center")

    example_row = 8
    voxel_axis = np.arange(fusion_example["truth"].shape[1])
    fusion_axes[2].plot(
        voxel_axis,
        fusion_example["truth"][example_row],
        color="#222222",
        linewidth=2.5,
        label="True response",
    )
    fusion_axes[2].plot(
        voxel_axis,
        fusion_example["cha_prediction"][example_row],
        color="#d95f02",
        alpha=0.45,
        label="Hybrid-CHA",
    )
    fusion_axes[2].plot(
        voxel_axis,
        fusion_example["srm_prediction"][example_row],
        color="#7570b3",
        alpha=0.45,
        label="connectivity-SRM",
    )
    fusion_axes[2].plot(
        voxel_axis,
        active_prediction[example_row],
        color="#1b9e77",
        linewidth=2,
        linestyle="--",
        label=active_label,
    )
    fusion_axes[2].axvline(
        fusion_example["region_indices"][0][-1] + 0.5,
        color="#aaaaaa",
        linestyle=":",
    )
    fusion_axes[2].set_xlabel("Synthetic voxels: region A | region B")
    fusion_axes[2].set_ylabel("Response")
    fusion_axes[2].set_title("One synthetic image")
    fusion_axes[2].legend(loc="best", fontsize=8)
    fusion_figure.tight_layout()
    fusion_figure
    return


@app.cell
def _(active_error, active_label, fusion_example, mo):
    cha_region_a, cha_region_b = fusion_example["cha_weights"]
    mo.callout(
        mo.md(
            f"""
            **Current result: {active_label}** — synthetic prediction error
            `{active_error:.3f}`.

            With both experts available, the learned Hybrid-CHA share is
            **{cha_region_a:.0%} in region A** and **{cha_region_b:.0%} in
            region B**. The remaining share comes from connectivity-SRM.

            This is the intended effect of fusion: preserve complementary
            details when they exist, while keeping a valid fallback when one
            expert is absent. Whether it helps on real NSD data is decided by
            the LOSO promotion gate, not by this demonstration.
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Take-home picture

    ```text
    same REST connectivity
             │
             ├── Hybrid-CHA: personal map → task-guided rotation
             │                                  │
             └── cSRM: shared REST pattern ↔ personal map
                                                │
                              two voxel predictions
                                                │
                             learned regional mixture
                                                │
                                  final brain prediction
    ```

    - Hybrid-CHA contributes a task-oriented translation of personal REST structure.
    - Connectivity-SRM contributes a group-oriented structure learned from REST.
    - Fusion can exploit the difference only when the experts make meaningfully
      complementary errors.
    """)
    return


if __name__ == "__main__":
    app.run()
