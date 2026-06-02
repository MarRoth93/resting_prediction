# REST Motion Regression

This branch adds optional motion nuisance regression for REST preprocessing.

## NSD Motion Files

NSD documents motion estimates under:

```text
nsddata_timeseries/ppdata/subjAA/func*/motion/motion_BB_runCC.tsv
```

The files contain 6 columns per volume:

- 3 translations in mm
- 3 rotations in radians

The number of rows should match the corresponding preprocessed time-series run.

## Enable Friston-24

In `config.yaml`, set:

```yaml
rest_preprocessing:
  nuisance_regression:
    enabled: true
    motion_model: "friston24"
    standardize: true
    require_motion: true
```

Supported `motion_model` values:

- `none`
- `motion6`
- `motion12`
- `friston24`

`friston24` is built from:

```text
6 motion parameters
+ 6 temporal derivatives
+ 6 squared motion parameters
+ 6 squared derivatives
```

If `require_motion: true`, preprocessing fails when the matching motion file cannot be found. If `false`, the run is processed without motion regression and logs a warning.

Motion censoring can stay enabled at the same time:

```yaml
rest_preprocessing:
  motion_censoring:
    enabled: true
    fd_threshold_mm: 0.5
    max_censored_fraction: 0.3
```

The code loads motion parameters once per REST run and uses them for both FD censoring and nuisance regression when enabled.
