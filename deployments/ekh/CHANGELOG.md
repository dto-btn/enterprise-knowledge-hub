## [0.1.1] - 2026-10-22

### Fixed

- `pgvector.initdb.scriptsConfigMap` now has a default (`ekh-initdb`) in `values.yaml`; it was only
  set in `values-local.yaml`, so any deployment enabling pgvector without overriding it rendered a
  ConfigMap with an empty name (`resource name may not be empty`). The template now also fails
  early with a clear message if the value is blanked out.

## [0.1.0] - 2026-01-20

### Added

- Initial charts

## Notes: 

For version updates do not forget to update:

 * pyproject.toml and deployments/charts/ekh/Chart.yaml (appVersion prop.)
 * and if modifying charts and or templates the `version` prop.
