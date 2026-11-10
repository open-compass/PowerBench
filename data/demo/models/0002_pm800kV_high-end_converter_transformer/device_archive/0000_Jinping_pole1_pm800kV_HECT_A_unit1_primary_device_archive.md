# Device Archive

> Asset registry / design and O&M notes (not a single-incident record). Facts are taken from this model’s metadata and the device `per_device` annotations.

## 1. Basic Information

| Item | Content |
| --- | --- |
| Archive ID | SB-DA-0002_pm800kV_high-end_converter_transformer-0000 |
| Device ID | 0002_pm800kV_high-end_converter_transformer-0000 |
| Device name | Jinping Converter Station Pole-1 ±800kV High-End Converter Transformer Phase-A Unit-1 Primary |
| Model / type | ±800kV high-end converter transformer |
| Equipment level | Primary equipment (transformer class) |
| Device type | ±800kV high-end converter transformer |
| Insulation type | Oil-immersed |
| Station / line | Jinping Converter Station |
| Installation location | Jinping Converter Station Pole-1 ±800kV High-End Converter Transformer Phase-A Unit-1 Primary |
| Commissioning date | 2023-12-31 |
| Role | Primary unit |
| Rated voltage | 500kV |
| Insulating medium | Transformer oil |
| Structure | Single-phase converter transformer (Y/Y) |
| Primary/standby | Primary |
| Grid-side voltage | 500kV |
| Tap-changing method | On-load tap changer (OLTC) |

## 2. Metric Dictionary (field names)

English **database field names** are the telemetry CSV columns. Chinese display names in `metric_catalog.json` are retained for compatibility with the generation / rule pipeline (see README note).

| DB field | Unit | Kind |
| --- | --- | --- |
| bushing_pressure | kPa | raw |
| conservator_pressure | kPa | raw |
| transformer_body_pressure | kPa | raw |
| cooling_system_status | — | discrete |
| core_ground_current | mA | raw |
| oil_dga_c2h2 / c2h4 / c2h6 / ch4 / co / co2 / h2 / total_hydrocarbon | μL/L | raw |
| oil_level | mm | raw |
| oltc_motor_status / oltc_overall_status / oltc_tap_position / oltc_operation_count | — | OLTC |
| pd_level | pC | raw |
| rel_change_tan | tanδ | raw |
| sf6_gas_pressure_density | MPa | raw |
| top_oil_temp | °C | raw |
| winding_hotspot_temp | °C | raw |
| dga_ratio_c2h2_c2h4, winding_hotspot_temperature_rise, oil_to_ambient_temperature_diff, pressure_diff_body_conservator, dga_ratio_co2_co | — | derived |

## 3. Manufacturing / Procurement (illustrative)

| Item | Content |
| --- | --- |
| Manufacturer | TBEA Shenyang Transformer Group |
| Factory serial | TB-SY-2023-800A-001 |
| Factory date | 2023-10-15 |
| Contract No. | JP-HZ-2022-001-01 |
| Incoming inspection | 2023-11-20 — accepted |
| Warranty | 36 months from commissioning |

## 4. Notes

This file is a **minimal English sample** of an operational document for the demo device instance. Full-corpus Chinese documents are withheld from this review-stage repository.
