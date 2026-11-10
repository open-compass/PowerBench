# Fault Report

## 1. Basic Information

- **Device**: Jinping Converter Station Pole-1 ±800kV High-End Converter Transformer Phase-A Unit-1 Primary
- **Discovery time**: 2024-01-08 12:00:00
- **Linked rule**: K2
- **Fault window**: 2024-01-08 10:00:00 ~ 2024-01-08 12:00:00
- **Input metric field**: winding_hotspot_temp
- **Injected field**: winding_hotspot_temp
- **Scenario**: Fault injection succeeded (density filled)
- **Report ID**: FB-2024-002
- **Station**: Jinping Converter Station
- **Discoverer**: Operator A
- **Crew**: Maintenance Team 1
- **Weather**: Clear
- **Load level**: 85% of rated

## 2. Event Summary

During 2024-01-08 10:00:00 ~ 2024-01-08 12:00:00, winding hotspot temperature crossed the trip threshold at 2024-01-08 12:00:00, triggering rule K2.

- **Outage scope**: Load served by Pole-1 ±800kV HECT Phase-A Unit-1
- **User impact**: HVDC transmission users
- **System risk**: Possible single-pole block / reduced transfer capacity

## 3. Alarm Record

| Timestamp | Alarm | Source | Result | Metric | Value |
| --- | --- | --- | --- | --- | --- |
| 2024-01-08 12:00:00 | Winding hotspot temperature abnormal (alarm 120°C / trip 140°C) | SCADA / online monitoring | Alarm | winding_hotspot_temp | (see telemetry) |

### Rule detail at 2024-01-08 12:00:00

- **Object**: winding hotspot temperature
- **Description**: Abnormal if above alarm for 3 consecutive hours, or a single point above trip threshold.
- **Alarm threshold (°C)**: 120
- **Trip threshold (°C)**: 140

## 4. Cause Analysis (illustrative)

- **Direct cause**: Single-point trip-threshold exceedance of winding hotspot temperature.
- **Contributing factors**: Possible cooling degradation or internal thermal fault; monitoring/response lag.

This English report is a **review-stage sample** aligned with the injected telemetry episode for the demo device.
