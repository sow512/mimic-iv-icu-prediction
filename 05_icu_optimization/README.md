
# ICU Resource Optimization & Capacity Simulation

This folder contains simulation scripts for evaluating ICU bed availability, patient risk prioritization, and resource planning using predicted ICU-transfer probabilities.

## Included Scripts

| File | Description |
|---|---|
| `icu_crisis_capacity_simulation.py` | Simulates ICU demand surges, bed availability, waiting-time constraints, and risk-prioritized admissions. |
| `derive_icu_stay_lengths.py` | Extracts historical ICU stay durations from MIMIC-IV to support resource-planning analysis. |

## Methodology

- **Risk-Based Prioritization:** Patients are ranked according to predicted ICU-transfer probability.
- **Capacity Stress Testing:** Evaluates ICU demand under 1×–5× surge scenarios.
- **Bed Availability:** Considers occupied beds, unavailable beds, and hypothetical patient discharges.
- **Waiting-Time Analysis:** Tracks immediate admissions, delayed admissions, and unmet ICU requests.
- **Capacity Alerts:** Classifies simulated capacity risk into Green, Amber, and Red levels using illustrative thresholds.

## Objective

To investigate how machine learning predictions can support ICU capacity planning and identify potential resource shortages under varying demand conditions.

## Limitations

The simulations are hypothetical and do not represent validated clinical bed-allocation decisions. Predicted ICU-transfer risk is not equivalent to clinical urgency.

Historical MIMIC-IV data and patient-level records are not included in this repository.
