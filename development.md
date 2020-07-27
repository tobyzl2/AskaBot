# Development Log

## 07.19.2020
- Completed exploratory notebooks for json to csv conversion and wikipedia article matching.

## 07.26.2020
- Baseline keyword matching implementations are finished and trained over 5 epochs on data generated using SQUADv2.0 and wikipedia matcher API.

Scores are as follows:
- accuracy: 0.983
- f1: 0.764
- precision: 0.838
- recall: 0.701

## Future Development
- Improvements on keyword matching, especially for identifying more than one keyword.
- Implement spell-checking/auto-correct for question input
- Implement bot to ask yes/no clarification questions
