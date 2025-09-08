# VanillaSteel Case Study Pipeline


## Prerequisites
Required Python packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

## Pipeline Execution
Run the complete pipeline:
```bash
python pipeline.py
```
Note: the execution could take many minutes since mulitiple experiments will run. On my machine it took around 10 min in total.

## Pipeline Steps

### Task 1: Data Merging
1. Merges supplier data from two Excel files
2. Normalizes and cleans the merged data
3. Output: `task_1_output/merged_supplier_data.csv`

### Task 2: RFQ Analysis
1. Reference Property Processing
   - Processes reference properties
   - Output: `task_2_output/reference_properties_split.tsv`

2. RFQ Enrichment
   - Enriches RFQ data with reference properties
   - Output: `task_2_output/rfq_enriched.csv`

3. Similarity Analysis
### Step 3: Similarity Analysis
- Experiment Results:
  - `top3_balanced.csv`: Equal weight to all features
  - `top3_dimension_focus.csv`: Higher weight on dimensions
  - `top3_grade_focus.csv`: Higher weight on grade properties
  - `top3_categorical_focus.csv`: Higher weight on categorical features
- Visualizations:
  - `similarity_analysis.png`: Comparison of different weighting strategies
  - `cluster_analysis.png`: Clustering visualization
- Additional Output:
  - `rfq_clusters.csv`: Cluster assignments for each RFQ

## Output Files
Each step generates specific output files in their respective output directories.
Check the console output for the complete list of generated files.

## Error Handling
- The pipeline stops on any script error
- Error messages are displayed in the console
- Each script runs in its correct working directory

## Note
Ensure all input files are present in their respective directories before running the pipeline.



## Experiment notes and insights

The outputs of my original experiments can be found in original_file_outputs. They should be the same as the files generated after the pipeline has run and are included in the case that there are issues executing the pipeline. 

Here are my insights, notes and unanswered questions from my experiments:

The process took a few minutes to compute, think of ways to optimize if speed is a consideration. Here removing sparse features could help. Including a runtime analysis could also be helpful.

Todo next: 
Find ways to validate the results like using domain knowledge. Without validation, it is hard to know if the similarity scores are correct.

For validation: analyze the top3 outputs across different weighting schemes. If they are consistent, it could imply that the results are robust.

Unanswered questions:
- How to validate with no labels or ground truth? Is it safe to assume that if the similarity scores are on average higher that the results are more accurate?
- I should analyze the top3 and see if order has a significant effect on which get selected? If I always take the first three, and more than 3 have the same score, how do I make sure that there is no bias in that selection?

Insights from similarity analysis of top3 output:
Regardless of weighting, the distribution of scores is very similar. This could imply that the reuslts are robust.
The cluster analysis shows relatively high similarity score within clusters. This could imply that the similarity feature design is robust.


