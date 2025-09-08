import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

"""
Feature Vector Format: RFQ Similarity Model
------------------------------------------

Each RFQ is represented as a structured feature vector combining:

1. Dimensional Intervals
2. Categorical Product Metadata
3. Grade Properties (Numeric Midpoints)

1. Dimensional Intervals (`*_range`)
------------------------------------
Each dimension is represented as a (min, max) tuple:

    'thickness_range'         → (float, float)
    'width_range'             → (float, float)
    'length_range'            → (float, float)
    'height_range'            → (float, float)
    'weight_range'            → (float, float)
    'inner_diameter_range'    → (float, float)
    'outer_diameter_range'    → (float, float)

- Singleton values are encoded as (6.0, 6.0)
- Used for overlap metric comparison (IoU)

2. Categorical Metadata
------------------------
These are string fields compared using exact match (1.0 or 0.0):

    'grade'                     
    'grade_suffix'             
    'coating'                  
    'finish'                   
    'surface_type'             
    'surface_protection'       
    'form'                     
    'Standards'                
    'Hardness (HB, HV, HRC)'   
    'Impact toughness (Charpy V-notch)'
    'Source_Pages'             
    'Application'              
    'Category'                 

3. Grade Properties (`*_mid`)
------------------------------
Each is the midpoint of a numeric range, representing:

    'yield_strength_mid'           → MPa
    'tensile_strength_mid'         → MPa
    'Elongation (A%)_mid'          → %
    'Nb + V + Ti (Others)_mid'     → wt %
    'Carbon (C)_mid'               → wt %
    'Manganese (Mn)_mid'           → wt %
    'Silicon (Si)_mid'             → wt %
    'Sulfur (S)_mid'               → wt %
    'Phosphorus (P)_mid'           → wt %
    'Chromium (Cr)_mid'            → wt %
    'Nickel (Ni)_mid'              → wt %
    'Molybdenum (Mo)_mid'          → wt %
    'Vanadium (V)_mid'             → wt %
    'Tungsten (W)_mid'             → wt %
    'Cobalt (Co)_mid'              → wt %
    'Copper (Cu)_mid'              → wt %
    'Aluminum (Al)_mid'            → wt %
    'Titanium (Ti)_mid'            → wt %
    'Niobium (Nb)_mid'             → wt %
    'Boron (B)_mid'                → wt %
    'Nitrogen (N)_mid'             → wt %

- Compared using vector similarity (cosine similarity)
- NaNs or missing values handled during feature construction

Similarity Computation
-------------------------------
Weighted average of:
- Overlap (IoU) for dimension ranges
- Exact match for categorical features
- Cosine similarity for grade property vectors
"""

def cluster_and_visualize(feature_vectors, n_clusters=20):
    """Cluster RFQs and visualize results"""
    # Prepare feature matrix
    features = []
    ids = []
    for rfq_id, f in feature_vectors.items():
        # Extract numeric features
        dims = [v[0] for k, v in f.items() if k.endswith('_range')]
        grades = [v for k, v in f.items() if k.endswith('_mid')]
        features.append(dims + grades)
        ids.append(rfq_id)
    
    X = np.array(features)
    X = np.nan_to_num(X)  # Handle NaN values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Create visualizations
    plt.figure(figsize=(12, 5))
    
    # Cluster sizes
    plt.subplot(1, 2, 1)
    cluster_sizes = np.bincount(clusters)
    plt.bar(range(n_clusters), cluster_sizes)
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of RFQs')
    
    # Within-cluster similarities 
    # Used to validate if clusters are meaningful - a high within cluster score could indicate that similiarty scores are robust.
    plt.subplot(1, 2, 2)
    within_cluster_sims = []
    for i in range(n_clusters):
        mask = clusters == i
        cluster_ids = np.array(ids)[mask]
        sims = []
        for id1, id2 in combinations(cluster_ids, 2):
            sim = compute_similarity(
                feature_vectors[id1], 
                feature_vectors[id2], 
                weights={'dimensions': 0.33, 'categorical': 0.33, 'grade_properties': 0.34}
            )
            sims.append(sim)
        within_cluster_sims.append(np.mean(sims) if sims else 0)
    
    plt.bar(range(n_clusters), within_cluster_sims)
    plt.title('Average Within-Cluster Similarity')
    plt.xlabel('Cluster')
    plt.ylabel('Similarity Score')
    
    plt.tight_layout()
    plt.savefig('task_2_output/cluster_analysis.png')
    plt.close()
    
    return dict(zip(ids, clusters))

def experiment_weights():
    """Different weight configurations to try"""
    return [
        {
            'name': 'balanced',
            'weights': {'dimensions': 0.33, 'categorical': 0.33, 'grade_properties': 0.34}
        },
        {
            'name': 'dimension_focus',
            'weights': {'dimensions': 0.5, 'categorical': 0.25, 'grade_properties': 0.25}
        },
        {
            'name': 'grade_focus',
            'weights': {'dimensions': 0.25, 'categorical': 0.25, 'grade_properties': 0.5}
        },
        {
            'name': 'categorical_focus',
            'weights': {'dimensions': 0.25, 'categorical': 0.5, 'grade_properties': 0.25}
        }
    ]

def visualize_results(experiments_results):
    """Visualize similarity scores across experiments"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Average similarity scores
    plt.subplot(2, 2, 1)
    avg_scores = {exp: np.mean([r['similarity_score'] for r in results]) 
                 for exp, results in experiments_results.items()}
    plt.bar(avg_scores.keys(), avg_scores.values())
    plt.title('Average Similarity Scores by Experiment')
    plt.xticks(rotation=45)
    
    # Plot 2: Score distributions
    plt.subplot(2, 2, 2)
    for exp, results in experiments_results.items():
        sns.kdeplot(data=[r['similarity_score'] for r in results], label=exp)
    plt.title('Similarity Score Distributions')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('task_2_output/similarity_analysis.png')
    plt.close()

def compute_midpoint(min_val, max_val):
    """Calculate midpoint of a range, handling NaN values"""
    if pd.isna(min_val) and pd.isna(max_val):
        return np.nan
    if pd.isna(min_val): return max_val
    if pd.isna(max_val): return min_val
    return (min_val + max_val) / 2

def iou(min1, max1, min2, max2):
    """Calculate Intersection over Union for dimensional ranges"""
    if pd.isna(min1) or pd.isna(max1) or pd.isna(min2) or pd.isna(max2):
        return 0.0
    intersection = max(0, min(max1, max2) - max(min1, min2))
    union = max(max1, max2) - min(min1, min2)
    return intersection / union if union > 0 else 0.0

def compute_similarity(f1, f2, weights=None):
    """
    Compute weighted similarity between two RFQs
    Args:
        f1, f2: Feature dictionaries for two RFQs
        weights: Optional weights for different feature types
    """
    if weights is None:     # default weights
        weights = {
            'dimensions': 0.34,
            'categorical': 0.33,
            'grade_properties': 0.33
        }
    
    # Dimension similarity (IoU)
    # Concern with IoU similiarty is that if the dimensions are both 0, (i.e. both 0 width) this will 
    # result in a score of 0 even though it could be an indication that they are similar. 
    # When mean is caluculated, this could reduce overall score.
    dim_sims = [iou(*f1[k], *f2[k]) for k in f1 if k.endswith('_range')]
    # Find average across all dimensions for final weighted score
    dim_score = np.mean(dim_sims) if dim_sims else 0

    # Categorical similarity (exact match)

    cat_cols = ["grade",
        "grade_suffix",
        "coating",
        "finish",
        "surface_type",
        "surface_protection",
        "form",
        "Standards",
        "Hardness (HB, HV, HRC)",
        "Impact toughness (Charpy V-notch)",
        "Source_Pages",
        "Application",
        "Category"]
    
    # Exact match - true if exact match. 
    # If one or both null, not a match (TODO: Is this the best approach for if both are Null?).
    cat_sims = [f1[k] == f2[k] and pd.notna(f1[k]) for k in cat_cols]
    # Find average across all dimensions for final weighted score
    cat_score = np.mean(cat_sims) if cat_sims else 0

    # Grade properties similarity (cosine)
    # Using cosine similarity because we care about propotional similairty of the properties
    # Ex: [Carbon=0.2, Manganese=1.2, Yield=295] should be considerderd more similar to
    # [Carbon=0.22, Manganese=1.3, Yield=300] than [Carbon=0.4, Manganese=0.1, Yield=500]
    # TODO: compare with Euclidian distance, or Jaccard similairy 
    # Consideration: should I normalise the values first, I think cosine is sensitive to scale? 
    # Otherwise yeild strength could dominate results since not a percentage?
    
    grade_cols = [k for k in f1 if k.endswith('_mid')]
    g1 = np.array([f1[k] for k in grade_cols])
    g2 = np.array([f2[k] for k in grade_cols])
    
    if not np.isnan(g1).all() and not np.isnan(g2).all():
        g1 = np.nan_to_num(g1)
        g2 = np.nan_to_num(g2)
        grade_score = cosine_similarity([g1], [g2])[0][0]
    else:
        grade_score = 0

    # Weighted average
    final_score = (
        weights['dimensions'] * dim_score +
        weights['categorical'] * cat_score +
        weights['grade_properties'] * grade_score
    )
    
    return final_score

def main():
    # Load data
    rfq = pd.read_csv("task_2_output/rfq_enriched.csv")
    
    # Define feature columns
    dimension_cols = [
        ("thickness_min", "thickness_max"),
        ("width_min", "width_max"),
        ("length_min", "length_min"),       # length only has min so min = max
        ("height_min", "height_max"),
        ("weight_min", "weight_max"),
        ("inner_diameter_min", "inner_diameter_max"),
        ("outer_diameter_min", "outer_diameter_max"),
    ]

    # TODO: figure out which properties are too sparse and should be dropped
    grade_property_cols = [
        # Mechanical properties
        ("yield_strength_min", "yield_strength_max"),
        ("tensile_strength_min", "tensile_strength_max"),
        ("Elongation (A%)_min", "Elongation (A%)_max"),
        ("Nb + V + Ti (Others)_min", "Nb + V + Ti (Others)_max"),

        # Chemical composition
        ("Carbon (C)_min", "Carbon (C)_max"),
        ("Manganese (Mn)_min", "Manganese (Mn)_max"),
        ("Silicon (Si)_min", "Silicon (Si)_max"),
        ("Sulfur (S)_min", "Sulfur (S)_max"),
        ("Phosphorus (P)_min", "Phosphorus (P)_max"),
        ("Chromium (Cr)_min", "Chromium (Cr)_max"),
        ("Nickel (Ni)_min", "Nickel (Ni)_max"),
        ("Molybdenum (Mo)_min", "Molybdenum (Mo)_max"),
        ("Vanadium (V)_min", "Vanadium (V)_max"),
        ("Tungsten (W)_min", "Tungsten (W)_max"),
        ("Cobalt (Co)_min", "Cobalt (Co)_max"),
        ("Copper (Cu)_min", "Copper (Cu)_max"),
        ("Aluminum (Al)_min", "Aluminum (Al)_max"),
        ("Titanium (Ti)_min", "Titanium (Ti)_max"),
        ("Niobium (Nb)_min", "Niobium (Nb)_max"),
        ("Boron (B)_min", "Boron (B)_max"),
        ("Nitrogen (N)_min", "Nitrogen (N)_max"),
    ]

    # due to its variable scales, hardness is included in categorial features
    categorical_cols = [
        "grade",
        "grade_suffix",
        "coating",
        "finish",
        "surface_type",
        "surface_protection",
        "form",
        "Standards",
        "Hardness (HB, HV, HRC)",
        "Impact toughness (Charpy V-notch)",
        "Source_Pages",
        "Application",
        "Category"
    ]
    
    # Build feature vectors
    feature_vectors = {}
    for _, row in rfq.iterrows():
        fid = row['id']
        f = {}
        
        # Dimension intervals
        for min_col, max_col in dimension_cols:
            base = min_col.replace("_min", "")
            f[f"{base}_range"] = (row[min_col], row[max_col])

        # Categorical features
        for col in categorical_cols:
            f[col] = row.get(col, np.nan)
            
        # Grade properties midpoints
        for min_col, max_col in grade_property_cols:
            base = min_col.replace("_min", "") 
            min_val = row.get(min_col, np.nan)
            max_val = row.get(max_col, np.nan)
            f[f"{base}_mid"] = compute_midpoint(min_val, max_val)
               
        feature_vectors[fid] = f


# Experiment 1: Run experiments with different weights
    experiments_results = {}
    for exp in experiment_weights():
        results = []
        rfq_ids = list(feature_vectors.keys())
        
        for i, id1 in enumerate(rfq_ids):
            scores = []
            for id2 in rfq_ids:
                if id1 != id2:
                    sim = compute_similarity(
                        feature_vectors[id1], 
                        feature_vectors[id2], 
                        exp['weights']
                    )
                    scores.append((id2, sim))
            
            # Get top 3 matches
            top3 = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
            for match_id, sim in top3:
                results.append({
                    "rfq_id": id1,
                    "match_id": match_id,
                    "similarity_score": sim,
                    "experiment": exp['name']
                })
        
        experiments_results[exp['name']] = results
        
        # Save results for this experiment
        pd.DataFrame(results).to_csv(f"task_2_output/top3_{exp['name']}.csv", index=False)
    
    # Visualize results
    visualize_results(experiments_results)
    

# Experiment 2: K-Means Clustering

    clusters = cluster_and_visualize(feature_vectors)
    cluster_df = pd.DataFrame(list(clusters.items()), columns=['rfq_id', 'cluster'])
    cluster_df.to_csv('task_2_output/rfq_clusters.csv', index=False)

if __name__ == "__main__":
    main()


