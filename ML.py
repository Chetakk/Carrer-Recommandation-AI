
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate synthetic student data
def generate_student_data(num_students=100):
    """Generate synthetic student data with academic scores, interests, and personality traits."""

    # Academic subjects
    subjects = ['Math', 'Science', 'Literature', 'History', 'Art', 'Computer Science', 'Economics']

    # Interest areas
    interests = ['Technology', 'Healthcare', 'Business', 'Arts', 'Education', 'Engineering']

    # Generate data
    student_ids = [f"S{i+1:03d}" for i in range(num_students)]

    # Generate academic scores (0-100)
    academic_scores = np.random.randint(60, 100, size=(num_students, len(subjects)))

    # Generate interest scores (1-5)
    interest_scores = np.random.randint(1, 6, size=(num_students, len(interests)))

    # Generate personality traits (0-1 scale for analytical, creative, social)
    personality_traits = np.random.random(size=(num_students, 3))

    # Create DataFrame
    columns = subjects + interests + ['Analytical', 'Creative', 'Social']
    df = pd.DataFrame(
        np.hstack((academic_scores, interest_scores, personality_traits)),
        index=student_ids,
        columns=columns
    )

    df.to_csv('student.csv')

    return df

# Step 2: Define careers and their requirements
def define_careers():
    """Define various careers and their required skills/traits."""

    careers = {
        'Software Developer': {
            'Math': 0.8, 'Computer Science': 0.9, 'Science': 0.5,
            'Technology': 0.9, 'Engineering': 0.7,
            'Analytical': 0.8, 'Creative': 0.6, 'Social': 0.4
        },
        'Data Scientist': {
            'Math': 0.9, 'Computer Science': 0.8, 'Science': 0.7,
            'Technology': 0.8, 'Business': 0.6,
            'Analytical': 0.9, 'Creative': 0.5, 'Social': 0.5
        },
        'Doctor': {
            'Science': 0.9, 'Math': 0.7,
            'Healthcare': 0.9, 'Technology': 0.5,
            'Analytical': 0.8, 'Creative': 0.5, 'Social': 0.9
        },
        'Teacher': {
            'Literature': 0.7, 'History': 0.7,
            'Education': 0.9, 'Arts': 0.5,
            'Analytical': 0.6, 'Creative': 0.7, 'Social': 0.9
        },
        'Business Analyst': {
            'Math': 0.7, 'Economics': 0.8, 'Computer Science': 0.5,
            'Business': 0.9, 'Technology': 0.6,
            'Analytical': 0.9, 'Creative': 0.5, 'Social': 0.7
        },
        'Graphic Designer': {
            'Art': 0.9, 'Computer Science': 0.6,
            'Arts': 0.9, 'Technology': 0.6,
            'Analytical': 0.5, 'Creative': 0.9, 'Social': 0.6
        },
        'Marketing Specialist': {
            'Economics': 0.7, 'Literature': 0.6,
            'Business': 0.8, 'Arts': 0.7,
            'Analytical': 0.7, 'Creative': 0.8, 'Social': 0.9
        },
        'Researcher': {
            'Science': 0.9, 'Math': 0.8, 'Computer Science': 0.7,
            'Technology': 0.7, 'Healthcare': 0.6, 'Education': 0.6,
            'Analytical': 0.9, 'Creative': 0.6, 'Social': 0.5
        }
    }

    # Market demand score (1-10 scale)
    market_demand = {
        'Software Developer': 9,
        'Data Scientist': 10,
        'Doctor': 8,
        'Teacher': 7,
        'Business Analyst': 8,
        'Graphic Designer': 6,
        'Marketing Specialist': 7,
        'Researcher': 7
    }

    return careers, market_demand

# Step 3: Create career recommendation function
def recommend_careers(student_data, careers, market_demand, top_n=3):
    """Recommend careers based on student data and career requirements."""

    # Normalize student data
    scaler = StandardScaler()
    student_data_normalized = pd.DataFrame(
        scaler.fit_transform(student_data),
        index=student_data.index,
        columns=student_data.columns
    )

    # Create career vectors based on the same features as student data
    career_vectors = []
    career_names = []

    for career, requirements in careers.items():
        career_vector = []

        for feature in student_data.columns:
            if feature in requirements:
                # Convert percentage score to comparable scale
                # Map career requirement (0-1) to similar scale as normalized student data
                career_vector.append(requirements[feature] * 2 - 1)
            else:
                career_vector.append(-1)  # Low preference if not specified

        career_vectors.append(career_vector)
        career_names.append(career)

    career_df = pd.DataFrame(career_vectors, index=career_names, columns=student_data.columns)

    # Calculate compatibility scores using cosine similarity
    recommendations = {}
    all_scores = {}

    for student_id in student_data.index:
        student_vector = student_data_normalized.loc[student_id].values.reshape(1, -1)

        # Calculate similarity to each career
        similarities = {}
        for career in career_names:
            career_vector = career_df.loc[career].values.reshape(1, -1)
            similarity = cosine_similarity(student_vector, career_vector)[0][0]

            # Add market demand factor (weighted 30%)
            demand_score = market_demand[career] / 10  # Normalize to 0-1
            final_score = 0.7 * similarity + 0.3 * demand_score

            similarities[career] = round(final_score, 3)

        # Save all scores for visualization
        all_scores[student_id] = similarities

        # Sort and get top recommendations
        sorted_careers = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        recommendations[student_id] = sorted_careers[:top_n]

    return recommendations, all_scores

# Step 4: Segment students to identify common patterns
def segment_students(student_data, n_clusters=4):
    """Segment students into clusters based on their attributes."""

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(student_data)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    # Add cluster information to student data
    clustered_students = student_data.copy()
    clustered_students['Cluster'] = clusters

    # Analyze each cluster
    cluster_profiles = {}

    for i in range(n_clusters):
        cluster_data = clustered_students[clustered_students['Cluster'] == i]
        profile = {
            'size': len(cluster_data),
            'top_subjects': [],
            'top_interests': []
        }

        # Find top subjects for this cluster
        subjects = ['Math', 'Science', 'Literature', 'History', 'Art', 'Computer Science', 'Economics']
        cluster_avg = cluster_data[subjects].mean()
        full_avg = student_data[subjects].mean()

        # Subjects where this cluster exceeds the average
        for subject in subjects:
            if cluster_avg[subject] > full_avg[subject]:
                profile['top_subjects'].append(subject)

        # Find top interests for this cluster
        interests = ['Technology', 'Healthcare', 'Business', 'Arts', 'Education', 'Engineering']
        cluster_avg = cluster_data[interests].mean()
        full_avg = student_data[interests].mean()

        # Interests where this cluster exceeds the average
        for interest in interests:
            if cluster_avg[interest] > full_avg[interest]:
                profile['top_interests'].append(interest)

        cluster_profiles[i] = profile

    return clustered_students, cluster_profiles

# New function to visualize career recommendations
def visualize_career_recommendations(student_id, all_scores, recommendations, market_demand):
    """Create a visual plot showing all careers with dotted line for top 3 recommendations."""

    # Get all career scores for this student
    career_scores = all_scores[student_id]

    # Get top 3 recommended careers
    top_careers = [career for career, _ in recommendations[student_id]]

    # Create figure
    plt.figure(figsize=(12, 6))

    # Sort careers by score for better visualization
    sorted_careers = sorted(career_scores.items(), key=lambda x: x[1])
    careers = [item[0] for item in sorted_careers]
    scores = [item[1] for item in sorted_careers]

    # Create bar colors based on whether it's a top recommendation
    colors = ['lightblue' if career not in top_careers else 'blue' for career in careers]

    # Create the bar chart
    bars = plt.barh(careers, scores, color=colors, alpha=0.7)

    # Add market demand indicators
    for i, career in enumerate(careers):
        demand = market_demand[career]
        plt.text(0.01, i, f"Demand: {demand}/10", va='center', color='black', fontweight='bold')

    # Create a horizontal dotted line connecting the top 3 recommendations
    top_indices = [careers.index(career) for career in top_careers]
    for idx in top_indices:
        plt.plot([0, scores[idx]], [idx, idx], 'r--', linewidth=2)
        plt.scatter(scores[idx], idx, color='red', s=100, zorder=5)

    plt.title(f"Career Recommendations for Student {student_id}", fontsize=14)
    plt.xlabel("Match Score", fontsize=12)
    plt.ylabel("Career Options", fontsize=12)
    plt.xlim(0, 1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    return plt

# Main function to put everything together
def career_recommendation_system():
    print("Generating student data...")
    # Generate student data
    student_data = generate_student_data(num_students=2000)  # Using small sample for demonstration
    print("\nStudent Data (Sample):")
    print(student_data.head())

    # Define careers and market demand
    print("\nDefining career requirements and market demand...")
    careers, market_demand = define_careers()
    print("\nCareers and Their Market Demand:")
    for career, demand in market_demand.items():
        print(f"{career}: {demand}/10")

    # Segment students
    print("\nSegmenting students into clusters...")
    clustered_students, cluster_profiles = segment_students(student_data)
    print("\nCluster Profiles:")
    for cluster_id, profile in cluster_profiles.items():
        print(f"Cluster {cluster_id}:")
        print(f"  Size: {profile['size']} students")
        print(f"  Top Subjects: {', '.join(profile['top_subjects'])}")
        print(f"  Top Interests: {', '.join(profile['top_interests'])}")

    # Make recommendations
    print("\nGenerating career recommendations...")
    recommendations, all_scores = recommend_careers(student_data, careers, market_demand)

    # Print recommendations for each student
    print("\nPersonalized Career Recommendations:")
    for i, student_id in enumerate(list(recommendations.keys())[:5]):  # Show first 5 students for brevity
        print(f"\nStudent {student_id} (Cluster {clustered_students.loc[student_id, 'Cluster']}):")
        print(f"Academic Profile: {dict(student_data.loc[student_id, ['Math', 'Science', 'Literature', 'Computer Science']])}")
        print(f"Interest Profile: {dict(student_data.loc[student_id, ['Technology', 'Healthcare', 'Business', 'Arts']])}")
        print(f"Personality: {dict(student_data.loc[student_id, ['Analytical', 'Creative', 'Social']])}")
        print("Recommended Careers:")
        for i, (career, score) in enumerate(recommendations[student_id], 1):
            print(f"  {i}. {career} (Match Score: {score:.2f}, Market Demand: {market_demand[career]}/10)")

        # Visualize recommendations for this student
        plt.figure(i)
        visualize_career_recommendations(student_id, all_scores, recommendations, market_demand)
        plt.tight_layout()
        plt.show()
        print("\n" + "-"*80)

# Run the system
if __name__ == "__main__":
    career_recommendation_system()
