import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        .course-item h4 {
            font-family: 'Arial', sans-serif;
            color: skyblue;
        }
        .stMarkdown h4 {
            font-family: 'Arial', sans-serif;
            color: skyblue;
        }
    </style>
""", unsafe_allow_html=True)

def create_realistic_user_data(course_data, n_users=1000):
    """
    Creates realistic user data based on course characteristics and common learning patterns.
    """
    # Create base user profiles
    user_types = {
        'beginner': {
            'difficulty_prefs': {'easy': 0.7, 'medium': 0.25, 'hard': 0.05},
            'performance_params': {'mean': 75, 'std': 10},
            'learning_hours': {'mean': 5, 'std': 2},
            'weight': 0.4
        },
        'intermediate': {
            'difficulty_prefs': {'easy': 0.2, 'medium': 0.6, 'hard': 0.2},
            'performance_params': {'mean': 82, 'std': 8},
            'learning_hours': {'mean': 8, 'std': 3},
            'weight': 0.35
        },
        'advanced': {
            'difficulty_prefs': {'easy': 0.05, 'medium': 0.35, 'hard': 0.6},
            'performance_params': {'mean': 88, 'std': 7},
            'learning_hours': {'mean': 12, 'std': 4},
            'weight': 0.25
        }
    }
    
    user_data = []
    user_types_list = list(user_types.keys())
    user_type_weights = [user_types[t]['weight'] for t in user_types_list]
    topics = ['Data Science', 'AI', 'Machine Learning', 'Web Development', 'Python']
    
    for user_id in range(1, n_users + 1):
        user_type = np.random.choice(user_types_list, p=user_type_weights)
        profile = user_types[user_type]
        
        education_probs = {
            'beginner': {'High School': 0.5, 'Undergraduate': 0.4, 'Postgraduate': 0.1},
            'intermediate': {'High School': 0.2, 'Undergraduate': 0.5, 'Postgraduate': 0.3},
            'advanced': {'High School': 0.1, 'Undergraduate': 0.4, 'Postgraduate': 0.5}
        }
        
        education_level = np.random.choice(
            list(education_probs[user_type].keys()),
            p=list(education_probs[user_type].values())
        )
        
        # Generate performance scores
        base_performance = np.random.normal(
            profile['performance_params']['mean'],
            profile['performance_params']['std']
        )
        education_bonus = {'High School': 0, 'Undergraduate': 3, 'Postgraduate': 5}
        performance = min(100, max(0, base_performance + education_bonus[education_level]))
        
        user_data.append({
            'user_id': user_id,
            'user_type': user_type,
            'preferred_topics': np.random.choice(topics),
            'preferred_difficulty': np.random.choice(
                list(profile['difficulty_prefs'].keys()),
                p=list(profile['difficulty_prefs'].values())
            ),
            'education_level': education_level,
            'learning_hours_per_week': max(1, int(np.random.normal(
                profile['learning_hours']['mean'],
                profile['learning_hours']['std']
            ))),
            'previous_performance': performance
        })
    
    return pd.DataFrame(user_data)

def generate_realistic_ratings(user_df, course_data, ratings_per_user=50):
    """
    Generates realistic course ratings based on user profiles and course characteristics.
    """
    user_course_data = []
    
    for _, user in user_df.iterrows():
        # Select random courses for the user
        selected_courses = course_data.sample(n=min(ratings_per_user, len(course_data)))
        
        for _, course in selected_courses.iterrows():
            # Calculate base rating
            base_rating = 3.5
            
            # Adjust rating based on topic match (using course title as proxy)
            topic_match = user['preferred_topics'].lower() in course['course_title'].lower()
            base_rating += 0.5 if topic_match else -0.3
            
            # Adjust rating based on difficulty match
            difficulty_match = course['course_difficulty'] == user['preferred_difficulty']
            base_rating += 0.3 if difficulty_match else -0.2
            
            # Adjust based on user performance and course difficulty
            difficulty_levels = {'easy': 1, 'medium': 2, 'hard': 3}
            course_diff_level = difficulty_levels[course['course_difficulty']]
            performance_factor = (user['previous_performance'] - 70) / 30
            base_rating += performance_factor * (0.2 if course_diff_level <= 2 else 0.4)
            
            # Add some random noise
            rating = min(5, max(1, base_rating + np.random.normal(0, 0.2)))
            
            user_course_data.append({
                'user_id': user['user_id'],
                'course_title': course['course_title'],
                'course_difficulty': course['course_difficulty'],
                'course_organization': course['course_organization'],
                'course_Certificate_type': course['course_Certificate_type'],
                'preferred_topics': user['preferred_topics'],
                'preferred_difficulty': user['preferred_difficulty'],
                'previous_performance': user['previous_performance'],
                'user_rating': round(rating, 1)
            })
    
    return pd.DataFrame(user_course_data)

# --- Step 1: Load Coursera Dataset ---
@st.cache_data
def load_data():
    try:
        course_data = pd.read_csv('coursea_data.csv')
        return course_data
    except FileNotFoundError:
        st.error("Error: coursea_data.csv file not found. Please ensure the file exists in the same directory.")
        return None

course_data = load_data()

if course_data is not None:
    # --- Step 2: Preprocess Course Data ---
    difficulty_mapping = {
        'Beginner': 'easy',
        'Intermediate': 'medium',
        'Advanced': 'hard',
        'Mixed': 'medium'
    }
    course_data['course_difficulty'] = course_data['course_difficulty'].map(difficulty_mapping)

    # Fill missing values
    course_data = course_data.fillna({
        'course_difficulty': 'medium',
        'course_organization': 'unknown',
        'course_Certificate_type': 'none'
    })

    # --- Step 3: Generate Realistic User Data ---
    user_df = create_realistic_user_data(course_data)
    user_course_data = generate_realistic_ratings(user_df, course_data)

    # --- Step 4: Train the Model ---
    X = pd.get_dummies(user_course_data[['course_difficulty', 'course_organization', 'course_Certificate_type', 
                                         'preferred_topics', 'preferred_difficulty', 'previous_performance']])
    y = user_course_data['user_rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    @st.cache_resource
    def train_model(X_train, y_train):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    model = train_model(X_train, y_train)

    # --- Streamlit Interface ---
    st.title("Personalized Course Recommendation System")

    # Sidebar for User Inputs
    st.sidebar.header("User Preferences")
    preferred_topics = st.sidebar.selectbox("Select Preferred Topic", 
                                            ['Data Science', 'AI', 'Machine Learning', 'Web Development', 'Python'])
    preferred_difficulty = st.sidebar.selectbox("Select Difficulty", ['easy', 'medium', 'hard'])
    previous_performance = st.sidebar.slider("Previous Performance (Score)", 0, 100, 85)

    preferred_certificate_type = st.sidebar.selectbox(
        "Preferred Certificate Type",
        course_data['course_Certificate_type'].unique().tolist()
    )

    preferred_organization = st.sidebar.selectbox(
        "Preferred Course Organization",
        course_data['course_organization'].unique().tolist()
    )

    # Prepare user input dictionary
    user_input = {
        'preferred_topics': preferred_topics,
        'preferred_difficulty': preferred_difficulty,
        'previous_performance': previous_performance,
        'course_Certificate_type': preferred_certificate_type,
        'course_organization': preferred_organization
    }

    # Recommendation Function
    def recommend_courses(user_input, course_data, model):
        # Filter courses by the user's preferred topic
        filtered_courses = course_data[course_data['course_title'].str.contains(user_input['preferred_topics'], case=False, na=False)]

        if filtered_courses.empty:
            st.warning("No courses found for the selected topic. Showing recommendations across all topics.")
            filtered_courses = course_data

        course_predictions = []
        for _, course in filtered_courses.iterrows():
            X_pred = pd.DataFrame({
                'course_difficulty': [course['course_difficulty']],
                'course_organization': [course['course_organization']],
                'course_Certificate_type': [course['course_Certificate_type']],
                'preferred_topics': [user_input['preferred_topics']],
                'preferred_difficulty': [user_input['preferred_difficulty']],
                'previous_performance': [user_input['previous_performance']]
            })

            X_pred = pd.get_dummies(X_pred).reindex(columns=model.feature_names_in_, fill_value=0)
            prediction = model.predict(X_pred)[0]
            course_predictions.append((course['course_title'], prediction))

        return sorted(course_predictions, key=lambda x: x[1], reverse=True)[:5]

    # Get recommendations
    top_courses = recommend_courses(user_input, course_data, model)

    # Display recommendations
    st.header("Top 5 Recommended Courses")
    for i, (course, score) in enumerate(top_courses):
        st.markdown(f"""
        <div class="course-item">
            <h4>{i+1}. {course} - Predicted Rating: {score:.2f}</h4>
        </div>
        """, unsafe_allow_html=True)

    # Model evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.header("Model Evaluation")
    st.subheader(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    st.subheader(f"Mean Squared Error (MSE): {mse:.3f}")

    # Visualization
    col1, col2 = st.columns(2)

    with col1:
        plt.style.use('ggplot')
        fig1, ax1 = plt.subplots(figsize=(14, 9))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax1, alpha=0.5)
        ax1.plot([1, 5], [1, 5], 'r--')
        ax1.set_title("Predicted vs Actual Ratings", fontsize=18, fontweight='bold')
        ax1.set_xlabel("Actual Ratings", fontsize=14)
        ax1.set_ylabel("Predicted Ratings", fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(14, 9))
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=ax2)
        ax2.set_title("Top 10 Most Important Features", fontsize=18, fontweight='bold')
        ax2.set_xlabel("Importance", fontsize=14)
        ax2.set_ylabel("Feature", fontsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        st.pyplot(fig2)