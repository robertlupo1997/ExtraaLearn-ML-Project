# ExtraaLearn-ML-Project
## ExtraaLearn Lead Conversion Prediction

## Context
The EdTech industry has seen immense growth in the past decade. According to forecasts, the online education market is expected to be worth $286.62 billion by 2023, with a compound annual growth rate (CAGR) of 10.26% from 2018 to 2023. The modern era of online education has expanded far beyond traditional methods due to its ease of information sharing, personalized learning experiences, and transparent assessments.

In light of the Covid-19 pandemic, the online education sector has experienced rapid growth, attracting new customers. This growth has led to the emergence of new companies in this industry. With the availability of digital marketing resources, these companies can reach a broader audience. Customers who show interest in offerings are termed as leads. These leads are generated through:
- Social media and other online platforms.
- Website/app interactions, including brochure downloads.
- Email inquiries for more information.

The company nurtures these leads and aims to convert them into paid customers by sharing additional details through calls or emails.

## Objective
ExtraaLearn, an early-stage startup offering upskilling and reskilling programs, faces the challenge of identifying high-potential leads for effective resource allocation. As a data scientist at ExtraaLearn, your task is to:
1. Build a machine learning model to predict lead conversion.
2. Identify factors driving the lead conversion process.
3. Create a profile of leads most likely to convert.

## Data Description
The dataset contains 4,612 records with 15 attributes related to leads and their interactions with ExtraaLearn. The key features include:
| **Feature**              | **Description**                                                                 |
|---------------------------|---------------------------------------------------------------------------------|
| ID                       | Unique identifier for each lead.                                                |
| age                      | Age of the lead.                                                               |
| current_occupation       | Current occupation (`Professional`, `Unemployed`, `Student`).                   |
| first_interaction        | Initial interaction channel (`Website`, `Mobile App`).                          |
| profile_completed        | Profile completion percentage (`Low`, `Medium`, `High`).                        |
| website_visits           | Number of website visits.                                                      |
| time_spent_on_website    | Total time spent on the website (seconds).                                      |
| page_views_per_visit     | Average pages viewed per visit.                                                |
| last_activity            | Last interaction (`Email Activity`, `Phone Activity`, `Website Activity`).      |
| print_media_type1        | Flag for newspaper advertisement (`Yes`, `No`).                                |
| print_media_type2        | Flag for magazine advertisement (`Yes`, `No`).                                 |
| digital_media            | Flag for digital platform advertisement (`Yes`, `No`).                         |
| educational_channels     | Flag for presence on educational forums and websites (`Yes`, `No`).             |
| referral                 | Flag for referral source (`Yes`, `No`).                                        |
| status                   | Target variable indicating lead conversion (`1`: Converted, `0`: Not converted).|

## Key Observations
### Data Structure
- **Rows and Columns**: The dataset contains 4,612 rows and 15 columns.
- **Data Types**: A mix of numerical (`int64`, `float64`) and categorical (`object`) variables.
- **Missing Values**: No missing values detected.
- **Target Variable**: `status` (binary).

### Summary Statistics
- **Numerical Features**:
  - `age`: Mean = 46.20, Std Dev = 13.16, Range = [18, 63].
  - `time_spent_on_website`: Mean = 724.01 seconds, Std Dev = 743.83, Max = 2,537 seconds.
- **Categorical Features**: Balanced distribution for most features, except referrals (`Yes`: ~2%).

## Methodology

### Exploratory Data Analysis (EDA)
1. **Data Understanding**: Checked for duplicates, missing values, and data types.
2. **Univariate Analysis**: Analyzed the distribution of individual features using histograms and boxplots.
3. **Bivariate Analysis**: Explored relationships between features and the target variable using boxplots and cross-tabulations.
4. **Outlier Detection**: Retained outliers in `time_spent_on_website` and `website_visits` to preserve valuable insights.

### Key Findings
1. **Time Spent on Website**: Strongest predictor of lead conversion.
2. **First Interaction**: Leads engaging through the website convert more frequently than app users.
3. **Profile Completion**: High completion rates significantly boost conversion.
4. **Age**: Leads aged 45-55 exhibit higher conversion rates.
5. **Lead Sources**: Digital media and referrals outperform print media in conversions.

### Data Preprocessing
1. **Feature Engineering**: Converted categorical variables into dummy variables using one-hot encoding.
2. **Data Splitting**: Split the data into training and testing sets (70:30 ratio).
3. **Scaling**: Not required for tree-based models.

## Model Building

### Decision Tree Classifier
1. **Initial Model**: Built using default parameters.
2. **Issue**: Overfitting detected (perfect training performance but lower test performance).
3. **Hyperparameter Tuning**:
   - Used GridSearchCV to tune `max_depth`, `criterion`, and `min_samples_leaf`.
   - Optimized for recall to minimize false negatives.
4. **Final Model**: Improved generalization with balanced performance.

### Random Forest Classifier
1. **Initial Model**: Built using default parameters.
2. **Issue**: Overfitting observed (similar to the Decision Tree).
3. **Hyperparameter Tuning**:
   - Tuned `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `max_samples`, and `class_weight`.
   - Optimized for recall.
4. **Final Model**: Achieved the best overall performance with high recall and precision.

## Model Evaluation
Metrics used:
- **Accuracy**: Overall correctness of the model.
- **Precision**: Correct positive predictions over total positive predictions.
- **Recall**: Ability to identify actual positives (converted leads).
- **F1-Score**: Harmonic mean of precision and recall.

### Model Performance
| Model                  | Accuracy | F1-Score | Recall | Precision |
|------------------------|----------|----------|--------|-----------|
| Decision Tree (Tuned)  | 80%      | 72%      | 86%    | 62%       |
| Random Forest (Tuned)  | 83%      | 76%      | 85%    | 68%       |

## Results

### Key Findings
- **Time Spent on Website**: Leads spending over 1,000 seconds are more likely to convert.
- **First Interaction Channel**: Website interactions yield higher conversion rates.
- **Profile Completion**: High profile completion leads have a 60% conversion rate.
- **Age Factor**: Leads aged 45–55 have a 50% conversion rate.
- **Lead Source Channels**: Digital media and referrals outperform print media.

### Feature Importance
Top features influencing lead conversion:
1. `time_spent_on_website`
2. `first_interaction_Website`
3. `profile_completed`
4. `age`
5. `last_activity_Website Activity`

## Actionable Insights and Recommendations
1. **Optimize Website Experience**: Enhance design and content for >1,000 seconds of engagement.
2. **Encourage Profile Completion**: Incentivize users with rewards for full profiles.
3. **Target Professionals Aged 45–55**: Focus marketing efforts on this demographic.
4. **Invest in Digital Media and Referral Programs**: Allocate resources to high-performing channels.
5. **Improve Mobile App Experience**: Ensure feature parity with the website.

## Conclusion
By leveraging data-driven insights and implementing the recommended strategies, ExtraaLearn can improve lead conversion rates, optimize resource allocation, and focus on high-potential leads.

## Repository Structure
- `RobertLupoFullCodeVersionPotentialCustomersPredictionProject.ipynb`: Jupyter notebook containing the full analysis, modeling, and results.
- `RobertLupoFullCodeVersionPotentialCustomersPredictionProject.html`: HTML version of the notebook for easy viewing.
- `ExtraaLearn.csv`: Dataset used for the project.
- `README.md`: Project overview and documentation (this file).

## Requirements
- Python 3.x
- Jupyter Notebook
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - statsmodels
