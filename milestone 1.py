#imports
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler
from nltk.stem import WordNetLemmatizer
from collections import Counter

# DATASET 1
#reading patient_details.csv
df = pd.read_csv(r"C:\Users\Srimathi\ai-ehr-project\data\ehr\patient_details.csv")

#data inspection
print(df.head())
print(df.info())
print(df.describe())

#labels
print("Legend for 'sex' column:")
print("0 → Female")
print("1 → Male")

#checking for null values
print(df.isnull().sum())

#checking for duplicate values
print(df.duplicated().any())

#handling outliers
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['age'] >= Q1 - 1.5*IQR) & (df['age'] <= Q3 + 1.5*IQR)]

#converting object to date
df['DOB'] = pd.to_datetime(df['DOB'], format='%d/%m/%Y', errors='coerce')

#normalising age
scaler = MinMaxScaler()
df['age_norm'] = scaler.fit_transform(df[['age']])
print(df.info())

#saving cleaned dataset
df.to_csv("patient_details_cleaned.csv", index=False)

#labeling gender
df['sex'] = df['sex'].map({0: 'Female', 1: 'Male'})

#visualization
# 1. Histogram for Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['age'], bins=6, kde=True, color='skyblue')
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 2. Countplot for Gender Distribution
plt.figure(figsize=(5,3))
sns.countplot(x='sex', data=df, color='lightpink')
plt.title('Gender Distribution of Patients')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# 3. Boxplot for Age Distribution by Gender
plt.figure(figsize=(6,4))
sns.boxplot(x='sex', y='age', data=df, color='yellow')
plt.title('Age Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()

#################################################################################################
# DATASET 2
#reading patient_records.csv
df = pd.read_csv(r"C:\Users\Srimathi\ai-ehr-project\data\ehr\patient_records.csv")

#data inspection
print(df.head())
print(df.info())
print(df.describe())

#labels
cat_legend = {
    'target': {0: 'No Disease', 1: 'Disease'},
    'cp': {0: 'Asymptomatic', 1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-Anginal Pain'},
    'fbs': {0: 'Normal', 1: 'High'},
    'restecg': {0: 'Normal', 1: 'ST-T Abnormality', 2: 'Left Ventricular Hypertrophy'},
    'exang': {0: 'No', 1: 'Yes'},
    'slope': {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'},
    'thal': {1: 'Normal', 2: 'Fixed Defect', 3: 'Reversible Defect'}
}
for col, mapping in cat_legend.items():
    print(f"\nLegend for '{col}' column:")
    for code, meaning in mapping.items():
        print(f"{code} -> {meaning}")

#checking for null values
print(df.isnull().sum())

#checking for duplicate values
print(df.duplicated().any())

#checking for outliers
numeric_cols = ['trestbps', 'chol', 'thalach', 'oldpeak']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")
    
#converting object to date
df['visit_date'] = pd.to_datetime(df['visit_date'], format='%d/%m/%Y', errors='coerce')

#text normalisation
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
def normalize_text(text):
    text = str(text).lower().strip() 
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])  
    return text
df['description'] = df['description'].apply(normalize_text)
df['department'] = df['department'].apply(normalize_text)
print(df[['description', 'department']].head())

#using TF-IDF to convert text into numeric TF-IDF features to process the description column
df['description'] = df['description'].astype(str)
tfidf = TfidfVectorizer(stop_words='english', max_features=500)  
X_description = tfidf.fit_transform(df['description'])
feature_names = tfidf.get_feature_names_out()
tfidf_df = pd.DataFrame(X_description.toarray(), columns=feature_names)
print("TF-IDF shape : ",tfidf_df.shape)      
print(tfidf_df.head())

#saving cleaned dataset
df.to_csv("patient_records_cleaned.csv", index=False)
print(df.groupby('target')['chol'].mean())

#visualization
#countplot for Heart Disease Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='target', hue='target', palette='pastel', legend=False)
plt.title('Heart Disease Distribution')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

#countplot for Chest Pain Type Distribution
cp_labels = {
    0: 'Typical Angina',
    1: 'Atypical Angina',
    2: 'Non-anginal Pain',
    3: 'Asymptomatic'
}
df['cp_label'] = df['cp'].map(cp_labels)
plt.figure(figsize=(7,4))
sns.countplot(data=df, x='cp_label', hue='cp_label', palette='Set2', legend=False)
plt.title('Chest Pain Type Distribution')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.xticks(rotation=20)
plt.show()

#histogram for Cholesterol Level Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['chol'], bins=6, kde=True, color='skyblue')
plt.title('Cholesterol Level Distribution')
plt.xlabel('Cholesterol (mg/dL)')
plt.ylabel('Count')
plt.show()

#countplot for Max Heart Rate vs Heart Disease
plt.figure(figsize=(6,4))
sns.stripplot(data=df, x='target', y='thalach', color='dodgerblue', alpha=0.7)
plt.title('Max Heart Rate vs Heart Disease')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Max Heart Rate')
plt.show()

#barplot for Cholesterol vs Heart Disease
plt.figure(figsize=(6,4))
sns.barplot(data=df, x='target', y='chol', color='skyblue')
plt.title('Cholesterol vs Heart Disease')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Average Cholesterol')
plt.show()

####################################################################################################
#DATASET 3
#reading lab_details.csv
df = pd.read_csv(r"C:\Users\Srimathi\ai-ehr-project\data\ehr\lab_details.csv")

#data inspection
print(df.head())
print(df.info())
print(df.describe())

#checking for null values
print(df.isnull().sum())

#checking for duplicate values
print(df.duplicated().any())

#handling inconsistent data
df['sample_name'] = df['sample_name'].astype(str).str.lower().str.strip()

#converting object to date
df['visit_date'] = pd.to_datetime(df['visit_date'], format='%d/%m/%Y', errors='coerce')

#using TF-IDF to convert text into numeric TF-IDF features to process the sample_name column
df['sample_name'] = df['sample_name'].astype(str)
tfidf = TfidfVectorizer(stop_words='english', max_features=500) 
X_sample = tfidf.fit_transform(df['sample_name'])
feature_names = tfidf.get_feature_names_out()
tfidf_df = pd.DataFrame(X_sample.toarray(), columns=feature_names)
print("TF-IDF shape:", X_sample.shape)
print(tfidf_df.head())

#saving cleaned dataset
df.to_csv("lab_details_cleaned.csv", index=False)

#visualization
#plot for no of samples per week
df['visit_date'] = pd.to_datetime(df['visit_date'])
weekly_counts = df.groupby(df['visit_date'].dt.isocalendar().week).size()
plt.figure(figsize=(6,3))
plt.plot(weekly_counts.index, weekly_counts.values, marker='o', color='lightgreen')
plt.title('Weekly Lab Samples')
plt.xlabel('Week Number')
plt.ylabel('Number of Samples')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

#barplot for top 5 samples
top_samples = df['sample_name'].value_counts().head(5)
plt.figure(figsize=(6,3))
plt.bar(top_samples.index, top_samples.values, color='salmon')
plt.title('Top 5 Lab Samples')
plt.xlabel('Sample Name')
plt.ylabel('Count')
plt.xticks(rotation=20)
plt.show()

#####################################################################################################
#DATASET 4
#reading doctor_prescription.csv
df = pd.read_csv(r"C:\Users\Srimathi\ai-ehr-project\data\ehr\doctor_prescription.csv")

#data inspection
print(df.head())
print(df.info())
print(df.describe())

#checking for null values
print(df.isnull().sum())

#checking for duplicate values
print(df.duplicated().any())

#handling inconsistent data
df['transcription'] = df['transcription'].astype(str).str.lower().str.strip()
df['keywords'] = df['keywords'].astype(str).str.lower().str.strip()

#converting object to date
df['visit_date'] = pd.to_datetime(df['visit_date'], format='%d/%m/%Y', errors='coerce')

#using TF-IDF to convert text into numeric TF-IDF features to process the columns named 'transcription','keywords'.
text_cols = ['transcription', 'keywords']
for col in text_cols:
    df[col] = df[col].astype(str)
tfidf_features = []
feature_names = []
for col in text_cols:
    tfidf = TfidfVectorizer(stop_words='english', max_features=500)
    X_col = tfidf.fit_transform(df[col])
    tfidf_features.append(X_col)
    
    feature_names.extend([f"{col}_{word}" for word in tfidf.get_feature_names_out()])
X_text_final = hstack(tfidf_features)
tfidf_df = pd.DataFrame(X_text_final.toarray(), columns=feature_names)
print(tfidf_df.head())
print("Shape:", tfidf_df.shape)

#saving cleaned dataset
df.to_csv("doctor_prescription_cleaned.csv", index=False)

#visualization
#histogram for Distribution of Prescription Lengths
plt.figure(figsize=(6,3))
plt.hist(df['transcription'].str.split().str.len(), bins=5, color='skyblue', edgecolor='black')
plt.title('Distribution of Prescription Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Number of Prescriptions')
plt.show()

#barplot for top 10 keywords
all_keywords = df['keywords'].str.split(',').explode().str.strip()
top_keywords = Counter(all_keywords).most_common(10) 
labels, values = zip(*top_keywords)
plt.figure(figsize=(8,4))
plt.bar(labels, values, color='lightgreen')
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Keywords in Prescriptions')
plt.ylabel('Frequency')
plt.show()

###################################################################################################

