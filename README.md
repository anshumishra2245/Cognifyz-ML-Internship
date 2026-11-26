# Cognifyz Technologies — Machine Learning Internship

**Intern:** Anshu Mishra  
**Domain:** Machine Learning  
**Dataset:** Restaurant Dataset (`9,551` records, `21` features)

---

## Project Overview
This repository contains the completed tasks for the **Cognifyz Technologies Machine Learning Internship Program**. Three core tasks were completed using Python, Scikit-learn, Pandas, Seaborn, Matplotlib, and Plotly. Each task is implemented in a modular, well-documented Jupyter Notebook with executed code, markdown explanations, formatted summary tables, and visualizations.

---

## Repository Structure
```text
Cognifyz-ML-Internship/
├── Dataset .csv                            # Primary restaurant dataset (9,551 rows, 21 columns)
├── Task1_Predict_Restaurant_Ratings.ipynb  # Task 1: Regression modeling & feature importance
├── Task2_Restaurant_Recommendation.ipynb   # Task 2: Content-based recommendation engine
├── Task4_Location_Based_Analysis.ipynb     # Task 4: Geospatial mapping & location analytics
└── README.md                               # Internship project documentation
```

---

## Task 1: Predict Restaurant Ratings

### **Objective**
Build, evaluate, and interpret regression models to predict the continuous target variable **`Aggregate rating`** (scale: 0.0 to 5.0) using restaurant operational, geographical, and commercial attributes.

### **Workflow & Methodology**
1. **Data Cleaning:** Loaded `Dataset .csv` and dropped 9 rows with missing `Cuisines` (< 0.1% of records).
2. **Feature Selection & Leakage Prevention:**
   - **Target Leakage:** Dropped `Rating color` and `Rating text` because they are deterministic categorizations derived directly from `Aggregate rating` (e.g. 4.5–5.0 maps to "Dark Green" and "Excellent"). Using them would cause data leakage.
   - **Non-Predictive Text:** Dropped `Restaurant ID`, `Restaurant Name`, `Address`, `Locality Verbose`, and `Locality` to avoid overfitting on arbitrary identifiers and high-cardinality address strings (1,206 unique values).
3. **Feature Encoding:**
   - **Binary Mapping:** Converted `Has Table booking`, `Has Online delivery`, `Is delivering now`, and `Switch to order menu` (`{'Yes': 1, 'No': 0}`).
   - **Frequency Encoding:** Applied to high-cardinality categorical columns (`City`, `Cuisines`, `Currency`). This avoided expanding the feature space into over 1,900 sparse binary columns via One-Hot Encoding while retaining category popularity density.
4. **Data Splitting:** 80% training set (7,633 samples) and 20% test set (1,909 samples) using `random_state=42`.
5. **Model Training & Comparison:** Trained three regression algorithms:
   - `LinearRegression` (Parametric baseline)
   - `DecisionTreeRegressor` (Non-linear single tree)
   - `RandomForestRegressor` (Ensemble bagging of 100 decorrelated trees)

### **Performance Comparison Table**
Evaluated on the unseen 20% test set ($n = 1,909$):

| Model | MSE | RMSE | $R^2$ Score | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **0.0948** | **0.3079** | **0.9586** | 0.289 s |
| **Decision Tree Regressor** | 0.1869 | 0.4324 | 0.9184 | 0.025 s |
| **Linear Regression** | 1.4979 | 1.2239 | 0.3459 | 0.031 s |

### **Top 10 Influential Features (Random Forest MDI)**
| Rank | Feature | Importance Score | Relative Share (%) |
| :---: | :--- | :---: | :---: |
| 1 | `Votes` | 0.94444 | **94.44%** |
| 2 | `City` | 0.01653 | **1.65%** |
| 3 | `Longitude` | 0.01126 | **1.13%** |
| 4 | `Latitude` | 0.00981 | **0.98%** |
| 5 | `Cuisines` | 0.00722 | **0.72%** |
| 6 | `Average Cost for two` | 0.00653 | **0.65%** |
| 7 | `Currency` | 0.00137 | **0.14%** |
| 8 | `Has Online delivery` | 0.00118 | **0.12%** |
| 9 | `Price range` | 0.00094 | **0.09%** |
| 10 | `Has Table booking` | 0.00050 | **0.05%** |

### **Key Insights**
- **Random Forest dominates:** Achieved an $R^2$ of **0.9586** and RMSE of **0.3079**, outperforming Linear Regression ($R^2 \approx 0.3459$). The relationship between features and ratings is non-linear with threshold effects (e.g. unrated restaurants with zero votes having flat 0.0 ratings).
- **Votes is the primary driver:** Accounts for **94.4%** of relative importance. High review engagement strongly correlates with verified, higher ratings.

---

## Task 2: Restaurant Recommendation System

### **Objective**
Build a **Content-Based Filtering Recommendation Engine** that recommends dining venues based on a user's culinary preferences, price range tier (1 = Budget to 4 = Luxury), and optional city filter.

### **Workflow & Methodology**
1. **Data Preparation & Deduplication:** Dropped 9 rows missing `Cuisines`, filtered to 11 relevant attributes, and deduplicated 95 repeated `['Restaurant Name', 'Locality']` entries to yield 9,447 clean restaurants.
2. **Exploratory Visuals:** Extracted individual cuisine frequencies and analyzed price range distributions (Tier 1: 46.9%, Tier 2: 32.6%, Tier 3: 14.8%, Tier 4: 5.7%).
3. **Unified Content Representation:**
   - Text features (`Cuisines + City`) transformed using `TfidfVectorizer(stop_words='english')`.
   - Numerical features (`Price range` and `Aggregate rating`) scaled to $[0, 1]$ using `MinMaxScaler()`.
   - Combined via `scipy.sparse.hstack` into a unified feature matrix ($9,447 \times 313$).
4. **Recommendation Function:** Built `recommend_restaurants(preferred_cuisine, price_range, city=None, top_n=5)`:
   - Filters candidate restaurants by city when provided.
   - Synthesizes user query vector targeting ideal satisfaction rating (5.0) and user price tier.
   - Computes pairwise **Cosine Similarity**:
     $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{r}_i) = \frac{\mathbf{u} \cdot \mathbf{r}_i}{\|\mathbf{u}\|_2 \|\mathbf{r}_i\|_2}$$
   - Ranks matches using `Similarity Score`, `Aggregate rating`, and `Votes`.

### **Sample Test Profiles & Results**

#### **Profile 1: `preferred_cuisine="North Indian, Chinese"`, `price_range=2`, `city="New Delhi"`**
| Restaurant Name | City | Locality | Cuisines | Price range | Cost for two | Rating | Similarity Score |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **The Culinary Pitaara** | New Delhi | Defence Colony | North Indian, Chinese | 2 | 900 | 4.2 | **0.9963** |
| **Sid's Kitchen** | New Delhi | Kamla Nagar | North Indian, Chinese | 2 | 500 | 3.8 | **0.9909** |
| **Nik's Kitchen** | New Delhi | Rohini | North Indian, Chinese | 2 | 500 | 3.8 | **0.9909** |
| **Masala Factory** | New Delhi | Tagore Garden | North Indian, Chinese | 2 | 500 | 3.8 | **0.9909** |
| **Sethi's Restaurant & Barbeque** | New Delhi | Rajouri Garden | North Indian, Chinese | 2 | 700 | 3.7 | **0.9891** |

#### **Profile 2: `preferred_cuisine="Italian, Continental"`, `price_range=3`, `city="Bangalore"`**
| Restaurant Name | City | Locality | Cuisines | Price range | Cost for two | Rating | Similarity Score |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Hoot** | Bangalore | Sarjapur Road | Continental, Italian, North Indian | 3 | 1,400 | 3.9 | **0.9744** |
| **Communiti** | Bangalore | Residency Road | Continental | 3 | 1,200 | 4.2 | **0.9597** |
| **Koramangala Social** | Bangalore | Koramangala 7th Block | Continental, American | 3 | 1,500 | 4.5 | **0.9192** |
| **ECHOES Koramangala** | Bangalore | Koramangala 5th Block | Continental, American, Italian, etc. | 2 | 950 | 4.7 | **0.8910** |
| **Three Dots & A Dash** | Bangalore | Indiranagar | European, Continental | 3 | 1,300 | 3.9 | **0.8905** |

#### **Profile 3: `preferred_cuisine="Cafe, Fast Food"`, `price_range=1`, `city=None`**
| Restaurant Name | City | Locality | Cuisines | Price range | Cost for two | Rating | Similarity Score |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Canteen Till I Die** | New Delhi | Satyaniketan | Cafe, Fast Food | 1 | 300 | 3.9 | **0.9457** |
| **Slurp Pasta Express** | New Delhi | New Friends Colony | Cafe, Fast Food | 1 | 300 | 2.9 | **0.9148** |
| **Amigo's Hub** | New Delhi | Satyaniketan | Cafe, Fast Food | 2 | 500 | 3.7 | **0.9097** |
| **Fu.D** | Gurgaon | Sushant Lok | Cafe, Fast Food | 1 | 250 | 3.7 | **0.9030** |
| **Thee Pot** | Noida | Sector 37 | Cafe, Fast Food | 1 | 400 | 3.7 | **0.9012** |

---

## Task 4: Location-Based Analysis

### **Objective**
Perform geospatial and geographical analysis to explore restaurant spatial distribution, market concentration, locality clustering, and regional cuisine/pricing patterns.

### **Workflow & Methodology**
1. **Coordinate Cleaning:** Dropped 9 missing cuisine records and filtered out 499 invalid coordinates (`Longitude == 0` or `Latitude == 0`), leaving **9,043 valid geolocated restaurants** across 141 cities.
2. **Geographical Mapping:**
   - **Interactive Map:** Built using Plotly (`open-street-map` style), colored continuously by `Aggregate rating` (0.0 to 5.0), with hover data over restaurant name, locality, cuisines, and price tier.
   - **Static Density Plots:** Rendered global coordinate scatter plot alongside a localized zoom of the Delhi NCR cluster ($76.8^\circ - 77.6^\circ\text{ E}$, $28.3^\circ - 28.9^\circ\text{ N}$).
3. **Concentration Analysis:** Identified top cities and localities by restaurant count.
4. **Location-Based Statistical Aggregations:** Filtered cities with $\ge 20$ listings (48 qualifying cities) to calculate volume, mean rating, mean price tier, mean cost for two, and dominant primary cuisine.

### **Key Findings & Summary Statistics**
- **Extreme NCR Concentration:** **83.5% (7,550 restaurants)** of all valid geolocated venues belong to the Delhi NCR cluster (New Delhi: 5,240; Gurgaon: 1,070; Noida: 986; Faridabad: 229; Ghaziabad: 25).
- **Top Localities:** Connaught Place (122), Rajouri Garden (98), Malviya Nagar (85), Defence Colony (85), Pitampura (83), Shahdara (81), Sector 18 Noida (75).
- **Census vs. Curated Rating Variance:**
  - Delhi NCR exhibits lower average ratings (1.99–2.72) because Zomato exhaustively cataloged small local kiosks and takeaway counters with 0 votes and 0.0 default ratings.
  - Secondary Indian and international hubs show elevated ratings (4.11–4.38 for Bangalore, Ahmedabad, Ankara, Abu Dhabi, Cape Town) due to curated sampling of established dining spots.
- **Regional Culinary Specialization:**
  - North Indian dominates Delhi NCR and Northern India.
  - Bangalore diverges toward `Continental` (IT demographic and bistro/pub culture).
  - Ankara is dominated by `Kebab`, and Abu Dhabi by `Indian`.

---

## Tech Stack & Libraries
- **Language:** Python 3.12
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning & NLP:** `scikit-learn` (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `TfidfVectorizer`, `MinMaxScaler`, `cosine_similarity`)
- **Data Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Notebook Management:** `nbformat`, `nbclient`, `ipykernel`

---

## Installation & How to Run

### **1. Clone Repository**
```bash
git clone https://github.com/anshumishra2245/Cognifyz-ML-Internship.git
cd Cognifyz-ML-Internship
```

### **2. Install Dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly nbformat nbclient ipykernel
```

### **3. Launch Jupyter Notebook**
```bash
jupyter notebook
```
Open and run any of the notebooks:
- `Task1_Predict_Restaurant_Ratings.ipynb`
- `Task2_Restaurant_Recommendation.ipynb`
- `Task4_Location_Based_Analysis.ipynb`

All notebooks are pre-executed with all charts, metrics, and tables saved inline.

---

## Author
**Anshu Mishra**  
Machine Learning Intern — Cognifyz Technologies
