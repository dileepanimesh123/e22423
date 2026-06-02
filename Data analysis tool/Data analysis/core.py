import io
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OrdinalEncoder

# Suppress warnings for clean output
warnings.filterwarnings('ignore')


class PlottingMethods:
    """
    A dedicated class for granular chart generation that returns 
    HTML-wrapped or standalone Plotly figures for flexible embedding.
    """

    @staticmethod
    def generate_bar_chart(df, column, title=None):
        """Generates a bar chart with raw counts and percentage labels."""
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, 'count']
        counts['percentage'] = (counts['count'] / counts['count'].sum() * 100).round(2)
        
        fig = px.bar(
            counts, x=column, y='count',
            text=counts.apply(lambda r: f"{r['count']} ({r['percentage']}%)", axis=1),
            title=title or f"Frequency Distribution of {column}",
            labels={column: str(column), 'count': 'Count'}
        )
        fig.update_traces(textposition='outside')
        return fig

    @staticmethod
    def generate_pie_chart(df, column, title=None):
        """Generates a standard interactive pie chart."""
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, 'count']
        fig = px.pie(counts, names=column, values='count', title=title or f"Pie Chart of {column}")
        return fig

    @staticmethod
    def generate_histogram(df, column, bins=None, title=None):
        """Generates a classic histogram for a numeric column."""
        fig = px.histogram(df, x=column, nbins=bins, title=title or f"Histogram of {column}", marginal="rug")
        return fig


class DataInspector:
    """
    A reusable data science engine for CSV data ingestion, advanced cleaning,
    feature engineering normalization, and interactive statistical visualization.
    """

    def __init__(self):
        """Initializes the DataInspector with an empty DataFrame container."""
        self.df = None

    def upload_data(self):
        """
        Handles local file uploads directly within Google Colab environments.
        Automatically converts typical garbage strings into NaN values.
        """
        try:
            from google.colab import files
            uploaded = files.upload()
            if not uploaded:
                print("No file uploaded.")
                return None
            
            # Take the first uploaded file
            file_name = list(uploaded.keys())[0]
            garbage_strings = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ', '']
            
            self.df = pd.read_csv(io.BytesIO(uploaded[file_name]), na_values=garbage_strings)
            print(f"Successfully loaded '{file_name}' into engine.")
            self._auto_type_correction()
            return self.df
        except ImportError:
            print("Google Colab environments environment not detected. Please load data using load_dataframe(df) manually.")

    def load_dataframe(self, dataframe):
        """Manually loads an existing Pandas DataFrame and standardizes garbage strings."""
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a valid pandas DataFrame.")
        
        garbage_strings = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ', '']
        self.df = dataframe.replace(garbage_strings, np.nan)
        self._auto_type_correction()
        print("DataFrame loaded successfully.")

    def _auto_type_correction(self):
        """Force-converts columns to numeric types if it doesn't result in an entirely null column."""
        if self.df is None: return
        
        for col in self.df.columns:
            # Try to convert to numeric safely
            converted = pd.to_numeric(self.df[col], errors='coerce')
            # If the whole column didn't convert to NaN (or if it was already empty)
            if not converted.isna().all() and converted.isna().sum() < len(self.df):
                self.df[col] = converted

    def display_summary(self):
        """Displays data shape structural dimensions, a 20-row preview, and schema layouts."""
        if self.df is None:
            print("No dataset loaded.")
            return

        print("="*60)
        print(f"DATASET SUMMARY: {self.df.shape[0]} Rows | {self.df.shape[1]} Columns")
        print("="*60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print(f"Numerical Columns ({len(numeric_cols)}): {numeric_cols}")
        print(f"Categorical Columns ({len(categorical_cols)}): {categorical_cols}\n")
        
        print("Missing Values per Column:")
        print(self.df.isna().sum())
        print("-" * 60)
        print("First 20 Rows Preview:")
        display(self.df.head(20))

    def handle_missing_values(self, columns, strategy='median', constant_value=None):
        """
        Imputes missing items using strategies: 'mean', 'median', 'mode', or 'constant'.
        """
        if self.df is None: return
        
        if isinstance(columns, str):
            columns = [columns]
            
        for col in columns:
            if col not in self.df.columns:
                print(f"Column '{col}' not found. Skipping.")
                continue
                
            if strategy == 'mean':
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif strategy == 'median':
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif strategy == 'mode':
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            elif strategy == 'constant':
                self.df[col].fillna(constant_value, inplace=True)
            else:
                raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'constant'.")
        print(f"Imputation done using '{strategy}' strategy.")

    def remove_duplicates(self):
        """Prunes exact row duplicates from the current active DataFrame."""
        if self.df is None: return
        initial_count = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Removed {initial_count - len(self.df)} duplicate rows.")

    def handle_outliers(self, columns, action='flag'):
        """
        Uses Tukey's IQR Method to identify anomalies. 
        Supported actions: 'flag' (adds a boolean outlier column) or 'delete'.
        """
        if self.df is None: return
        if isinstance(columns, str):
            columns = [columns]
            
        for col in columns:
            if col not in self.df.columns or not np.issubdtype(self.df[col].dtype, np.number):
                print(f"Skipping non-numeric or missing column: {col}")
                continue
                
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            
            if action == 'flag':
                self.df[f'{col}_outlier'] = mask
                print(f"Flagged {mask.sum()} anomalies in '{col}'.")
            elif action == 'delete':
                initial_len = len(self.df)
                self.df = self.df[~mask]
                print(f"Deleted {initial_len - len(self.df)} outlier rows based on '{col}'.")

    def delete_rows(self, indices_string):
        """Accepts comma-separated row indexes (e.g., '3, 5, 12') and drops them."""
        if self.df is None: return
        try:
            indices = [int(idx.strip()) for idx in indices_string.split(',') if idx.strip()]
            self.df.drop(index=indices, inplace=True, errors='ignore')
            print(f"Successfully dropped specified rows.")
        except Exception as e:
            print(f"Error parsing row deletion array: {e}")

    def delete_columns(self, columns_string):
        """Accepts comma-separated column names (e.g., 'PassengerId, Name') and drops them."""
        if self.df is None: return
        cols = [c.strip() for c in columns_string.split(',') if c.strip()]
        self.df.drop(columns=cols, inplace=True, errors='ignore')
        print(f"Successfully dropped columns: {cols}")

    def extract_normalized_numeric_data(self, columns, method='standard'):
        """Scales continuous features using 'minmax', 'standard', or 'robust' options."""
        if self.df is None: return pd.DataFrame()
        
        subset = self.df[columns].copy()
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'minmax', 'standard', or 'robust'.")
            
        scaled_array = scaler.fit_transform(subset)
        return pd.DataFrame(scaled_array, columns=[f"{c}_{method}" for c in columns], index=self.df.index)

    def extract_normalized_categorical_data(self, columns, encoding='onehot'):
        """Encodes non-numeric keys using 'onehot', 'ordinal', or 'uniform' metrics."""
        if self.df is None: return pd.DataFrame()
        
        subset = self.df[columns].copy().astype(str)
        
        if encoding == 'onehot':
            return pd.get_dummies(subset, columns=columns, drop_first=True, dtype=float)
        elif encoding == 'ordinal':
            encoder = OrdinalEncoder()
            encoded_array = encoder.fit_transform(subset)
            return pd.DataFrame(encoded_array, columns=[f"{c}_encoded" for c in columns], index=self.df.index)
        elif encoding == 'uniform':
            # Encodes ordinally, then forces scaling to exact 0-1 metrics
            encoder = OrdinalEncoder()
            encoded_array = encoder.fit_transform(subset)
            scaled_array = MinMaxScaler().fit_transform(encoded_array)
            return pd.DataFrame(scaled_array, columns=[f"{c}_uniform" for c in columns], index=self.df.index)
        else:
            raise ValueError("Encoding must be 'onehot', 'ordinal', or 'uniform'.")

    def assemble_engineered_dataset(self, numeric_cols, categorical_cols, num_method='standard', cat_encoding='onehot'):
        """Merges scaled numerical matrices and encoded categorical fields together into a unified frame."""
        df_num = self.extract_normalized_numeric_data(numeric_cols, method=num_method)
        df_cat = self.extract_normalized_categorical_data(categorical_cols, encoding=cat_encoding)
        return pd.concat([df_num, df_cat], axis=1)

    def plot_univariate_subplots(self, column):
        """Generates an advanced 3-panel subplot layout for checking continuous distributions."""
        if self.df is None or column not in self.df.columns: return
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(f"Violin Plot of {column}", f"Index vs Value Scatter of {column}", f"Histogram of {column}"),
            vertical_spacing=0.1
        )
        
        # Panel 1: Violin
        fig.add_trace(go.Violin(x=self.df[column], box_visible=True, meanline_visible=True, name=column), row=1, col=1)
        # Panel 2: Scatter
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df[column], mode='markers', name='Data Point'), row=2, col=1)
        # Panel 3: Histogram
        fig.add_trace(go.Histogram(x=self.df[column], name='Count'), row=3, col=1)
        
        fig.update_layout(height=800, title_text=f"Advanced Diagnostic Portfolio: {column}", showlegend=False)
        fig.show()

    def plot_relationship(self, col_x, col_y):
        """Automatically evaluates data types and renders the optimal statistical chart match."""
        if self.df is None: return
        
        is_x_num = np.issubdtype(self.df[col_x].dtype, np.number)
        is_y_num = np.issubdtype(self.df[col_y].dtype, np.number)
        
        # Num-Num Correlation Matrix
        if is_x_num and is_y_num:
            fig = px.scatter(self.df, x=col_x, y=col_y, trendline="ols", title=f"Scatter Analysis: {col_x} vs {col_y}")
        
        # Cat-Num Distributions
        elif (not is_x_num and is_y_num) or (is_x_num and not is_y_num):
            cat, num = (col_x, col_y) if not is_x_num else (col_y, col_x)
            fig = px.box(self.df, x=cat, y=num, points="all", title=f"Box Distribution: {num} grouped by {cat}")
            
        # Cat-Cat Grouped Metrics
        else:
            counts = self.df.groupby([col_x, col_y]).size().reset_index(name='counts')
            fig = px.bar(counts, x=col_x, y='counts', color=col_y, barmode='group', title=f"Cross-Tabulation: {col_x} vs {col_y}")
            
        fig.show()

    def plot_all_associations_heatmap(self):
        """
        Computes a unified association heatmap across mixed data tables using:
        - Numeric vs Numeric: Pearson's r
        - Categorical vs Categorical: Cramér's V
        - Mixed (Num vs Cat): Correlation Ratio (Eta) via ANOVA
        """
        if self.df is None: return
        
        # Filter down columns to avoid heavy loops over high cardinality
        cols = [c for c in self.df.columns if not c.endswith('_outlier')]
        n = len(cols)
        corr_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    continue
                
                col1, col2 = cols[i], cols[j]
                is_1_num = np.issubdtype(self.df[col1].dtype, np.number)
                is_2_num = np.issubdtype(self.df[col2].dtype, np.number)
                
                # Dropping missing pairs locally for valid statistical evaluations
                temp_df = self.df[[col1, col2]].dropna()
                if len(temp_df) < 5:
                    corr_matrix[i, j] = 0.0
                    continue
                
                if is_1_num and is_2_num:
                    # Pearson Correlation Coefficient
                    r, _ = stats.pearsonr(temp_df[col1], temp_df[col2])
                    corr_matrix[i, j] = np.nan_to_num(r)
                    
                elif not is_1_num and not is_2_num:
                    # Cramér's V Coefficient
                    confusion_matrix = pd.crosstab(temp_df[col1], temp_df[col2])
                    chi2 = stats.chi2_contingency(confusion_matrix)[0]
                    total_obs = confusion_matrix.sum().sum()
                    phi2 = chi2 / total_obs
                    r, c = confusion_matrix.shape
                    # Bias correction adjustments
                    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(total_obs-1)) if (k:=c) else phi2
                    rcorr = r - ((r-1)**2)/(total_obs-1)
                    ccorr = c - ((c-1)**2)/(total_obs-1)
                    denom = min((rcorr-1), (ccorr-1))
                    corr_matrix[i, j] = np.sqrt(phi2corr / denom) if denom > 0 else 0.0
                    
                else:
                    # Mixed Categorical vs Numeric -> Correlation Ratio (Eta)
                    num_col, cat_col = (col1, col2) if is_1_num else (col2, col1)
                    groups = [group[num_col].values for name, group in temp_df.groupby(cat_col)]
                    if len(groups) > 1 and sum(len(g) for g in groups) > len(groups):
                        f_val, _ = stats.f_oneway(*groups)
                        # Derive Eta squared from ANOVA F-value elements
                        # Eta = sqrt( (F * df1) / (F * df1 + df2) )
                        df1 = len(groups) - 1
                        df2 = len(temp_df) - len(groups)
                        if (f_val * df1 + df2) > 0:
                            eta = np.sqrt((f_val * df1) / (f_val * df1 + df2))
                            corr_matrix[i, j] = np.nan_to_num(eta)
                        else:
                            corr_matrix[i, j] = 0.0
                    else:
                        corr_matrix[i, j] = 0.0
                        
        fig = px.imshow(
            corr_matrix, x=cols, y=cols, 
            color_continuous_scale='RdBu', aspect="auto", zmin=-1.0, zmax=1.0,
            title="Unified Association Heatmap (Pearson r | Cramér's V | Correlation Ratio Eta)"
        )
        fig.show()
