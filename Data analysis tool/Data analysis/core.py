import io
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OrdinalEncoder

warnings.filterwarnings('ignore')

class PlottingMethods:
    """A dedicated class for granular chart generation."""
    @staticmethod
    def generate_bar_chart(df, column, title=None):
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, 'count']
        counts['percentage'] = (counts['count'] / counts['count'].sum() * 100).round(2)
        fig = px.bar(
            counts, x=column, y='count',
            text=counts.apply(lambda r: f"{r['count']} ({r['percentage']}%)", axis=1),
            title=title or f"Frequency Distribution of {column}"
        )
        fig.update_traces(textposition='outside')
        return fig

    @staticmethod
    def generate_pie_chart(df, column, title=None):
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, 'count']
        return px.pie(counts, names=column, values='count', title=title or f"Pie Chart of {column}")

    @staticmethod
    def generate_histogram(df, column, bins=None, title=None):
        return px.histogram(df, x=column, nbins=bins, title=title or f"Histogram of {column}", marginal="rug")


class DataInspector:
    """A reusable data science engine for data assignment tracking and preparation."""
    def __init__(self):
        self.df = None

    def upload_data(self):
        try:
            from google.colab import files
            uploaded = files.upload()
            if not uploaded: return None
            file_name = list(uploaded.keys())[0]
            garbage_strings = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ', '']
            self.df = pd.read_csv(io.BytesIO(uploaded[file_name]), na_values=garbage_strings)
            self._auto_type_correction()
            return self.df
        except ImportError:
            print("Google Colab environment not detected.")

    def load_dataframe(self, dataframe):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a valid pandas DataFrame.")
        garbage_strings = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ', '']
        self.df = dataframe.replace(garbage_strings, np.nan)
        self._auto_type_correction()

    def _auto_type_correction(self):
        if self.df is None: return
        for col in self.df.columns:
            converted = pd.to_numeric(self.df[col], errors='coerce')
            if not converted.isna().all() and converted.isna().sum() < len(self.df):
                self.df[col] = converted

    def display_summary(self):
        if self.df is None: return
        print(f"DATASET SUMMARY: {self.df.shape[0]} Rows | {self.df.shape[1]} Columns")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        print(f"Numerical: {numeric_cols}\nCategorical: {categorical_cols}")
        display(self.df.head(20))

    def handle_missing_values(self, columns, strategy='median', constant_value=None):
        if self.df is None: return
        if isinstance(columns, str): columns = [columns]
        for col in columns:
            if col not in self.df.columns: continue
            if strategy == 'mean': self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif strategy == 'median': self.df[col].fillna(self.df[col].median(), inplace=True)
            elif strategy == 'mode': self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            elif strategy == 'constant': self.df[col].fillna(constant_value, inplace=True)

    def remove_duplicates(self):
        if self.df is None: return
        self.df.drop_duplicates(inplace=True)

    def handle_outliers(self, columns, action='flag'):
        if self.df is None: return
        if isinstance(columns, str): columns = [columns]
        for col in columns:
            if col not in self.df.columns or not np.issubdtype(self.df[col].dtype, np.number): continue
            q1, q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            iqr = q3 - q1
            mask = (self.df[col] < (q1 - 1.5 * iqr)) | (self.df[col] > (q3 + 1.5 * iqr))
            if action == 'flag': self.df[f'{col}_outlier'] = mask
            elif action == 'delete': self.df = self.df[~mask]

    def delete_rows(self, indices_string):
        if self.df is None: return
        indices = [int(idx.strip()) for idx in indices_string.split(',') if idx.strip()]
        self.df.drop(index=indices, inplace=True, errors='ignore')

    def delete_columns(self, columns_string):
        if self.df is None: return
        cols = [c.strip() for c in columns_string.split(',') if c.strip()]
        self.df.drop(columns=cols, inplace=True, errors='ignore')

    def extract_normalized_numeric_data(self, columns, method='standard'):
        if self.df is None: return pd.DataFrame()
        subset = self.df[columns].copy()
        scaler = MinMaxScaler() if method == 'minmax' else StandardScaler() if method == 'standard' else RobustScaler()
        return pd.DataFrame(scaler.fit_transform(subset), columns=[f"{c}_{method}" for c in columns], index=self.df.index)

    def extract_normalized_categorical_data(self, columns, encoding='onehot'):
        if self.df is None: return pd.DataFrame()
        subset = self.df[columns].copy().astype(str)
        if encoding == 'onehot': return pd.get_dummies(subset, columns=columns, drop_first=True, dtype=float)
        encoder = OrdinalEncoder()
        encoded = encoder.fit_transform(subset)
        if encoding == 'uniform': encoded = MinMaxScaler().fit_transform(encoded)
        return pd.DataFrame(encoded, columns=[f"{c}_{encoding}" for c in columns], index=self.df.index)

    def assemble_engineered_dataset(self, numeric_cols, categorical_cols, num_method='standard', cat_encoding='onehot'):
        df_num = self.extract_normalized_numeric_data(numeric_cols, method=num_method)
        df_cat = self.extract_normalized_categorical_data(categorical_cols, encoding=cat_encoding)
        return pd.concat([df_num, df_cat], axis=1)

    def plot_univariate_subplots(self, column):
        if self.df is None or column not in self.df.columns: return
        fig = make_subplots(rows=3, cols=1, subplot_titles=(f"Violin", f"Scatter", f"Histogram"))
        fig.add_trace(go.Violin(x=self.df[column], box_visible=True, name=column), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df[column], mode='markers'), row=2, col=1)
        fig.add_trace(go.Histogram(x=self.df[column]), row=3, col=1)
        fig.update_layout(height=800, showlegend=False)
        fig.show()

    def plot_relationship(self, col_x, col_y):
        if self.df is None: return
        is_x_num = np.issubdtype(self.df[col_x].dtype, np.number)
        is_y_num = np.issubdtype(self.df[col_y].dtype, np.number)
        if is_x_num and is_y_num:
            fig = px.scatter(self.df, x=col_x, y=col_y, trendline="ols")
        elif (not is_x_num and is_y_num) or (is_x_num and not is_y_num):
            cat, num = (col_x, col_y) if not is_x_num else (col_y, col_x)
            fig = px.box(self.df, x=cat, y=num, points="all")
        else:
            counts = self.df.groupby([col_x, col_y]).size().reset_index(name='counts')
            fig = px.bar(counts, x=col_x, y='counts', color=col_y, barmode='group')
        fig.show()

    def plot_all_associations_heatmap(self):
        if self.df is None: return
        cols = [c for c in self.df.columns if not c.endswith('_outlier')]
        n = len(cols)
        corr_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    continue
                col1, col2 = cols[i], cols[j]
                temp_df = self.df[[col1, col2]].dropna()
                if len(temp_df) < 5: continue
                is_1_num = np.issubdtype(self.df[col1].dtype, np.number)
                is_2_num = np.issubdtype(self.df[col2].dtype, np.number)
                
                if is_1_num and is_2_num:
                    r, _ = stats.pearsonr(temp_df[col1], temp_df[col2])
                    corr_matrix[i, j] = np.nan_to_num(r)
                elif not is_1_num and not is_2_num:
                    confusion_matrix = pd.crosstab(temp_df[col1], temp_df[col2])
                    chi2 = stats.chi2_contingency(confusion_matrix)[0]
                    total_obs = confusion_matrix.sum().sum()
                    phi2 = chi2 / total_obs
                    r, c = confusion_matrix.shape
                    k = c
                    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(total_obs-1))
                    rcorr, ccorr = r - ((r-1)**2)/(total_obs-1), c - ((c-1)**2)/(total_obs-1)
                    denom = min((rcorr-1), (ccorr-1))
                    corr_matrix[i, j] = np.sqrt(phi2corr / denom) if denom > 0 else 0.0
                else:
                    num_col, cat_col = (col1, col2) if is_1_num else (col2, col1)
                    groups = [group[num_col].values for name, group in temp_df.groupby(cat_col)]
                    if len(groups) > 1:
                        f_val, _ = stats.f_oneway(*groups)
                        df1, df2 = len(groups) - 1, len(temp_df) - len(groups)
                        corr_matrix[i, j] = np.sqrt((f_val * df1) / (f_val * df1 + df2)) if (f_val * df1 + df2) > 0 else 0.0
                        
        fig = px.imshow(corr_matrix, x=cols, y=cols, color_continuous_scale='RdBu', zmin=-1.0, zmax=1.0)
        fig.show()
