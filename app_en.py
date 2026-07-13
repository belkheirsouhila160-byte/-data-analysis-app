import streamlit as st
import pandas as pd
from scipy import stats
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(page_title="Statistical Analysis Application", page_icon="📊", layout="wide")
st.title("Statistical Analysis Application")

section = st.sidebar.radio(
    "📌 Analysis Sections",
    [
        "📁 Data",
        "ℹ️ Data Information",
        "📊 Descriptive Statistics",
        "🔔 Normality Tests",
        "🔗 Correlation Matrix",
        "📈 Linear Regression",
    ],
)

if section == "📁 Data":
    uploaded_file = st.file_uploader("Upload a data file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                data = pd.read_csv(uploaded_file, sep=None, engine="python")
            else:
                data = pd.read_excel(uploaded_file)
            st.session_state["data"] = data
            st.success("File uploaded successfully")
            st.subheader("👁️ Data Preview")
            st.dataframe(data, use_container_width=True)
        except Exception as error:
            st.error(f"Unable to read the file: {error}")

elif section == "ℹ️ Data Information":
    data = st.session_state.get("data")
    if data is None:
        st.warning("Please upload a dataset first from the 📁 Data section.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Number of Rows", data.shape[0])
        col2.metric("📊 Number of Columns", data.shape[1])
        col3.metric("⚠️ Missing Values", int(data.isna().sum().sum()))

        missing_by_column = data.isna().sum()
        missing_table = pd.DataFrame({
            "Variable": missing_by_column.index,
            "Number of Missing Values": missing_by_column.values,
            "Percentage %": (missing_by_column.values / len(data) * 100).round(2),
        })
        st.subheader("⚠️ Missing Values by Variable")
        st.dataframe(
            missing_table,
            column_config={"Variable": st.column_config.TextColumn(width="large")},
            hide_index=True,
            use_container_width=True,
        )

        variable_types = data.dtypes.astype(str).rename("Variable Type").reset_index()
        variable_types.columns = ["Variable", "Variable Type"]
        st.subheader("🔤 Variable Types")
        st.dataframe(variable_types, hide_index=True, use_container_width=True)

        numeric_data = data.select_dtypes(include="number")
        st.metric("🔢 Number of Numeric Variables", numeric_data.shape[1])
        csv_file = numeric_data.to_csv(index=False, sep=";").encode("utf-8-sig")
        st.download_button(
            "⬇️ Download Numeric Data as CSV",
            csv_file,
            "numeric_data.csv",
            "text/csv",
            key="numeric_csv",
        )
        excel_file = BytesIO()
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            numeric_data.to_excel(writer, index=False, sheet_name="Numeric Data")
        st.download_button(
            "⬇️ Download Numeric Data as Excel",
            excel_file.getvalue(),
            "numeric_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="numeric_excel",
        )

elif section == "📊 Descriptive Statistics":
    data = st.session_state.get("data")
    if data is None:
        st.warning("Please upload a dataset first from the 📁 Data section.")
    else:
        st.subheader("📊 Descriptive Statistics")
        numeric_data = data.select_dtypes(include="number")
        if numeric_data.empty:
            st.warning("No numeric variables were found in the dataset.")
        else:
            selected_variables = st.multiselect(
                "Select numeric variables",
                numeric_data.columns.tolist(),
                default=numeric_data.columns.tolist(),
            )
            if not selected_variables:
                st.warning("Please select at least one variable.")
            else:
                selected_data = numeric_data[selected_variables]
                descriptive_table = selected_data.describe().T.rename(columns={
                    "count": "Count", "mean": "Mean",
                    "std": "Standard Deviation", "min": "Minimum",
                    "25%": "First Quartile", "50%": "Median",
                    "75%": "Third Quartile", "max": "Maximum",
                })
                modes = selected_data.mode(dropna=True)
                descriptive_table["Mode"] = modes.iloc[0] if not modes.empty else pd.NA
                descriptive_table["Variance"] = selected_data.var()
                descriptive_table["Range"] = selected_data.max() - selected_data.min()
                descriptive_table["Skewness"] = selected_data.skew()
                descriptive_table["Kurtosis"] = selected_data.kurt()
                descriptive_table["Coefficient of Variation %"] = selected_data.std().div(selected_data.mean()).mul(100)
                descriptive_table = descriptive_table.round(2)
                descriptive_table.index.name = "Variable"
                st.dataframe(descriptive_table, use_container_width=True)

                descriptive_excel = BytesIO()
                with pd.ExcelWriter(descriptive_excel, engine="openpyxl") as writer:
                    descriptive_table.to_excel(writer, sheet_name="Statistics")
                st.download_button(
                    "⬇️ Download Descriptive Statistics as Excel",
                    descriptive_excel.getvalue(),
                    "descriptive_statistics.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="descriptive_excel",
                )

elif section == "🔔 Normality Tests":
    data = st.session_state.get("data")
    if data is None:
        st.warning("Please upload a dataset first from the 📁 Data section.")
    else:
        st.subheader("🔔 Normality Tests")
        numeric_columns = data.select_dtypes(include="number").columns.tolist()
        if not numeric_columns:
            st.warning("No numeric variables were found in the dataset.")
        else:
            selected_variable = st.selectbox("Select the variable to test", numeric_columns)
            values = data[selected_variable].dropna()
            if len(values) < 3:
                st.warning("The variable must contain at least three valid values.")
            else:
                shapiro_values = values if len(values) <= 5000 else values.sample(5000, random_state=42)
                shapiro_stat, shapiro_p = stats.shapiro(shapiro_values)
                jarque_stat, jarque_p = stats.jarque_bera(values)
                normality_table = pd.DataFrame({
                    "Test": ["Shapiro-Wilk", "Jarque-Bera"],
                    "Test Statistic": [shapiro_stat, jarque_stat],
                    "p-value": [shapiro_p, jarque_p],
                    "Result": [
                        "Normal distribution" if shapiro_p >= 0.05 else "Non-normal distribution",
                        "Normal distribution" if jarque_p >= 0.05 else "Non-normal distribution",
                    ],
                }).round({"Test Statistic": 4, "p-value": 4})
                st.subheader("📋 Test Results")
                st.dataframe(normality_table, hide_index=True, use_container_width=True)

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                sns.histplot(values, kde=True, ax=axes[0], color="#2E86C1")
                axes[0].set_title(f"Distribution: {selected_variable}")
                stats.probplot(values, dist="norm", plot=axes[1])
                axes[1].set_title("Normal Q-Q Plot")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("📝 Automatic Interpretation")
                if shapiro_p >= 0.05:
                    st.success("According to the Shapiro-Wilk test, the normality assumption is not rejected.")
                else:
                    st.error("According to the Shapiro-Wilk test, the normality assumption is rejected.")
                if jarque_p >= 0.05:
                    st.success("According to the Jarque-Bera test, the normality assumption is not rejected.")
                else:
                    st.error("According to the Jarque-Bera test, the normality assumption is rejected.")
                st.info("Decision rule: reject normality when the p-value is below 0.05.")
                if len(values) > 5000:
                    st.warning("A fixed sample of 5,000 observations was used for the Shapiro-Wilk test.")

elif section == "🔗 Correlation Matrix":
    data = st.session_state.get("data")
    if data is None:
        st.warning("Please upload a dataset first from the 📁 Data section.")
    else:
        st.subheader("🔗 Correlation Matrix and Heatmap")
        numeric_data = data.select_dtypes(include="number")
        numeric_columns = numeric_data.columns.tolist()

        if len(numeric_columns) < 2:
            st.warning("The dataset must contain at least two numeric variables.")
        else:
            selected_correlation_variables = st.multiselect(
                "Select numeric variables",
                numeric_columns,
                default=numeric_columns,
                key="correlation_variables",
            )
            correlation_method_label = st.selectbox(
                "Select the correlation method",
                ["Pearson", "Spearman", "Kendall"],
            )

            if len(selected_correlation_variables) < 2:
                st.warning("Please select at least two variables.")
            else:
                method_map = {
                    "Pearson": "pearson",
                    "Spearman": "spearman",
                    "Kendall": "kendall",
                }
                correlation_matrix = numeric_data[
                    selected_correlation_variables
                ].corr(method=method_map[correlation_method_label]).round(3)

                st.subheader("📋 Correlation Matrix")
                st.dataframe(
                    correlation_matrix.style.background_gradient(
                        cmap="RdBu_r", vmin=-1, vmax=1
                    ).format("{:.3f}"),
                    use_container_width=True,
                )

                figure_width = max(8, len(selected_correlation_variables) * 1.1)
                figure_height = max(6, len(selected_correlation_variables) * 0.85)
                fig, ax = plt.subplots(figsize=(figure_width, figure_height))
                sns.heatmap(
                    correlation_matrix,
                    annot=True,
                    fmt=".2f",
                    cmap="RdBu_r",
                    vmin=-1,
                    vmax=1,
                    center=0,
                    square=True,
                    linewidths=0.5,
                    ax=ax,
                )
                ax.set_title(f"{correlation_method_label} Correlation Heatmap")
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.subheader("🌡️ Heatmap")
                st.pyplot(fig)
                plt.close(fig)

                correlation_pairs = correlation_matrix.where(
                    ~pd.DataFrame(
                        False,
                        index=correlation_matrix.index,
                        columns=correlation_matrix.columns,
                    )
                )
                upper_mask = pd.DataFrame(
                    False,
                    index=correlation_matrix.index,
                    columns=correlation_matrix.columns,
                )
                for row_index in range(len(correlation_matrix.columns)):
                    upper_mask.iloc[row_index, row_index + 1:] = True
                correlation_pairs = correlation_pairs.where(upper_mask).stack()

                st.subheader("📝 Automatic Interpretation")
                if correlation_pairs.empty:
                    st.info("There are not enough variable pairs to interpret the correlation.")
                else:
                    strongest_pair = correlation_pairs.abs().idxmax()
                    strongest_value = correlation_pairs.loc[strongest_pair]
                    direction = "positive" if strongest_value > 0 else "negative"
                    absolute_value = abs(strongest_value)
                    if absolute_value >= 0.80:
                        strength = "very strong"
                    elif absolute_value >= 0.60:
                        strength = "strong"
                    elif absolute_value >= 0.40:
                        strength = "moderate"
                    elif absolute_value >= 0.20:
                        strength = "weak"
                    else:
                        strength = "very weak"
                    st.info(
                        f"The strongest correlation is between {strongest_pair[0]} and "
                        f"{strongest_pair[1]} with a value of {strongest_value:.3f}. "
                        f"It is a {direction} {strength}."
                    )
                    st.caption("Correlation does not establish causation between variables.")

                correlation_excel = BytesIO()
                with pd.ExcelWriter(correlation_excel, engine="openpyxl") as writer:
                    correlation_matrix.to_excel(writer, sheet_name="Correlation Matrix")
                st.download_button(
                    "⬇️ Download Correlation Matrix as Excel",
                    correlation_excel.getvalue(),
                    "correlation_matrix.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="correlation_excel",
                )

elif section == "📈 Linear Regression":
    data = st.session_state.get("data")
    if data is None:
        st.warning("Please upload a dataset first from the 📁 Data section.")
    else:
        st.subheader("📈 Simple and Multiple Linear Regression")
        numeric_data = data.select_dtypes(include="number")
        numeric_columns = numeric_data.columns.tolist()

        if len(numeric_columns) < 2:
            st.warning("The dataset must contain at least two numeric variables.")
        else:
            regression_type = st.radio(
                "Select the regression type",
                ["Simple Linear Regression", "Multiple Linear Regression"],
                horizontal=True,
            )
            dependent = st.selectbox("Dependent Variable (Y)", numeric_columns)
            available_predictors = [column for column in numeric_columns if column != dependent]

            if regression_type == "Simple Linear Regression":
                independent = st.selectbox("Independent Variable (X)", available_predictors)
                predictors = [independent]
            else:
                predictors = st.multiselect(
                    "Independent Variables (X)",
                    available_predictors,
                    default=available_predictors[: min(2, len(available_predictors))],
                )

            if not predictors:
                st.warning("Please select at least one independent variable.")
            elif st.button("▶️ Run Regression Model", type="primary"):
                model_data = numeric_data[[dependent] + predictors].replace(
                    [float("inf"), float("-inf")], pd.NA
                ).dropna()

                if len(model_data) <= len(predictors) + 1:
                    st.error("There are not enough valid observations to estimate the model.")
                elif any(model_data[column].nunique() < 2 for column in predictors):
                    st.error("One independent variable is constant and cannot be included in the regression.")
                else:
                    y = model_data[dependent].astype(float)
                    x = sm.add_constant(model_data[predictors].astype(float))
                    model = sm.OLS(y, x).fit()

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Number of Observations", int(model.nobs))
                    col2.metric("R²", f"{model.rsquared:.4f}")
                    col3.metric("Adjusted R²", f"{model.rsquared_adj:.4f}")
                    col4.metric("F-test p-value", f"{model.f_pvalue:.4g}")

                    confidence = model.conf_int()
                    coefficients = pd.DataFrame({
                        "Variable": model.params.index,
                        "Coefficient B": model.params.values,
                        "Standard Error": model.bse.values,
                        "t-value": model.tvalues.values,
                        "p-value": model.pvalues.values,
                        "Lower 95% CI": confidence[0].values,
                        "Upper 95% CI": confidence[1].values,
                    }).round(4)
                    coefficients["Significance"] = coefficients["p-value"].apply(
                        lambda value: "Statistically significant" if value < 0.05 else "Not significant"
                    )
                    st.subheader("📋 Regression Coefficients")
                    st.dataframe(coefficients, hide_index=True, use_container_width=True)

                    residuals = model.resid
                    fitted = model.fittedvalues
                    jb_stat, jb_p = stats.jarque_bera(residuals)
                    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
                    dw_value = durbin_watson(residuals)
                    diagnostics = pd.DataFrame({
                        "Diagnostic": [
                            "Jarque-Bera for residuals",
                            "Breusch-Pagan for homoscedasticity",
                            "Durbin-Watson for residual independence",
                        ],
                        "Statistic": [jb_stat, bp_stat, dw_value],
                        "p-value": [jb_p, bp_p, pd.NA],
                    }).round(4)
                    st.subheader("🧪 Model Diagnostics")
                    st.dataframe(diagnostics, hide_index=True, use_container_width=True)

                    vif_table = None
                    if len(predictors) > 1:
                        vif_table = pd.DataFrame({
                            "Variable": predictors,
                            "VIF": [
                                variance_inflation_factor(model_data[predictors].astype(float).values, index)
                                for index in range(len(predictors))
                            ],
                        }).round(3)
                        st.subheader("🔗 Multicollinearity VIF")
                        st.dataframe(vif_table, hide_index=True, use_container_width=True)

                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    sns.scatterplot(x=fitted, y=residuals, ax=axes[0], color="#2E86C1")
                    axes[0].axhline(0, color="red", linestyle="--")
                    axes[0].set_title("Residuals vs Fitted")
                    axes[0].set_xlabel("Fitted values")
                    axes[0].set_ylabel("Residuals")
                    sm.qqplot(residuals, line="45", fit=True, ax=axes[1])
                    axes[1].set_title("Residual Q-Q Plot")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.subheader("📝 Automatic Interpretation")
                    if model.f_pvalue < 0.05:
                        st.success("The overall model is statistically significant at the 0.05 level.")
                    else:
                        st.warning("The overall model is not statistically significant at the 0.05 level.")
                    st.info(f"The model explains approximately {model.rsquared * 100:.2f}% of the variance in the dependent variable.")
                    if jb_p >= 0.05:
                        st.success("There is no statistical evidence that the residuals are non-normal.")
                    else:
                        st.warning("The Jarque-Bera result indicates that the residuals are not normally distributed.")
                    if bp_p >= 0.05:
                        st.success("There is no statistical evidence of heteroscedasticity.")
                    else:
                        st.warning("The Breusch-Pagan result indicates heteroscedasticity.")
                    if 1.5 <= dw_value <= 2.5:
                        st.success("The Durbin-Watson value is close to 2 and does not indicate strong autocorrelation.")
                    else:
                        st.warning("The Durbin-Watson value suggests that residual autocorrelation should be examined.")

                    regression_excel = BytesIO()
                    with pd.ExcelWriter(regression_excel, engine="openpyxl") as writer:
                        coefficients.to_excel(writer, index=False, sheet_name="Coefficients")
                        diagnostics.to_excel(writer, index=False, sheet_name="Diagnostics")
                        pd.DataFrame({
                            "Indicator": ["N", "R2", "Adjusted R2", "F", "F p-value", "AIC", "BIC"],
                            "Value": [
                                model.nobs, model.rsquared, model.rsquared_adj,
                                model.fvalue, model.f_pvalue, model.aic, model.bic,
                            ],
                        }).to_excel(writer, index=False, sheet_name="Model Summary")
                        if vif_table is not None:
                            vif_table.to_excel(writer, index=False, sheet_name="VIF")
                    st.download_button(
                        "⬇️ Download Regression Results as Excel",
                        regression_excel.getvalue(),
                        "regression_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="regression_excel",
                    )
