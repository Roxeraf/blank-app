import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

# Funktion zum Laden der CSV-Datei
@st.cache_data
def load_csv_data(file):
    return pd.read_csv(file)

def main():
    st.title("CSV Datenanalyse App")

    # File uploader
    uploaded_file = st.file_uploader("Wählen Sie Ihre CSV-Datei", type="csv")
    
    if uploaded_file is not None:
        # Daten laden
        df = load_csv_data(uploaded_file)
        st.success("Daten erfolgreich geladen!")

        # Datenübersicht
        st.header("Datenübersicht")
        st.write(df.head())
        st.write(df.describe())

        # Informationen zur Datei
        st.subheader("Dateiinformationen")
        st.write(f"Anzahl der Zeilen: {df.shape[0]}")
        st.write(f"Anzahl der Spalten: {df.shape[1]}")

        # Spaltenauswahl für Analyse
        columns = df.columns.tolist()
        selected_columns = st.multiselect("Wählen Sie Spalten für die Analyse", columns, default=columns[:5])

        if selected_columns:
            df_selected = df[selected_columns]

            # Datentypen der ausgewählten Spalten
            st.subheader("Datentypen der ausgewählten Spalten")
            st.write(df_selected.dtypes)

            # Korrelationsmatrix
            st.subheader("Korrelationsmatrix")
            numeric_df = df_selected.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                corr_matrix = numeric_df.corr()
                fig_corr = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', aspect="auto")
                st.plotly_chart(fig_corr)
            else:
                st.write("Keine numerischen Spalten für Korrelationsanalyse verfügbar.")

            # Histogramme
            st.subheader("Histogramme")
            for col in numeric_df.columns:
                fig = px.histogram(df_selected, x=col, title=f'Histogram von {col}')
                st.plotly_chart(fig)

            # Streudiagramm
            st.subheader("Streudiagramm")
            if len(numeric_df.columns) >= 2:
                x_axis = st.selectbox("Wählen Sie die X-Achse", numeric_df.columns)
                y_axis = st.selectbox("Wählen Sie die Y-Achse", numeric_df.columns)
                fig_scatter = px.scatter(df_selected, x=x_axis, y=y_axis, title=f'{x_axis} vs {y_axis}')
                st.plotly_chart(fig_scatter)
            else:
                st.write("Nicht genügend numerische Spalten für ein Streudiagramm.")

            # Zeitreihenanalyse (falls zutreffend)
            date_columns = df_selected.select_dtypes(include=['datetime64']).columns
            if not date_columns.empty:
                st.subheader("Zeitreihenanalyse")
                date_column = st.selectbox("Wählen Sie die Datumsspalte", date_columns)
                value_column = st.selectbox("Wählen Sie die Wertspalte", numeric_df.columns)
                
                df_selected[date_column] = pd.to_datetime(df_selected[date_column])
                df_time = df_selected.set_index(date_column)
                daily_average = df_time[value_column].resample('D').mean()
                
                fig_time = px.line(daily_average, title=f'Täglicher Durchschnitt von {value_column}')
                st.plotly_chart(fig_time)

            # Kategorische Datenanalyse
            cat_columns = df_selected.select_dtypes(include=['object']).columns
            if not cat_columns.empty:
                st.subheader("Kategorische Datenanalyse")
                cat_column = st.selectbox("Wählen Sie eine kategorische Spalte", cat_columns)
                fig_bar = px.bar(df_selected[cat_column].value_counts(), title=f'Häufigkeitsverteilung: {cat_column}')
                st.plotly_chart(fig_bar)

            # Statistische Tests
            st.subheader("Statistische Tests")
            if len(numeric_df.columns) >= 2:
                test_col1 = st.selectbox("Wählen Sie die erste Spalte für den T-Test", numeric_df.columns, key='t_test1')
                test_col2 = st.selectbox("Wählen Sie die zweite Spalte für den T-Test", numeric_df.columns, key='t_test2')
                t_stat, p_value = stats.ttest_ind(df_selected[test_col1].dropna(), df_selected[test_col2].dropna())
                st.write(f"T-Test Ergebnis zwischen {test_col1} und {test_col2}:")
                st.write(f"t-statistic: {t_stat}")
                st.write(f"p-value: {p_value}")

            # Datenexport-Option
            if st.button("Analysierte Daten exportieren"):
                csv = df_selected.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="analysierte_daten.csv",
                    mime="text/csv",
                )

if __name__ == "__main__":
    main()