
# ============================================
# APP REGRESIÓN LINEAL MÚLTIPLE - GAC
# ============================================

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


# ============================================
# CARGA DE DATOS
# ============================================

@st.cache_resource
def load_data():

    df = pd.read_csv("Leads_Reales_Limpia.csv")

    return df


df = load_data()


# ============================================
# TÍTULO DEL DASHBOARD
# ============================================

st.sidebar.title("GAC")

View = st.sidebar.selectbox(
    label="Tipo de Análisis",
    options=[
        "Regresión Lineal"
    ]
)


# ============================================
# REGRESIÓN LINEAL
# ============================================

if View == "Regresión Lineal":

    # Seleccionamos únicamente variables numéricas
    numeric_df = df.select_dtypes(include=["float64", "int64", "int32"])

    Lista_num = numeric_df.columns


    # ----------------------------------------
    # VARIABLES
    # ----------------------------------------

    Variable_y = st.sidebar.selectbox(
        label="Variable objetivo (Y)",
        options=Lista_num,
        index=Lista_num.get_loc("Ventas")
    )


    Variable_x = st.sidebar.selectbox(
        label="Variable independiente del modelo simple (X)",
        options=Lista_num,
        index=Lista_num.get_loc("Efectivos")
    )


    Variables_x = st.sidebar.multiselect(
        label="Variables independientes del modelo múltiple (X)",
        options=Lista_num,
        default=[
            "Abiertos",
            "Efectivos",
            "Total",
            "Duplicados"
        ]
    )


    # ============================================
    # ENCABEZADO
    # ============================================

    st.title("Regresión Lineal - GAC")


    # Dos columnas
    Contenedor_A, Contenedor_B = st.columns(2)


    # ============================================
    # REGRESIÓN LINEAL SIMPLE
    # ============================================

    with Contenedor_A:

        st.subheader("Correlación Lineal Simple")

        model = LinearRegression()

        model.fit(
            X=df[[Variable_x]],
            y=df[Variable_y]
        )


        # Predicciones
        y_pred = model.predict(
            df[[Variable_x]]
        )


        # Coeficiente de determinación
        coef_Deter_simple = model.score(
            df[[Variable_x]],
            df[Variable_y]
        )


        # Coeficiente de correlación
        coef_Correl_simple = np.sqrt(
            coef_Deter_simple
        )


        st.write("Coeficiente de correlación:")

        st.write(
            round(coef_Correl_simple, 4)
        )


        st.write("R²:")

        st.write(
            round(coef_Deter_simple, 4)
        )


        # Gráfica
        figure1 = px.scatter(
            data_frame=numeric_df,
            x=Variable_x,
            y=Variable_y,
            trendline="ols",
            title="Modelo Lineal Simple"
        )


        st.plotly_chart(
            figure1,
            use_container_width=True
        )


    # ============================================
    # REGRESIÓN LINEAL MÚLTIPLE
    # ============================================

    with Contenedor_B:

        st.subheader("Correlación Lineal Múltiple")


        if len(Variables_x) >= 2:


            model_M = LinearRegression()


            model_M.fit(
                X=df[Variables_x],
                y=df[Variable_y]
            )


            # Predicción
            y_pred_M = model_M.predict(
                df[Variables_x]
            )


            # R cuadrado
            coef_Deter_multiple = model_M.score(
                df[Variables_x],
                df[Variable_y]
            )


            # Correlación múltiple
            coef_Correl_multiple = np.sqrt(
                coef_Deter_multiple
            )


            st.write(
                "Coeficiente de correlación múltiple:"
            )

            st.write(
                round(coef_Correl_multiple, 4)
            )


            st.write("R²:")

            st.write(
                round(coef_Deter_multiple, 4)
            )


            # ------------------------------------
            # COEFICIENTES
            # ------------------------------------

            st.write("Coeficientes del modelo")


            coeficientes = pd.DataFrame({

                "Variable": Variables_x,

                "Coeficiente":
                model_M.coef_

            })


            st.dataframe(
                coeficientes
            )


            st.write(
                "Intercepto:"
            )

            st.write(
                round(
                    model_M.intercept_,
                    4
                )
            )


            # ====================================
            # VENTAS REALES VS PREDICHAS
            # ====================================

            resultados = pd.DataFrame({

                "Ventas reales":
                df[Variable_y],

                "Ventas predichas":
                y_pred_M

            })


            figure2 = px.scatter(

                resultados,

                x="Ventas reales",

                y="Ventas predichas",

                title=
                "Modelo Lineal Múltiple: Ventas reales vs predichas"

            )


            st.plotly_chart(

                figure2,

                use_container_width=True

            )


        else:

            st.warning(
                "Selecciona mínimo dos variables independientes."
            )
