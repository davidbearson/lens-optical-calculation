import streamlit as st
import numpy as np
import math
from itertools import cycle
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

if "alternative_checkbox" not in st.session_state:
    st.session_state.alternative_checkbox = False

if "mode" not in st.session_state:
    st.session_state.mode = "鍍膜反射率計算"

if "show_result" not in st.session_state:
    st.session_state.show_result = False

if "refresh" not in st.session_state:
    st.session_state.refresh = False

def reset():
    st.session_state.show_result = False

with st.container(border=True):
    st.radio("選擇計算", ["鍍膜反射率計算", "光線追蹤", "厚透鏡焦距計算"], key="mode", on_change=reset)

if st.session_state.mode == "鍍膜反射率計算":
    tab1, tab2 = st.tabs(['計算', '公式'])
    with tab1:
        st.header('輸入數據', divider='rainbow')
        with st.container(border=True):
            number_of_layer = st.number_input("層數:", min_value=0, step=1, format="%d", value=0, on_change=reset)

            refractive_index = np.zeros(number_of_layer)
            QWOT = np.zeros(number_of_layer)

        with st.container(border=True):
            center_wavelength = st.number_input("中心波長 (nm):",min_value=float(0), value=float(0), on_change=reset)
            wavelength_mode = st.selectbox('單色波/多色波',['單色波','多色波'], on_change=reset, index=0)
            col1 ,col2 = st.columns(2)
            with col1:
                if wavelength_mode == '單色波':
                    wavelength = st.number_input("分光波長 (nm):",min_value=float(0), value=float(0), disabled=False)
                    wavelength = np.array(wavelength, ndmin=1)
                else:
                    st.number_input("分光波長 (nm):", min_value=float(0), value=float(0), on_change=reset, disabled=True)
            with col2:
                if wavelength_mode == '多色波':
                    wavelength = st.slider('範圍 (nm)', min_value=200, max_value=1500, value=(380,700), step=1, disabled=False)
                    wavelength = np.arange(wavelength[0], wavelength[1]+1)
                else:
                    st.slider('範圍 (nm)', min_value=200, max_value=1500, value=(380, 700), step=1, disabled=True)

        with st.container(border=True):
            N_s = st.number_input("基板折射率:", min_value=float(0), value=float(0), format="%.4f", on_change=reset)
            st.checkbox("交替材料?", key="alternative_checkbox")
            if st.session_state.alternative_checkbox:
                with st.container(border=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        refractive_index1 = st.number_input("第1種材料折射率:", min_value=float(0), value=float(0), format="%.4f", on_change=reset)
                    with col2:
                        refractive_index2 = st.number_input("第2種材料折射率:", min_value=float(0), value=float(0), format="%.4f", on_change=reset)
                    alternative_refractive_index = cycle([refractive_index1, refractive_index2])
            for i in range(number_of_layer):
                if i % 5 == 0:
                    if i + 5 > number_of_layer:
                        end = number_of_layer
                    else:
                        end = i + 5
                    if i == 0:
                        with st.expander(f"層數{i+1} - {end}", expanded=True):
                            material_col1, material_col2 = st.columns(2)
                    else:
                        with st.expander(f"層數{i+1} - {end}", expanded=False):
                            material_col1, material_col2 = st.columns(2)
                with material_col1:
                    if st.session_state.alternative_checkbox is True:
                        refractive_index[i] = st.number_input(f"第{i+1}層折射率:",min_value=float(0), value=next(alternative_refractive_index), format="%.4f", disabled=True, on_change=reset)
                    else:
                        refractive_index[i] = st.number_input(f"第{i+1}層折射率:", min_value=float(0), value=float(0),
                                                              format="%.4f", on_change=reset)
                with material_col2:
                        QWOT[i] = st.number_input(f"第{i + 1}層QWOT:", min_value=float(0), value=float(0), format="%.2f", on_change=reset)
            def analyse():
                st.session_state.show_result = True

            if np.any((refractive_index == 0) | (QWOT == 0)) or N_s == 0:
                st.button("計算", on_click=analyse, disabled=True, type="primary")
            else:
                st.button("計算", on_click=analyse, disabled=False, type="primary")
        st.header('結果', divider='rainbow')
        result_container = st.container(border=True)
    with tab2:
        with st.container(border=True):
            col1 ,col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("1. Let the refractive index of the $i$ layer be $n_{i}$.")
                    st.latex(r'''
                    \rho_i = \frac{n_{i+1}-n_i}{n_{i+1}+n_i}
                    ''')
            with col2:
                with st.expander('Example: 3 layers', expanded=True):
                    st.latex(r'''
                    \begin{split}
                    \rho_3 &= \frac{1-n_3}{1+n_3},\quad \text{1 because air}\\
                    \rho_2 &= \frac{n_3-n_2}{n_3+n_2}\\
                    \rho_1 &= \frac{n_2-n_1}{n_2+n_1}\\
                    \rho_0 &= \frac{n_1-n_s}{n_1+n_s}
                    \end{split}
                    ''')
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("2. Let the QWOT of the $i$ layer be $\\text{QWOT}_{i}$ and let the total number of layers be {k}.")
                    st.latex(r'''
                    \begin{cases}
                    \delta_{k+1} = 0\\
                    \delta_i = \dfrac{\text{中心波長}\times\text{QWOT}_{i}}{4\times分光波長}\times 2\pi
                    \end{cases}
                    ''')
            with col2:
                with st.expander('Example: 3 layers', expanded=True):
                    st.latex(r'''
                    \begin{split}
                    \delta_4 &= 0\\
                    \delta_3 &= \dfrac{\text{中心波長}\times\text{QWOT}_{3}}{4\times分光波長}\times 2\pi\\
                    \delta_2 &= \dfrac{\text{中心波長}\times\text{QWOT}_{2}}{4\times分光波長}\times 2\pi\\
                    \delta_1 &= \dfrac{\text{中心波長}\times\text{QWOT}_{1}}{4\times分光波長}\times 2\pi
                    \end{split}
                    ''')
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("4. Real part")
                    st.latex(r'''
                    x_i = \rho_i\cos\left(-2\cdot\sum_{j=1}^{k+1-i}\delta_j\right)
                    ''')
            with col2:
                with st.expander('Example: 3 layers', expanded=True):
                    st.latex(r'''
                    \begin{split}
                    x_3 &= \rho_3\cos(-2\cdot\delta_4)\\
                    x_2 &= \rho_2\cos[-2\cdot(\delta_4+\delta_3)]\\
                    x_1 &= \rho_1\cos[-2\cdot(\delta_4+\delta_3+\delta_2)]\\
                    x_0 &= \rho_0\cos[-2\cdot(\delta_4+\delta_3+\delta_2+\delta_1)]\\
                    \end{split}
                    ''')
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("5. Imaginary part")
                    st.latex(r'''
                    y_i = \rho_i\sin\left(-2\cdot\sum_{j=1}^{k+1-i}\delta_j\right)
                    ''')
            with col2:
                with st.expander('Example: 3 layers', expanded=True):
                    st.latex(r'''
                    \begin{split}
                    y_3 &= \rho_3\sin(-2\cdot\delta_4)\\
                    y_2 &= \rho_2\sin[-2\cdot(\delta_4+\delta_3)]\\
                    y_1 &= \rho_1\sin[-2\cdot(\delta_4+\delta_3+\delta_2)]\\
                    y_0 &= \rho_0\sin[-2\cdot(\delta_4+\delta_3+\delta_2+\delta_1)]\\
                    \end{split}
                    ''')
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("6.")
                    st.latex(r'''
                       R = \left(\sum_{k}^{j=0}x_j\right)^2 + \left(\sum_{k}^{j=0}y_j\right)^2
                       ''')
            with col2:
                with st.expander('Example: 3 layers', expanded=True):
                    st.latex(r'''
                       R = (x_0+x_1+x_2+x_3)^2 + (y_0+y_1+y_2+y_3)^2
                       ''')
elif st.session_state.mode == "光線追蹤":
    tab1, tab2 = st.tabs(['Calculation', 'Formula'])
    with tab1:
        st.header('Input', divider='rainbow')
        with st.container(border=True):
            with st.container(border=True):
                number_of_surface = st.number_input("Number of surface:", min_value=1, step=1, format="%d", value=1, on_change=reset, help="輸入資料後如果沒有按下「計算」而改變鏡片數目，會導致輸入資料沒有更新!")

                Q = np.zeros(number_of_surface)
                Q_prime = np.zeros(number_of_surface)
                U = np.zeros(number_of_surface)
                U_prime = np.zeros(number_of_surface)
                I = np.zeros(number_of_surface)
                I_prime = np.zeros(number_of_surface)

                col1, col2 = st.columns(2)
                with col1:
                    Q[0] = st.number_input("Q:", min_value=float(0), format="%.4f", value=float(0), on_change=reset)
                with col2:
                    U[0] = math.radians(st.number_input("U:", min_value=float(0), format="%.4f", value=float(0), on_change=reset))
            with st.container(border=True):
                st.header('Data of surface', divider='rainbow')

                if 'data_editor' not in st.session_state:
                    data = np.zeros([number_of_surface, 4])
                    data[-1][3] = None
                    data = pd.DataFrame(np.insert(data, 0, np.array([i + 1 for i in range(number_of_surface)]), axis=1),
                                        columns=["surface", "R", "is_inf", "n", "d"])
                    data['is_inf'] = data['is_inf'].astype(bool)
                    data['surface'] = data['surface'].astype(int)

                    st.session_state['data_editor'] = data

                if len(st.session_state['data_editor'].index) == number_of_surface:
                    pass
                else:
                    if len(st.session_state['data_editor'].index) < number_of_surface:
                        st.session_state['data_editor'].loc[st.session_state['data_editor']["d"].isnull(), "d"] = 0
                        original_length = len(st.session_state['data_editor'].index)
                        different = number_of_surface - original_length
                        extra_data = np.zeros([different, 4])
                        extra_data[-1][3] = None
                        extra_data = pd.DataFrame(np.insert(extra_data, 0, np.array([i + 1 + original_length for i in range(different)]), axis=1),
                                            columns=["surface", "R", "is_inf", "n", "d"])
                        extra_data['is_inf'] = extra_data['is_inf'].astype(bool)
                        extra_data['surface'] = extra_data['surface'].astype(int)

                        st.session_state['data_editor'] = pd.concat([st.session_state['data_editor'],extra_data], ignore_index=True)
                    elif len(st.session_state['data_editor'].index) > number_of_surface:
                        different = len(st.session_state['data_editor'].index) - number_of_surface
                        st.session_state['data_editor'].drop(st.session_state['data_editor'].tail(different).index, inplace=True)
                        st.session_state['data_editor'].at[st.session_state['data_editor'].index[-1], 'd'] = None

                column_setting = {
                    "surface": st.column_config.NumberColumn(
                        "Number of surface",
                        disabled=True
                    ),
                    "R": st.column_config.NumberColumn(
                        "R",
                        required=False,
                        default=0,
                        format="%.4f"
                    ),
                    "is_inf": st.column_config.CheckboxColumn(
                        "平面?",
                        required=True,
                        default=False,
                    ),
                    "n": st.column_config.NumberColumn(
                        "n",
                        required=False,
                        default=0,
                        format="%.8f"
                    ),
                    "d": st.column_config.NumberColumn(
                        "d",
                        required=False,
                        default=0,
                        format="%.4f"
                    )
                }

                if not st.session_state['data_editor'].empty:
                    data_editor = st.data_editor(st.session_state['data_editor'], use_container_width=True, column_config=column_setting, num_rows="fixed", hide_index=True, on_change=reset)

                if not data_editor['is_inf'].equals(st.session_state['data_editor']['is_inf']):
                    data_editor.loc[(data_editor["is_inf"] == False) & (data_editor["R"].isna()), "R"] = 0
                    data_editor.loc[data_editor["is_inf"] == True, "R"] = None
                    st.session_state['data_editor'] = data_editor
                    st.rerun()

                def analyse():
                    st.session_state.show_result = True
                    st.session_state.refresh = True

                if (data_editor[['R', 'n']] == 0).any().any() or data_editor['d'].iloc[:-1].isna().any():
                    st.button("計算", on_click=analyse, disabled=True, type="primary")
                else:
                    st.button("計算", on_click=analyse, disabled=False, type="primary")

            if st.session_state.refresh:
                if not st.session_state['data_editor'].equals(data_editor):
                    st.session_state['data_editor'] = data_editor
                    st.rerun()
                st.session_state.refresh = False

        st.header('Results', divider='rainbow')
        result_container = st.container(border=True)
    with tab2:
        with st.container(border=True):
            st.markdown("1.")
            st.latex(r'''
            Q=\begin{cases}
                Q=Q,\quad\text{first surface}\\
                Q = Q' + d\sin{U'},\quad\text{otherwise}
            \end{cases}
            ''')
        with st.container(border=True):
            col1 ,col2 = st.columns(2)
            with col1:
                st.markdown("- If $R$ is finite:")
                with st.container(border=True):
                    st.markdown("2.")
                    st.latex(r'''
                    \sin{I} = \frac{Q}{R} + \sin{U}
                    ''')
                with st.container(border=True):
                    st.markdown("3.")
                    st.latex(r'''
                    \sin{I'} = \frac{n}{n'}\sin{I}
                    ''')
                with st.container(border=True):
                    st.markdown("4.")
                    st.latex(r'''
                    U' = U + I' - I
                    ''')
                with st.container(border=True):
                    st.markdown("5.")
                    st.latex(r'''
                    Q' = R(\sin{I'}-\sin{U'})
                    ''')
            with col2:
                st.markdown("- If $R$ is infinite:")
                with st.container(border=True):
                    st.markdown("2.")
                    st.latex(r'''
                    \sin{U'} = \frac{n}{n'}\sin{U}
                    ''')
                with st.container(border=True):
                    st.markdown("3.")
                    st.latex(r'''
                    Q' = Q\frac{\cos{U'}}{\cos{U}}
                    ''')
        with st.container(border=True):
            st.markdown("6. Repeat step 1")
elif st.session_state.mode == "厚透鏡焦距計算":
    tab1, tab2 = st.tabs(['計算', '公式'])
    with tab1:
        st.header('輸入數據', divider='rainbow')
        with st.container(border=True):
            with st.container(border=True):
                number_of_lens = st.number_input("鏡片數目:", min_value=1, step=1, format="%d", value=1, on_change=reset)
            with st.container(border=True):
                if 'data_editor_focus' not in st.session_state:
                    data = np.zeros([number_of_lens, 7])
                    data[-1][6] = None
                    data = pd.DataFrame(np.insert(data, 0, np.array([i + 1 for i in range(number_of_lens)]), axis=1),
                                        columns=["lens", "R1", "is_R1_inf", "R2", "is_R2_inf", "n", "t", 'd'])
                    data['is_R1_inf'] = data['is_R2_inf'].astype(bool)
                    data['is_R2_inf'] = data['is_R2_inf'].astype(bool)
                    data['lens'] = data['lens'].astype(int)

                    st.session_state['data_editor_focus'] = data

                if len(st.session_state['data_editor_focus'].index) == number_of_lens:
                    pass
                else:
                    if len(st.session_state['data_editor_focus'].index) < number_of_lens:
                        original_lens = len(st.session_state['data_editor_focus'].index)
                        different = number_of_lens - original_lens
                        extra_data = np.zeros([different, 7])
                        extra_data[-1][6] = None
                        extra_data = pd.DataFrame(np.insert(extra_data, 0, np.array([i + 1 + original_lens for i in range(different)]), axis=1),
                                            columns=["lens", "R1", "is_R1_inf", "R2", "is_R2_inf", "n", "t", 'd'])
                        extra_data['is_R1_inf'] = extra_data['is_R1_inf'].astype(bool)
                        extra_data['is_R2_inf'] = extra_data['is_R2_inf'].astype(bool)
                        extra_data['lens'] = extra_data['lens'].astype(int)

                        st.session_state['data_editor_focus'] = pd.concat([st.session_state['data_editor_focus'],extra_data], ignore_index=True)
                    elif len(st.session_state['data_editor_focus'].index) > number_of_lens:
                        different = len(st.session_state['data_editor_focus'].index) - number_of_lens
                        st.session_state['data_editor_focus'].drop(st.session_state['data_editor_focus'].tail(different).index, inplace=True)
                        st.session_state['data_editor_focus'].at[st.session_state['data_editor_focus'].index[-1], 'd'] = None

                column_setting = {
                    "lens": st.column_config.NumberColumn(
                        "Number of lens",
                        disabled=True
                    ),
                    "R1": st.column_config.NumberColumn(
                        "R1",
                        required=False,
                        default=0,
                        format="%.4f"
                    ),
                    "is_R1_inf": st.column_config.CheckboxColumn(
                        "R1平面?",
                        required=True,
                        default=False,
                    ),
                    "R2": st.column_config.NumberColumn(
                        "R2",
                        required=False,
                        default=0,
                        format="%.4f"
                    ),
                    "is_R2_inf": st.column_config.CheckboxColumn(
                        "R2平面?",
                        required=True,
                        default=False,
                    ),
                    "n": st.column_config.NumberColumn(
                        "折射率",
                        required=True,
                        default=0,
                        format="%.8f"
                    ),
                    "t": st.column_config.NumberColumn(
                        "中心厚度",
                        required=True,
                        default=0,
                        format="%.4f"
                    ),
                    "d": st.column_config.NumberColumn(
                        "下一個鏡片距離",
                        required=False,
                        default=0,
                        format="%.4f"
                    )
                }

                if not st.session_state['data_editor_focus'].empty:
                    data_editor = st.data_editor(st.session_state['data_editor_focus'], use_container_width=True, column_config=column_setting, num_rows="fixed", hide_index=True, on_change=reset)

                if not data_editor['is_R1_inf'].equals(st.session_state['data_editor_focus']['is_R1_inf']):
                    data_editor.loc[(data_editor["is_R1_inf"] == False) & (data_editor["R1"].isna()), "R1"] = 0
                    data_editor.loc[data_editor["is_R1_inf"] == True, "R1"] = None
                    st.session_state['data_editor_focus'] = data_editor
                    st.rerun()

                if not data_editor['is_R2_inf'].equals(st.session_state['data_editor_focus']['is_R2_inf']):
                    data_editor.loc[(data_editor["is_R2_inf"] == False) & (data_editor["R2"].isna()), "R2"] = 0
                    data_editor.loc[data_editor["is_R2_inf"] == True, "R2"] = None
                    st.session_state['data_editor_focus'] = data_editor
                    st.rerun()

                def analyse():
                    st.session_state.show_result = True
                    st.session_state.refresh = True

                if (data_editor[['R1', 'R2', 'n', 't']] == 0).any().any() or data_editor['d'].iloc[:-1].isna().any():
                    st.button("計算", on_click=analyse, disabled=True, type="primary")
                else:
                    st.button("計算", on_click=analyse, disabled=False, type="primary")

            if st.session_state.refresh:
                if not st.session_state['data_editor_focus'].equals(data_editor):
                    st.session_state['data_editor_focus'] = data_editor
                    st.rerun()
                st.session_state.refresh = False
        st.header('結果', divider='rainbow')
        result_container = st.container(border=True)
    with tab2:
        col1 ,col2 = st.columns(2)
        with col1:
            st.header('單鏡片', divider='rainbow')
            with st.expander('EFL', expanded=True):
                st.markdown("- Let EFL be f.")
                st.latex(r'''
                \frac{1}{\text{f}} = (n-1)\left(\frac{1}{R_1}-\frac{1}{R_2}+\frac{t(n-1)}{nR_1R_2}\right)
                ''')
            with st.expander('BFL', expanded=True):
                st.latex(r'''
                \text{BFL} = f\left(1-\frac{t(n-1)}{R_1n}\right)
                ''')
            with st.expander('FEL', expanded=True):
                st.latex(r'''
                \text{FEL} = f\left(1-\frac{t(n-1)}{R_2n}\right)
                ''')
            with st.expander('A1H', expanded=True):
                st.latex(r'''
                A_1H = \frac{-tR_1}{n(R_2-R_1)+t(n-1)}
                ''')
            with st.expander('A2H', expanded=True):
                st.latex(r'''
                A_2H = \frac{-tR_2}{n(R_2-R_1)+t(n-1)}
                ''')
        with col2:
            st.header('兩鏡片合併', divider='rainbow')
            st.markdown("- Let EFL of the first lens be $f_1$ and EFL of the second lens be $f_1$.")
            st.markdown("- Let $A_2H$ of first lens be $x$ and $A_1H$ of second lens be $y$.")
            st.markdown("- D is the displacement from $x$ to $y$.")
            with st.expander('EFl', expanded=True):
                st.latex(r'''
                D=\begin{cases}
                -x+d+y,\quad\text{兩片}\\
                -z-x+d+y,\quad\text{三片以上}
                \end{cases}
                ''')
                st.latex(r'''
                f = \frac{f_1f_2}{f_1+f_2-D}
                ''')
            with st.expander("S\"", expanded=True):
                st.latex(r'''
                S_2" = \frac{f_2(f_1-D)}{f_1+f_2-D}
                ''')
            with st.expander("Z", expanded=True):
                st.latex(r'''
                Z = S_2" - f
                ''')
            with st.expander("BEF", expanded=True):
                st.latex(r'''
                \text{BEL} = S_2" + A_2H
                ''')
def main():
    if st.session_state.mode == "鍍膜反射率計算":
        magnitude = np.zeros(number_of_layer + 1)
        for i in range(number_of_layer - 1, -1, -1):
            if i == number_of_layer - 1:
                magnitude[i + 1] = (1 - refractive_index[i]) / (1 + refractive_index[i])
            else:
                magnitude[i + 1] = (refractive_index[i + 1] - refractive_index[i]) / (
                            refractive_index[i + 1] + refractive_index[i])
        magnitude[0] = (refractive_index[0] - N_s) / (refractive_index[0] + N_s)

        magnitude = np.tile(magnitude,(len(wavelength), 1))

        angle = np.zeros([len(wavelength),number_of_layer + 1])
        for j in range(len(wavelength)):
            for i in range(number_of_layer):
                angle[j][i] = 2 * math.pi * center_wavelength * QWOT[i] / 4 / wavelength[j]

        phase = np.zeros([len(wavelength),number_of_layer + 1])
        for j in range(len(wavelength)):
            for i in range(number_of_layer - 1, -1, -1):
                phase[j][i] = phase[j][i + 1] + angle[j][i]

        reflection = magnitude * np.exp(1j * -2 * phase)
        if wavelength_mode == '單色波':
            total_reflection = np.sum(reflection)
        else:
            total_reflection = np.sum(reflection, axis=1)
        total_reflection = np.abs(total_reflection) ** 2 * 100

        with st.container(border=True):
            with result_container:
                if wavelength_mode == '多色波':
                    co1, col2, col3 = st.columns(3)
                    with col2:
                        fig, ax = plt.subplots(nrows=1, ncols=1)
                        ax.plot(wavelength, total_reflection)
                        ax.set_title('Reflectivity of coating')
                        ax.set_xlabel('Wavelength (nm)')
                        ax.set_ylabel('Reflectivity (%)')
                        st.pyplot(fig, use_container_width=True)
                with st.expander("詳細數值", expanded=True):
                    if wavelength_mode == '單色波':
                        st.number_input("波長:", format="%.4f", value=np.min(wavelength), disabled=True)
                        magnitude_analysed = magnitude
                        angle_analysed = angle
                        reflection_analysed = reflection
                        total_reflection_analysed = total_reflection
                    else:
                        analysed_wavelength = st.number_input("波長:", min_value=float(np.min(wavelength)),max_value=float(np.max(wavelength)), format="%.4f", value=float(np.min(wavelength)))
                        magnitude_analysed = magnitude[0]

                        angle_analysed = np.zeros_like(magnitude_analysed)
                        for i in range(number_of_layer):
                            angle_analysed[i] = 2 * math.pi * center_wavelength * QWOT[i] / 4 / analysed_wavelength

                        phase_analysed = np.zeros_like(magnitude_analysed)
                        for i in range(number_of_layer - 1, -1, -1):
                            phase_analysed[i] = phase_analysed[i + 1] + angle_analysed[i]

                        reflection_analysed = magnitude_analysed * np.exp(1j * -2 * phase_analysed)
                        total_reflection_analysed = np.sum(reflection_analysed)
                        total_reflection_analysed = np.abs(total_reflection_analysed) ** 2 * 100

                    st.markdown(f"- 反射率: {total_reflection_analysed}%")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.markdown("- 層數")
                    with col2:
                        st.markdown(r"- $\rho$")
                    with col3:
                        st.markdown("- $\delta$")
                    with col4:
                        st.write("- x")
                    with col5:
                        st.write("- y")

                    result = pd.DataFrame({
                        'Number of layer': [i for i in range(number_of_layer+1)],
                        'magnitude': np.squeeze(magnitude_analysed),
                        "phase": np.squeeze(angle_analysed),
                        'x': np.squeeze(reflection_analysed.real),
                        "y": np.squeeze(reflection_analysed.imag)
                    })
                    result.at[0, 'Number of layer'] = '基板'
                    result_setting = {
                        "Number of layer": st.column_config.Column(
                            "Number of layer",
                            disabled=True
                        ),
                        "magnitude": st.column_config.NumberColumn(
                            "magitude",
                            format="%.20f",
                            disabled=True
                        ),
                        "phase": st.column_config.NumberColumn(
                            "phase",
                            format="%.20f",
                            disabled=True
                        ),
                        "x": st.column_config.NumberColumn(
                            "x",
                            format="%.20f",
                            disabled=True
                        ),
                        "y": st.column_config.NumberColumn(
                            "y",
                            format="%.20f",
                            disabled=True
                        )
                    }
                    st.dataframe(result, use_container_width=True, hide_index=True, column_config=result_setting)
    elif st.session_state.mode == "光線追蹤":
        sur = st.session_state['data_editor']['surface'].values.copy()
        is_plane = st.session_state['data_editor']['is_inf'].values.copy()
        R = st.session_state['data_editor']['R'].values.copy()
        d = st.session_state['data_editor']['d'].values.copy()
        n = st.session_state['data_editor']['n'].values.copy()
        n = np.insert(n, 0, 1)
        for i in range(number_of_surface):
            if is_plane[i]:
                I[i] = U[i]
                U_prime[i] = math.asin(math.sin(U[i]) * n[i] / n[i + 1])
                I_prime[i] = U_prime[i]
                Q_prime[i] = Q[i] * math.cos(U_prime[i]) / math.cos(U[i])
                if i != number_of_surface - 1:
                    Q[i + 1] = Q_prime[i] + d[i] * math.sin(U_prime[i])
                    U[i + 1] = U_prime[i]
            else:
                I[i] = math.asin(Q[i]/R[i]+math.sin(U[i]))
                I_prime[i] = math.asin(math.sin(I[i])*n[i]/n[i+1])
                U_prime[i] = U[i] + I_prime[i] - I[i]
                Q_prime[i] = R[i]*(math.sin(I_prime[i])-math.sin(U_prime[i]))
                if i != number_of_surface - 1:
                    Q[i+1] = Q_prime[i] + d[i]*math.sin(U_prime[i])
                    U[i+1] = U_prime[i]

        result = pd.DataFrame({
            's': sur,
            'Q': Q,
            "Q'": Q_prime,
            'U': np.rad2deg(U),
            "U'": np.rad2deg(U_prime),
            'I': np.rad2deg(I),
            "I'": np.rad2deg(I_prime)
        })
        result_setting = {
            "s": st.column_config.NumberColumn(
                "Number of surface",
                format="%d",
                disabled=True
            ),
            "Q": st.column_config.NumberColumn(
                "Q",
                format="%.20f",
                disabled=True
            ),
            "Q'": st.column_config.NumberColumn(
                "Q'",
                format="%.20f",
                disabled=True
            ),
            "I": st.column_config.NumberColumn(
                "I",
                format="%.20f",
                disabled=True
            ),
            "I'": st.column_config.NumberColumn(
                "I'",
                format="%.20f",
                disabled=True
            ),
            "U": st.column_config.NumberColumn(
                "U",
                format="%.20f",
                disabled=True
            ),
            "U'": st.column_config.NumberColumn(
                "U'",
                format="%.20f",
                disabled=True
            ),
            "I": st.column_config.NumberColumn(
                "I",
                format="%.20f",
                disabled=True
            ),
            "I'": st.column_config.NumberColumn(
                "I'",
                format="%.20f",
                disabled=True
            )
        }

        with result_container:
            st.dataframe(result, use_container_width=True, hide_index=True, column_config=result_setting)
    elif st.session_state.mode == "厚透鏡焦距計算":
        lens = st.session_state['data_editor_focus']['lens'].values.copy()
        R1 = st.session_state['data_editor_focus']['R1'].values.copy()
        is_R1_plane = st.session_state['data_editor_focus']['is_R1_inf'].values.copy()
        R2 = st.session_state['data_editor_focus']['R2'].values.copy()
        is_R2_plane = st.session_state['data_editor_focus']['is_R2_inf'].values.copy()
        n = st.session_state['data_editor_focus']['n'].values.copy()
        t = st.session_state['data_editor_focus']['t'].values.copy()
        d = st.session_state['data_editor_focus']['d'].values.copy()

        EFT = np.zeros(number_of_lens)
        EFT_inverse = np.zeros(number_of_lens)
        BFL = np.zeros(number_of_lens)
        FEL = np.zeros(number_of_lens)
        A_1H = np.zeros(number_of_lens)
        A_2H = np.zeros(number_of_lens)

        D = np.zeros(number_of_lens)
        combined_EFL = np.zeros(number_of_lens)
        s_2 = np.zeros(number_of_lens)
        z = np.zeros(number_of_lens)
        combined_BFL = np.zeros(number_of_lens)

        for i in range(number_of_lens):
            if is_R1_plane[i]:
                R1[i] = float('inf')
            if is_R2_plane[i]:
                R2[i] = float('inf')

            EFT_inverse[i] = (n[i]-1)*(1/R1[i]-1/R2[i]+(t[i]*(n[i]-1))/(n[i]*R1[i]*R2[i]))

            if EFT_inverse[i] == 0:
                EFT[i] = float('inf')
            else:
                EFT[i] = 1/EFT_inverse[i]

            BFL[i] = EFT[i]*(1-t[i]/R1[i]*(n[i]-1)/n[i])
            FEL[i] = EFT[i]*(1+t[i]/R2[i]*(n[i]-1)/n[i])
            A_1H[i] = -t[i]/(n[i]*(R2[i]/R1[i]-1)+t[i]*(n[i]-1)/R1[i])
            A_2H[i] = -t[i]/(n[i]*(1-R1[i]/R2[i])+t[i]*(n[i]-1)/R2[i])

            if i != 0:

                D[i] = - z[i-1] - A_2H[i-1] + d[i-1] + A_1H[i]

                if i == 1:
                    combined_EFL[i] = 1/(1/EFT[i-1]+1/EFT[i]-D[i]/(EFT[i-1]*EFT[i]))
                else:
                    combined_EFL[i] = 1 / (1 / combined_EFL[i - 1] + 1 / EFT[i] - D[i] / (combined_EFL[i - 1] * EFT[i]))

                if i == 1:
                    s_2[i] = (1-D[i]/EFT[i-1])/(1/EFT[i-1]+1/EFT[i]-D[i]/(EFT[i-1]*EFT[i]))
                else:
                    s_2[i] = (1 - D[i] / combined_EFL[i - 1]) / (
                                1 / combined_EFL[i - 1] + 1 / EFT[i] - D[i] / (combined_EFL[i - 1] * EFT[i]))

                z[i] = s_2[i] - combined_EFL[i]
                combined_BFL[i] = s_2[i] + A_2H[i]

        D[0] = None
        combined_EFL[0] = None
        s_2[0] = None
        z[0] = None
        combined_BFL[0] = None

        result_single_lens = pd.DataFrame({
            'lens': lens,
            'EFT': EFT,
            "BFL": BFL,
            'FEL': FEL,
            'A_1H': A_1H,
            'A_2H': A_2H
        })

        result_combined_lens = pd.DataFrame({
            'lens': lens,
            'D': D,
            'F': combined_EFL,
            "S2\"": s_2,
            "Z": z,
            'BFL': combined_BFL
        })

        result_single_lens = result_single_lens.replace(float('inf'), "infinite")
        result_single_lens = result_single_lens.fillna("Not well-defined")

        result_combined_lens = result_combined_lens.replace(float('inf'), "infinite")

        result_single_lens_setting = {
            "lens": st.column_config.NumberColumn(
                "Number of surface",
                format="%d",
                disabled=True
            ),
            "EFT": st.column_config.NumberColumn(
                "EFT",
                format="%.20f",
                disabled=True
            ),
            "BFL": st.column_config.NumberColumn(
                "BFL",
                format="%.20f",
                disabled=True
            ),
            "FEL": st.column_config.NumberColumn(
                "FEL",
                format="%.20f",
                disabled=True
            ),
            "A_1H": st.column_config.NumberColumn(
                "A_1H",
                format="%.20f",
                disabled=True
            ),
            "A_2H": st.column_config.NumberColumn(
                "A_2H",
                format="%.20f",
                disabled=True
            )
        }
        with result_container:
            with st.expander("單片鏡片數值",expanded=True):
                st.dataframe(result_single_lens, use_container_width=True, hide_index=True, column_config=result_single_lens_setting)

            result_combined_lens_setting = {
                "lens": st.column_config.NumberColumn(
                    "加上鏡片",
                    format="%d",
                    disabled=True
                ),
                "D": st.column_config.NumberColumn(
                    "A2H至A1H的距離",
                    format="%.20f",
                    disabled=True
                ),
                "F": st.column_config.NumberColumn(
                    "合併EFL",
                    format="%.20f",
                    disabled=True
                ),
                "S2\"": st.column_config.NumberColumn(
                    "S2\"",
                    format="%.20f",
                    disabled=True
                ),
                "z": st.column_config.NumberColumn(
                    "Z",
                    format="%.20f",
                    disabled=True
                ),
                "BFL": st.column_config.NumberColumn(
                    "合併BFL",
                    format="%.20f",
                    disabled=True
                )
            }

            with st.expander("鏡片合併數值",expanded=True):
                st.dataframe(result_combined_lens, use_container_width=True, hide_index=True,
                             column_config=result_combined_lens_setting)
if st.session_state.show_result:
    main()