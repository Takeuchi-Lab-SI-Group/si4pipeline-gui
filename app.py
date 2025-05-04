import pandas as pd
import streamlit as st
from barfi import barfi_schemas, st_barfi
from blocks.base_blocks import get_base_blocks
from blocks.load_blocks import get_dataset_loader_blocks
from blocks.test_blocks import get_test_blocks

import si4pipeline as plp

base_blocks = get_base_blocks(plp)
load_bloks = get_dataset_loader_blocks(st, plp)
test_blocks = get_test_blocks(st)

base_blocks.extend(test_blocks)
base_blocks.extend(load_bloks)


def main():
    st.set_page_config(page_icon="🍍", page_title="SI4PIPELINE", layout="wide")

    st.title("SI4PIPELINE")

    st.sidebar.title("Settings")
    cv = st.sidebar.slider("Number of folds in cross-validation:", 0, 10, 5)
    if cv not in st.session_state:
        st.session_state.cv = cv

    # load data
    st.header("STEP1: Upload data")
    _, col1, col2 = st.columns([1, 8, 7])
    with col1:
        uploaded_file = st.file_uploader("Upload your own data", type="csv")
        if uploaded_file is not None:
            header_exists = st.checkbox("The file have a header", value=True)
            if header_exists:
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_csv(uploaded_file, header=None)
            default_target_column = data.columns[-1]
            target_column = st.text_input(
                "Target column name:", value=default_target_column
            )
            y = data[target_column].values
            X = data.drop(columns=[target_column]).values
            features = data.drop(columns=[target_column]).columns
            st.session_state.dataset = "uploaded"
            st.session_state.uploaded_dataset = [X, y, features]
    with col2:
        # if st.checkbox('or select existing dataset'):
        if uploaded_file is None:
            existing_data_options = [
                "-",
                "prostate_cancer",
                "random",
                "red_wine",
                "concrete",
                "abalone",
            ]
            # デモ用のデータセットを選択
            selected_dataset = st.selectbox(
                "Or select a demo dataset:", existing_data_options
            )
            if selected_dataset != "-":
                st.session_state.dataset = selected_dataset
    if uploaded_file:
        _, col1 = st.columns([1, 15])
        with col1:
            with st.expander("Show data"):
                st.dataframe(data, height=300)

    # load and define pipeline
    st.header("STEP2: Define and execute pipeline")
    _, col1, col2 = st.columns([1, 8, 7])
    with col1:
        st.write("Define your data processing pipeline")
        # st.write('You can create blocks by right-clicking and connect them to create a pipeline.')
        # st.write('You can also set parameters for each block.')
        # st.write('After defining the pipeline, click the "Execute" button to perform the analysis.')
        # st.write('The results will be displayed in the next section.')
    with col2:
        load_pipeline = st.selectbox(
            "Or select a pre-defined pipeline:", barfi_schemas()
        )

    _, col1 = st.columns([1, 15])

    with col1:
        barfi_result = st_barfi(
            base_blocks=base_blocks, compute_engine=True, load_schema=load_pipeline
        )

    # inference results
    st.header("STEP3: Inference results")
    _, col1 = st.columns([1, 15])
    with col1:
        if "results_df" not in st.session_state:
            if "executed" in st.session_state:
                st.error(
                    "The analysis has failed.\n\
                         Please check the pipeline structure and the dataset format."
                )
            else:
                st.write("(No analysis has been performed yet.)")
        else:

            def highlight_significant(row):
                color = (
                    "background-color: green"
                    if row["Significance"] == "significant"
                    else ""
                )
                return [color] * len(row)

            styled_df = st.session_state["results_df"].style.apply(
                highlight_significant, axis=1
            )
            st.dataframe(styled_df)


if __name__ == "__main__":
    main()
