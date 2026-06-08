from __future__ import annotations

import re
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth
from requests.exceptions import RequestException


PROJECT_TITLE = "CAFA-5 MLOps Solution"
PROJECT_DESCRIPTION = (
    "Interactive sequence-to-GO inference UI backed by the embedding and GO "
    "prediction APIs through the NGINX gateway."
)
DEFAULT_GATEWAY_URL = os.getenv("GATEWAY_BASE_URL", "https://localhost")
PREDICT_ENDPOINT = "/api/v1/predict-go-from-sequences"
MAX_TOP_K = 500
AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
WORKFLOW_DOT = """
digraph cafa5 {
    rankdir=LR;
    node [shape=box, style=rounded];
    user [label="User Browser"];
    ui [label="Streamlit UI (/ui/)"];
    gw [label="NGINX Gateway (HTTPS)"];
    api [label="Embedding API"];
    pred [label="GO Prediction API"];

    user -> ui;
    ui -> gw [label="Basic Auth"];
    gw -> api [label="/api/v1/predict-go-from-sequences"];
    api -> pred [label="predict()"];
    pred -> api;
    api -> ui;
    ui -> user;
}
""".strip()


def normalize_sequence(raw_sequence: str) -> str:
    compact = re.sub(r"\s+", "", raw_sequence or "")
    return compact.upper()


def validate_sequence(sequence: str) -> tuple[bool, str]:
    if not sequence:
        return False, "Sequence is empty after whitespace cleanup."
    if not AA_PATTERN.fullmatch(sequence):
        return (
            False,
            "Sequence includes invalid symbols. Allowed amino acids: ACDEFGHIKLMNPQRSTVWY.",
        )
    return True, ""


def parse_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code} returned empty body."
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if detail is not None:
            return str(detail)
    return str(payload)


def build_request_payload(sequence: str, top_k: int) -> dict[str, Any]:
    return {
        "backend": "esm2",
        "pooling": "mean",
        "batch_size": 1,
        "max_length": 1280,
        "top_k": top_k,
        "sequences": [{"id": "input_1", "sequence": sequence}],
    }


def call_prediction_api(
    gateway_base_url: str,
    username: str,
    password: str,
    sequence: str,
    top_k: int,
    verify_tls: bool,
    timeout_seconds: int,
) -> tuple[bool, dict[str, Any] | str]:
    payload = build_request_payload(sequence=sequence, top_k=top_k)
    endpoint = f"{gateway_base_url.rstrip('/')}{PREDICT_ENDPOINT}"

    try:
        response = requests.post(
            endpoint,
            json=payload,
            auth=HTTPBasicAuth(username, password),
            verify=verify_tls,
            timeout=timeout_seconds,
        )
    except RequestException as exc:
        return False, f"Request failed before receiving a response: {exc}"

    if response.status_code != 200:
        return False, f"API returned HTTP {response.status_code}: {parse_error_message(response)}"

    try:
        return True, response.json()
    except ValueError as exc:
        return False, f"API returned non-JSON success response: {exc}"


def render_predictions(payload: dict[str, Any]) -> None:
    st.success("Prediction request completed successfully.")
    model_version = payload.get("model_version") or "unknown"
    st.caption(f"Model version: `{model_version}`")

    results = payload.get("results", [])
    if not results:
        st.warning("No prediction rows were returned.")
        return

    for result in results:
        sequence_id = result.get("sequence_id", "unknown")
        st.markdown(f"#### Sequence `{sequence_id}`")
        predictions = result.get("predictions", [])
        if not predictions:
            st.info("No GO terms returned for this sequence.")
            continue
        table = pd.DataFrame(predictions)
        table = table.rename(columns={"go_term": "GO Term", "score": "Score"})
        st.dataframe(table, use_container_width=True, hide_index=True)

    failures = payload.get("failures", [])
    if failures:
        st.warning("Some sequence predictions failed.")
        st.json(failures)


def main() -> None:
    st.set_page_config(page_title="CAFA-5 UI", page_icon="🧬", layout="wide")
    st.title("🧬 CAFA-5 Sequence-to-GO Prediction UI")
    st.write(PROJECT_DESCRIPTION)

    st.subheader("Platform Links")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("- [MLflow](https://localhost/mlflow/)")
    with c2:
        st.markdown("- [Prometheus](http://localhost:9090)")
    with c3:
        st.markdown("- [Grafana](http://localhost:3000)")

    st.subheader("Workflow")
    st.graphviz_chart(WORKFLOW_DOT, use_container_width=True)

    st.subheader("Predict GO Terms")
    with st.form("predict_form"):
        sequence_text = st.text_area(
            "Protein sequence",
            height=180,
            placeholder="Paste a single protein sequence (FASTA header excluded).",
        )
        top_k = st.number_input("top_k", min_value=1, max_value=MAX_TOP_K, value=10, step=1)
        gateway_base_url = st.text_input("Gateway base URL", value=DEFAULT_GATEWAY_URL)
        username = st.text_input("API username")
        password = st.text_input("API password", type="password")
        verify_tls = st.checkbox("Verify TLS certificates", value=False)
        submit = st.form_submit_button("Run prediction")

    if not submit:
        return

    cleaned = normalize_sequence(sequence_text)
    is_valid, message = validate_sequence(cleaned)
    if not is_valid:
        st.error(message)
        return
    if not gateway_base_url.strip():
        st.error("Gateway base URL is required.")
        return
    if not username.strip() or not password:
        st.error("Both API username and password are required.")
        return

    with st.spinner("Submitting request to /api/v1/predict-go-from-sequences ..."):
        ok, result = call_prediction_api(
            gateway_base_url=gateway_base_url.strip(),
            username=username.strip(),
            password=password,
            sequence=cleaned,
            top_k=int(top_k),
            verify_tls=verify_tls,
            timeout_seconds=600,
        )

    if not ok:
        st.error(str(result))
        return
    render_predictions(result if isinstance(result, dict) else {})


if __name__ == "__main__":
    main()
