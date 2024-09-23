import logging
import os
from typing import Any, Dict, List, Tuple, Union
from angle_emb import AnglE
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from numpy.typing import NDArray
from openai import AzureOpenAI, OpenAI
from sacrebleu import BLEU
from torch import nn
from tqdm import tqdm
import re
from llm_studio.python_configs.base import DefaultConfigProblemBase
from llm_studio.src.datasets.text_utils import get_texts
from llm_studio.src.utils.logging_utils import TqdmToLogger
from scipy import spatial
from rouge import Rouge
logger = logging.getLogger(__name__)


LLM_RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", 3))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 60))

modelAnglE = AnglE.from_pretrained("WhereIsAI/UAE-Large-V1", pooling_strategy="cls", device="cuda:0")
modelAnglE = modelAnglE.cuda()

def date_diff(ref_date, comp_date):
    DATE_NOT_FOUND_CODE = 9999
    if not comp_date:
        return DATE_NOT_FOUND_CODE
    if ref_date.isdigit():
        comp_year = re.findall(r"\b\d{3,4}\b", comp_date)
        if comp_year:
            return abs(int(ref_date) - int(comp_year[0])) * 365
        else:
            return DATE_NOT_FOUND_CODE
    try:
        ref_date = pd.to_datetime(ref_date)
        comp_date = pd.to_datetime(comp_date)
        return abs((ref_date - comp_date).days)
    except Exception as _:
        return DATE_NOT_FOUND_CODE

def parse_dates_from_text(text):
    date_pattern = r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}\b|\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember))\s+\d{4}\b|\b\d{4}\b"
    date_regex = re.compile(date_pattern)
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    for sentence in sentences:
        dates = date_regex.findall(sentence)
        if dates:
            return dates[0]
    return None

def date_score(reference, completion):
    score = 0
    if not completion:
        return score
    ref_date = parse_dates_from_text(reference)
    comp_date = parse_dates_from_text(completion)
    score = np.exp(-(date_diff(ref_date, comp_date) ** 2 / 1000))
    if score < 0.001:
        score = 0
    return score

def relevance_score(reference, completion, model):
    reference_embedding = model.encode(reference, to_numpy=True)
    baseline = 1 - float(spatial.distance.cosine(reference_embedding.flatten(), model.encode("", to_numpy=True).flatten()))
    emb = model.encode(completion, to_numpy=True)
    score = 1 - float(spatial.distance.cosine(reference_embedding.flatten(), emb.flatten() - baseline))
    return score

def rouge_score(reference, completion):
    rouge = Rouge()
    if not completion or not reference:
        return 0.0
    return rouge.get_scores(reference, completion, avg=False)[0]["rouge-l"]["f"]

######################## Final scores ############################################

# Date Q&A score
def date_qa_score(reference, completion) :
    rouge = rouge_score(reference, completion)
    date = date_score(reference, completion)
    return 0.7 * date + 0.3 * rouge

# Multi Choice metric
def multi_choice_score(reference, completion):
    matches = [
        word
        for word in re.sub(r"\W", " ", completion).split()
        if word in ("A", "B", "C", "D")
    ]
    return int(len(matches) > 0 and matches[-1] == reference)

# QA, Summarization and Organic Scoring metric
def organic_synth_score(reference, completion):
    rouge = rouge_score(reference, completion)
    return rouge

# QA, Summarization and Organic Scoring metric
def others_score(reference, completion, modelAnglE):
    relevance = relevance_score(reference, completion, modelAnglE)
    return relevance

def sacrebleu_score(
    cfg: DefaultConfigProblemBase, results: Dict, val_df: pd.DataFrame
) -> NDArray:
    """
    Calculate BLEU scores for predicted texts against target texts.

    This function computes the BLEU score for each pair of predicted and target texts.
    It handles empty target texts by assigning a score of 0.0.
    BLEU scores are given in the range [0.0, 100.0].

    Args:
        cfg: DefaultConfigProblemBase, ignored
        results: Dict, containing 'predicted_text' and 'target_text' lists
        val_df: pd.DataFrame, ignored

    Returns:
        NDArray: An array of BLEU scores for each text pair

    Note:
        - Empty target texts are assigned a score of 0.0
    """
    # Input validation
    if len(results["target_text"]) != len(results["predicted_text"]):
        raise ValueError(
            f"Length of target_text ({len(results['target_text'])}) and predicted_text "
            f"({len(results['predicted_text'])}) should be the same."
        )
    if len(results["target_text"]) == 0:
        raise ValueError("No data to calculate BLEU score")

    scores = []
    for predicted_text, target_text in zip(
        results["predicted_text"], results["target_text"]
    ):
        if target_text == "":
            score = 0.0
        else:
            score = (
                BLEU(effective_order=True)
                .sentence_score(predicted_text, [target_text])
                .score
            )
        scores.append(score)
    return np.array(scores)


def get_openai_client() -> AzureOpenAI | OpenAI:
    if os.getenv("OPENAI_API_TYPE", "open_ai") == "azure":
        endpoint = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        client: AzureOpenAI | OpenAI = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            azure_deployment=os.getenv("OPENAI_API_DEPLOYMENT_ID"),
            # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
            api_version=os.getenv("OPENAI_API_VERSION", "2023-05-15"),
            # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
            azure_endpoint=endpoint,
            max_retries=LLM_RETRY_ATTEMPTS,
            timeout=LLM_TIMEOUT,  # unit is seconds
        )
        logger.info("Using Microsoft Azure Endpoint for OpenAI API")
        logger.info(f"Endpoint: {endpoint}")
    else:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            max_retries=LLM_RETRY_ATTEMPTS,
            timeout=LLM_TIMEOUT,  # unit is seconds
        )
    return client


def call_openai_api(template: str, model: str):
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful and precise assistant "
                "for checking the quality of the answer.",
            },
            {
                "role": "user",
                "content": template,
            },
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    ret = response.choices[0].message.content
    try:
        score = float(ret.split("SCORE:")[-1].split()[0].split("/")[0])
    except ValueError:
        raise ValueError(f"Could not parse score from response: {ret}")
    return score, ret


def rate_reply(filled_eval_template: str, model: str):
    try:
        return call_openai_api(filled_eval_template, model)
    except Exception as e:
        logger.warning(f"Exception caught in api call: {e}")
        return 0.0, ""


def gpt_score(
    cfg: DefaultConfigProblemBase,
    results: Dict,
    val_df: pd.DataFrame,
    raw_results: bool = False,
) -> Union[NDArray, Tuple[NDArray, List[str]]]:
    vdf = val_df.copy()
    vdf["_PROMPT"] = get_texts(val_df, cfg)
    vdf["_PREDICTED_TEXT"] = results["predicted_text"]
    vdf["_TARGET_TEXT"] = results["target_text"]

    model = cfg.prediction.metric_gpt_model
    template_name = cfg.prediction.metric_gpt_template

    if template_name == "mt-bench":
        eval_template = open("prompts/mt-bench/general.txt", "r").read()
    else:
        eval_template = open(f"prompts/{template_name}.txt", "r").read()
    vdf["filled_eval_template"] = eval_template
    if template_name == "mt-bench":
        eval_template = open("prompts/mt-bench/reference.txt", "r").read()
        vdf.loc[
            vdf.category.isin(["math", "reasoning", "coding"]), "filled_eval_template"
        ] = eval_template

    vdf["filled_eval_template"] = vdf.apply(
        lambda row: row["filled_eval_template"].format(**row), axis=1
    )

    ret = Parallel(n_jobs=8, backend="multiprocessing")(
        delayed(rate_reply)(
            filled_eval_template,
            model,
        )
        for filled_eval_template in tqdm(
            vdf["filled_eval_template"].values,
            file=TqdmToLogger(logger, level=logging.INFO),
            desc=f"GPT eval {model} - {template_name}",
            total=len(vdf),
        )
    )
    scores = [x[0] for x in ret]
    explanations = [x[1] for x in ret]

    if template_name == "mt-bench":
        vdf["score"] = scores
        score_by_category = vdf.groupby("category").agg({"score": "mean"}).reset_index()
        logger.info(
            "MT-Bench scores by category:\n" + score_by_category.to_string(index=False)
        )

    if raw_results:
        return np.array(scores), explanations
    return np.mean(scores)


class Perplexity(nn.Module):
    def __init__(self, cfg: DefaultConfigProblemBase, reduce: bool = True):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss()
        self.reduce = reduce

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        perplexity = []
        for i in range(labels.shape[0]):
            perplexity.append(self.loss_fn(shift_logits[i], shift_labels[i]))
        perplexity = torch.stack(perplexity, dim=0)
        perplexity = torch.exp(perplexity)
        if self.reduce:
            perplexity = torch.mean(perplexity)
        return perplexity


def perplexity(cfg: DefaultConfigProblemBase, results: Dict, val_df: pd.DataFrame):
    return results["perplexity"].detach().float().cpu().numpy()

def relevance_metric(results: Dict) -> NDArray:

    predictions = results["predicted_text"]
    labels = results["target_text"]

    scores  = []

    # Calculate metrics for each task and accumulate results
    for pred, label in zip(predictions, labels):
        scores.append(relevance_score(label, pred))


    return np.array(scores)

def multichoice_metric(results: Dict) -> NDArray :

    predictions = results["predicted_text"]
    labels = results["target_text"]

    scores  = []

    # Calculate metrics for each task and accumulate results
    for pred, label in zip(predictions, labels):
        scores.append(multi_choice_score(label, pred))


    return np.array(scores)

class Metrics:
    """
    Metrics factory. Returns:
        - metric value
        - should it be maximized or minimized
        - Reduce function

    Maximized or minimized is needed for early stopping (saving best checkpoint)
    Reduce function to generate a single metric value, usually "mean" or "none"
    """

    _metrics = {
        "Perplexity": (perplexity, "min", "mean"),
        "BLEU": (sacrebleu_score, "max", "mean"),
        "GPT": (gpt_score, "max", "mean"),
        "Relevance": (relevance_metric, "max", "mean"),
        "MultiChoice": (multichoice_metric, "max", "mean"),
    }

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._metrics.keys())

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to Metrics.

        Args:
            name: metrics name
        Returns:
            A class to build the Metrics
        """
        return cls._metrics.get(name, cls._metrics["BLEU"])
